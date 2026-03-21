#!/usr/bin/env python3
"""
Агент: coder-tracking (под governance контролем)
Задача: дообучить модель на новых данных

ВНИМАНИЕ: Этот скрипт требует governance approval перед запуском!
"""

import os
import shutil
import argparse
from pathlib import Path


def prepare_dataset(new_frames_dir: str, new_labels_dir: str, 
                    existing_dataset: str, output_dataset: str):
    """
    Объединяет существующий датасет с новыми размеченными данными.
    
    Args:
        new_frames_dir: папка с новыми кадрами (.jpg)
        new_labels_dir: папка с разметкой новых кадров (.txt)
        existing_dataset: путь к существующему датасету
        output_dataset: путь для объединённого датасета
    """
    
    print("="*60)
    print("ПОДГОТОВКА ДАТАСЕТА ДЛЯ ДООБУЧЕНИЯ")
    print("="*60)
    
    # Создаём структуру
    train_images = os.path.join(output_dataset, 'train', 'images')
    train_labels = os.path.join(output_dataset, 'train', 'labels')
    val_images = os.path.join(output_dataset, 'valid', 'images')
    val_labels = os.path.join(output_dataset, 'valid', 'labels')
    
    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)
    
    # Копируем существующие данные
    existing_train_img = os.path.join(existing_dataset, 'train', 'images')
    existing_train_lbl = os.path.join(existing_dataset, 'train', 'labels')
    
    if os.path.exists(existing_train_img):
        for f in os.listdir(existing_train_img):
            src = os.path.join(existing_train_img, f)
            dst = os.path.join(train_images, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        print(f"Скопировано существующих изображений: {len(os.listdir(existing_train_img))}")
    
    if os.path.exists(existing_train_lbl):
        for f in os.listdir(existing_train_lbl):
            src = os.path.join(existing_train_lbl, f)
            dst = os.path.join(train_labels, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        print(f"Скопировано существующих меток: {len(os.listdir(existing_train_lbl))}")
    
    # Добавляем новые данные
    new_count = 0
    if os.path.exists(new_frames_dir) and os.path.exists(new_labels_dir):
        for f in os.listdir(new_frames_dir):
            if f.endswith(('.jpg', '.png')):
                name = os.path.splitext(f)[0]
                label_file = name + '.txt'
                
                if os.path.exists(os.path.join(new_labels_dir, label_file)):
                    # Копируем изображение
                    shutil.copy2(
                        os.path.join(new_frames_dir, f),
                        os.path.join(train_images, f)
                    )
                    # Копируем метку
                    shutil.copy2(
                        os.path.join(new_labels_dir, label_file),
                        os.path.join(train_labels, label_file)
                    )
                    new_count += 1
        
        print(f"Добавлено новых размеченных кадров: {new_count}")
    
    # Создаём data.yaml
    data_yaml = f"""path: {os.path.abspath(output_dataset)}
train: train/images
val: valid/images

nc: 5
names: ['Ball', 'Net', 'Player1', 'Player2', 'Table']
"""
    
    yaml_path = os.path.join(output_dataset, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)
    
    print(f"\nСоздан data.yaml: {yaml_path}")
    print(f"\nИтого в train:")
    print(f"  Изображений: {len(os.listdir(train_images))}")
    print(f"  Меток: {len(os.listdir(train_labels))}")
    
    return yaml_path


def finetune_model(base_model: str, data_yaml: str, output_name: str, epochs: int = 50):
    """
    Дообучает модель на расширенном датасете.
    
    GOVERNANCE: Требует approval перед запуском!
    """
    
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print("ДООБУЧЕНИЕ МОДЕЛИ")
    print("="*60)
    print(f"База: {base_model}")
    print(f"Данные: {data_yaml}")
    print(f"Эпох: {epochs}")
    print(f"Выход: runs/segment/{output_name}")
    print()
    
    # Загружаем базовую модель
    model = YOLO(base_model)
    
    # Дообучаем
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=1280,
        batch=8,
        name=output_name,
        exist_ok=True,
        # Начинаем с текущих весов (transfer learning)
        pretrained=True,
        # Сохраняем лучшую модель
        save=True,
        save_period=10,
        # Аугментации
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.0,  # не переворачиваем вверх-вниз (теннис специфичен)
        fliplr=0.5,  # можно отзеркалить влево-вправо
        mosaic=0.5,
        mixup=0.0,
    )
    
    print("\n" + "="*60)
    print("ДООБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    print(f"Новая модель: runs/segment/{output_name}/weights/best.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Дообучить модель на новых данных')
    parser.add_argument('--action', choices=['prepare', 'train', 'full'], default='prepare',
                        help='prepare - только подготовка данных, train - только обучение, full - всё')
    parser.add_argument('--new-frames', default='results/missed_detections/frames_to_annotate',
                        help='Папка с новыми кадрами')
    parser.add_argument('--new-labels', default='results/missed_detections/labels',
                        help='Папка с разметкой новых кадров')
    parser.add_argument('--existing', default='dataset_backup',
                        help='Существующий датасет')
    parser.add_argument('--output-dataset', default='dataset_extended',
                        help='Выходной датасет')
    parser.add_argument('--base-model', default='runs/segment/train10/weights/best.pt',
                        help='Базовая модель для дообучения')
    parser.add_argument('--model-name', default='train11_extended',
                        help='Имя новой модели')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох')
    
    args = parser.parse_args()
    
    if args.action in ['prepare', 'full']:
        data_yaml = prepare_dataset(
            new_frames_dir=args.new_frames,
            new_labels_dir=args.new_labels,
            existing_dataset=args.existing,
            output_dataset=args.output_dataset
        )
    else:
        data_yaml = os.path.join(args.output_dataset, 'data.yaml')
    
    if args.action in ['train', 'full']:
        print("\n" + "!"*60)
        print("ВНИМАНИЕ: Дообучение модели требует governance approval!")
        print("Убедитесь что новые данные корректно размечены.")
        print("!"*60)
        
        confirm = input("\nПродолжить дообучение? [y/N]: ")
        if confirm.lower() == 'y':
            finetune_model(
                base_model=args.base_model,
                data_yaml=data_yaml,
                output_name=args.model_name,
                epochs=args.epochs
            )
        else:
            print("Отменено.")


if __name__ == '__main__':
    main()
