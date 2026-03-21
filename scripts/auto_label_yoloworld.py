#!/usr/bin/env python3
"""
Авторазметка с YOLO-World (open-vocabulary detection)
Использует текстовый промпт для поиска объектов.
"""

import os
import cv2
from pathlib import Path
from ultralytics import YOLO


def auto_label_with_yoloworld(
    images_dir: str,
    output_labels_dir: str,
    classes: list[str] = ["ball", "ping pong ball", "table tennis ball"],
    model_size: str = "yolov8x-worldv2",  # x = большая, точная
    confidence: float = 0.1,
    max_images: int = 50
):
    """
    Авторазметка изображений с YOLO-World.
    
    Args:
        images_dir: папка с изображениями
        output_labels_dir: папка для сохранения меток
        classes: текстовые описания объектов для поиска
        model_size: размер модели (s/m/l/x)
        confidence: порог уверенности
        max_images: максимум изображений для обработки
    """
    
    os.makedirs(output_labels_dir, exist_ok=True)
    
    print("="*60)
    print("АВТОРАЗМЕТКА С YOLO-World")
    print("="*60)
    print(f"Модель: {model_size}")
    print(f"Классы для поиска: {classes}")
    print(f"Порог уверенности: {confidence}")
    print()
    
    # Загружаем YOLO-World
    print("Загрузка модели...")
    model = YOLO(model_size + ".pt")
    
    # Устанавливаем классы для поиска
    model.set_classes(classes)
    
    # Получаем список изображений
    image_files = sorted([
        f for f in os.listdir(images_dir) 
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])[:max_images]
    
    print(f"Изображений для обработки: {len(image_files)}")
    print()
    
    stats = {
        'processed': 0,
        'detected': 0,
        'total_objects': 0
    }
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        
        # Детекция
        results = model(img_path, conf=confidence, verbose=False)[0]
        
        # Получаем размеры изображения
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # Формируем метки в формате YOLO
        labels = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            # Конвертируем в YOLO формат (normalized x_center, y_center, width, height)
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            # Class 0 = Ball
            labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            stats['total_objects'] += 1
        
        # Сохраняем метку
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(output_labels_dir, label_file)
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
        
        stats['processed'] += 1
        if labels:
            stats['detected'] += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Обработано: {i + 1}/{len(image_files)}")
    
    print()
    print("="*60)
    print("РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"Обработано изображений: {stats['processed']}")
    print(f"Найден мяч на: {stats['detected']} ({100*stats['detected']/stats['processed']:.1f}%)")
    print(f"Всего объектов: {stats['total_objects']}")
    print(f"\nМетки сохранены в: {output_labels_dir}")
    
    return stats


def visualize_results(images_dir: str, labels_dir: str, output_dir: str, max_images: int = 10):
    """Визуализация результатов авторазметки."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted([
        f for f in os.listdir(images_dir) 
        if f.endswith(('.jpg', '.png'))
    ])[:max_images]
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x_c, y_c, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        
                        # Конвертируем обратно в пиксели
                        x1 = int((x_c - bw/2) * w)
                        y1 = int((y_c - bh/2) * h)
                        x2 = int((x_c + bw/2) * w)
                        y2 = int((y_c + bh/2) * h)
                        
                        # Рисуем bbox
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, "Ball (YOLO-World)", (x1, y1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, img)
    
    print(f"Визуализация сохранена в: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='results/missed_detections/frames_to_annotate')
    parser.add_argument('--output', default='results/auto_labels_yoloworld')
    parser.add_argument('--max', type=int, default=50)
    parser.add_argument('--conf', type=float, default=0.1)
    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    
    stats = auto_label_with_yoloworld(
        images_dir=args.images,
        output_labels_dir=args.output,
        confidence=args.conf,
        max_images=args.max
    )
    
    if args.visualize:
        visualize_results(
            images_dir=args.images,
            labels_dir=args.output,
            output_dir=args.output + '_viz',
            max_images=10
        )
