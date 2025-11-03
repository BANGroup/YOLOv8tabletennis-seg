#!/usr/bin/env python3
"""
Сегментация конкретного диапазона кадров через обученную YOLOv8 модель
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

def segment_frames(start_frame, end_frame, source_folder="extracted_frames_end"):
    """Сегментация диапазона кадров"""
    
    # Путь к обученной модели
    model_path = "runs/segment/train10/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Ошибка: модель не найдена: {model_path}")
        return
    
    print(f"Загружаем обученную модель: {model_path}")
    model = YOLO(model_path)
    
    # Папка с исходными кадрами
    source_path = Path(source_folder)
    if not source_path.exists():
        print(f"Ошибка: папка не найдена: {source_folder}")
        return
    
    # Создаем папку для результатов
    output_dir = Path("segmentation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Очищаем папку от старых результатов
    for old_file in output_dir.glob("*.jpg"):
        old_file.unlink()
    
    print(f"\nСегментация кадров {start_frame} - {end_frame}")
    print("=" * 60)
    
    # Счетчики
    processed = 0
    found = 0
    
    # Обрабатываем каждый кадр в диапазоне
    for frame_num in range(start_frame, end_frame + 1):
        # Формируем имя файла
        frame_filename = f"frame_{frame_num:06d}.jpg"
        frame_path = source_path / frame_filename
        
        if not frame_path.exists():
            print(f"Предупреждение: кадр не найден: {frame_filename}")
            continue
        
        # Загружаем изображение
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        
        # Выполняем сегментацию
        results = model(image, conf=0.25, verbose=False)
        
        # Получаем аннотированное изображение
        annotated_image = results[0].plot()
        
        # Сохраняем результат
        output_path = output_dir / f"seg_{frame_filename}"
        cv2.imwrite(str(output_path), annotated_image)
        
        processed += 1
        
        # Проверяем, есть ли детекции
        if len(results[0].boxes) > 0:
            found += 1
            
        # Выводим прогресс каждые 5 кадров
        if processed % 5 == 0 or processed == 1:
            print(f"Обработано: {processed}/{end_frame - start_frame + 1} кадров, детекций: {found}")
    
    print("\n" + "=" * 60)
    print("Сегментация завершена!")
    print("=" * 60)
    
    print(f"\nСТАТИСТИКА:")
    print(f"   Обработано кадров: {processed}")
    print(f"   Кадров с детекциями: {found}")
    if processed > 0:
        print(f"   Процент обнаружения: {found/processed*100:.1f}%")
    print(f"   Результаты сохранены в: {output_dir}")
    
    # Открываем папку с результатами
    print("\nОткрываю папку с результатами...")
    os.system(f'explorer "{output_dir}"')

if __name__ == "__main__":
    # Диапазон кадров для траектории со скриншота
    START_FRAME = 39560
    END_FRAME = 39589
    
    segment_frames(START_FRAME, END_FRAME)

