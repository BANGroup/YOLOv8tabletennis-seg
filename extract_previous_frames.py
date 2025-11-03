#!/usr/bin/env python3
"""
Извлечение 5 секунд кадров ПЕРЕД текущими кадрами
"""

import cv2
import os
from pathlib import Path

def extract_previous_frames():
    """Извлечь кадры перед текущими"""
    
    video_path = "GX010025.MP4"
    
    if not os.path.exists(video_path):
        print(f"Ошибка: видео не найдено: {video_path}")
        return
    
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео")
        return
    
    # Получаем параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"FPS видео: {fps:.2f}")
    print(f"Всего кадров в видео: {total_frames}")
    
    # Вычисляем диапазон
    current_start_frame = 39473
    frames_to_extract = int(fps * 5)  # 5 секунд
    new_start_frame = current_start_frame - frames_to_extract
    
    print(f"\nИзвлечение кадров:")
    print(f"  От кадра: {new_start_frame}")
    print(f"  До кадра: {current_start_frame - 1}")
    print(f"  Всего кадров: {frames_to_extract}")
    
    # Создаем папку для новых кадров
    output_dir = Path("extracted_frames_previous")
    output_dir.mkdir(exist_ok=True)
    
    # Очищаем папку
    for old_file in output_dir.glob("*.jpg"):
        old_file.unlink()
    
    print(f"\nИзвлечение в папку: {output_dir}")
    print("=" * 60)
    
    # Устанавливаем позицию на начальный кадр
    cap.set(cv2.CAP_PROP_POS_FRAMES, new_start_frame)
    
    extracted_count = 0
    
    for frame_num in range(new_start_frame, current_start_frame):
        ret, frame = cap.read()
        
        if not ret:
            print(f"Предупреждение: не удалось прочитать кадр {frame_num}")
            break
        
        # Сохраняем кадр
        output_path = output_dir / f"frame_{frame_num:06d}.jpg"
        cv2.imwrite(str(output_path), frame)
        
        extracted_count += 1
        
        # Прогресс каждые 50 кадров
        if extracted_count % 50 == 0 or extracted_count == 1:
            print(f"Извлечено: {extracted_count}/{frames_to_extract} кадров")
    
    cap.release()
    
    print("=" * 60)
    print(f"\nИзвлечение завершено!")
    print(f"Извлечено кадров: {extracted_count}")
    print(f"Папка: {output_dir}")

if __name__ == "__main__":
    extract_previous_frames()


