#!/usr/bin/env python3
"""
Извлечение кадров ближе к концу видео
"""

import cv2
import os
from pathlib import Path

def extract_end_frames(video_path, output_dir, duration_seconds=5, offset_from_end=30):
    """
    Извлечение кадров ближе к концу видео
    
    Args:
        video_path: путь к видео файлу
        output_dir: папка для сохранения кадров
        duration_seconds: длительность в секундах для извлечения
        offset_from_end: отступ от конца видео в секундах
    """
    
    if not os.path.exists(video_path):
        print(f"Ошибка: видео файл не найден: {video_path}")
        return
    
    # Создаем папку для кадров
    frames_dir = Path(output_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Очищаем предыдущие результаты
    for old_file in frames_dir.glob("*.jpg"):
        old_file.unlink()
    
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return
    
    # Получаем параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration = total_frames / fps if fps > 0 else 0
    
    print(f"Информация о видео:")
    print(f"   FPS: {fps:.2f}")
    print(f"   Разрешение: {width}x{height}")
    print(f"   Общая длительность: {total_duration:.2f} секунд")
    print(f"   Всего кадров: {total_frames}")
    
    # Вычисляем позицию ближе к концу
    end_time = total_duration - offset_from_end
    start_time = end_time - duration_seconds
    
    # Вычисляем кадры для извлечения
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frames_to_extract = end_frame - start_frame
    
    print(f"\nИзвлечение кадров ближе к концу:")
    print(f"   От конца: {offset_from_end} секунд")
    print(f"   Начало извлечения: {start_time:.2f} секунд (кадр {start_frame})")
    print(f"   Конец извлечения: {end_time:.2f} секунд (кадр {end_frame})")
    print(f"   Кадров к извлечению: {frames_to_extract}")
    print(f"   Длительность: {duration_seconds} секунд")
    
    # Переходим к начальному кадру
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    extracted_count = 0
    
    print(f"\nИзвлечение кадров...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        current_frame = start_frame + frame_count
        
        if current_frame >= end_frame:
            break
        
        # Сохраняем кадр
        frame_filename = frames_dir / f"frame_{current_frame:06d}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        extracted_count += 1
        
        # Прогресс каждые 30 кадров
        if extracted_count % 30 == 0:
            print(f"   Извлечено кадров: {extracted_count}/{frames_to_extract}")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nИзвлечение завершено!")
    print(f"   Извлечено кадров: {extracted_count}")
    print(f"   Сохранено в: {frames_dir}")
    
    return extracted_count

def main():
    video_path = "GX010025.MP4"
    output_dir = "extracted_frames_end"
    duration_seconds = 5
    offset_from_end = 30  # Берем за 30 секунд до конца
    
    extract_end_frames(video_path, output_dir, duration_seconds, offset_from_end)

if __name__ == "__main__":
    main()



