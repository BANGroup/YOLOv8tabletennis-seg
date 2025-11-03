#!/usr/bin/env python3
"""
Извлечение кадров из видео
"""

import cv2
import os
from pathlib import Path

def extract_frames_from_video(video_path, output_dir, duration_seconds=5, start_time=0):
    """
    Извлечение кадров из видео
    
    Args:
        video_path: путь к видео файлу
        output_dir: папка для сохранения кадров
        duration_seconds: длительность в секундах для извлечения
        start_time: время начала извлечения в секундах
    """
    
    if not os.path.exists(video_path):
        print(f"Ошибка: видео файл не найден: {video_path}")
        return
    
    # Создаем папку для кадров
    frames_dir = Path(output_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
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
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Информация о видео:")
    print(f"   FPS: {fps:.2f}")
    print(f"   Разрешение: {width}x{height}")
    print(f"   Общая длительность: {duration:.2f} секунд")
    print(f"   Всего кадров: {total_frames}")
    
    # Вычисляем кадры для извлечения
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration_seconds) * fps)
    frames_to_extract = end_frame - start_frame
    
    print(f"\nИзвлечение кадров:")
    print(f"   Начальный кадр: {start_frame}")
    print(f"   Конечный кадр: {end_frame}")
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
    output_dir = "extracted_frames_mid"
    
    # Получаем длительность видео для вычисления середины
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    # Вычисляем середину видео
    mid_time = duration / 2
    duration_seconds = 5
    
    print(f"Середина видео: {mid_time:.2f} секунд")
    print(f"Извлекаем {duration_seconds} секунд начиная с {mid_time:.2f} секунды")
    
    extract_frames_from_video(video_path, output_dir, duration_seconds, mid_time)

if __name__ == "__main__":
    main()
