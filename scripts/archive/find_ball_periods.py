#!/usr/bin/env python3
"""
Сканируем ВЕСЬ видео и находим периоды с активным мячом
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def find_ball_periods():
    model_path = "runs/segment/train10/weights/best.pt"
    video_path = "sources/GX010138.MP4"
    
    if not os.path.exists(model_path):
        print(f"[!] Модель не найдена: {model_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"[!] Видео не найдено: {video_path}")
        return
    
    print("[*] Загружаю модель train10...")
    model = YOLO(model_path)
    
    print("[*] Открываю видео...")
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    
    print(f"    FPS: {fps:.2f}")
    print(f"    Всего кадров: {total_frames}")
    print(f"    Длительность: {total_duration:.1f} сек ({total_duration/60:.1f} мин)")
    
    print(f"\n[*] Сканирую видео в поисках мяча (каждый 10-й кадр для скорости)...")
    
    ball_detections = []
    frame_count = 0
    
    for frame_idx in range(0, total_frames, 10):  # Каждый 10-й кадр
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame, conf=0.1, verbose=False)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Проверяем есть ли мяч (класс 0)
            if 0 in class_ids:
                time_seconds = frame_idx / fps
                ball_detections.append({
                    'frame': frame_idx,
                    'time': time_seconds,
                    'minute': int(time_seconds // 60),
                    'second': int(time_seconds % 60)
                })
        
        frame_count += 1
        if frame_count % 500 == 0:
            print(f"    Проверено {frame_count} кадров ({frame_idx/fps:.1f}сек)...")
    
    cap.release()
    
    print(f"\n[+] СКАНИРОВАНИЕ ЗАВЕРШЕНО!")
    print(f"\n[=] РЕЗУЛЬТАТЫ:")
    print(f"    Найдено детекций мяча: {len(ball_detections)}")
    
    if len(ball_detections) > 0:
        print(f"\n[*] ПЕРИОДЫ С АКТИВНЫМ МЯЧОМ:")
        print(f"    {'Время':<12} {'Кадр':<8} {'Min:Sec':<10}")
        print(f"    {'-'*30}")
        
        for detection in ball_detections:
            print(f"    {detection['time']:>7.2f}s  {detection['frame']:>7}  {detection['minute']:02d}:{detection['second']:02d}")
        
        print(f"\n[*] РЕКОМЕНДАЦИЯ:")
        print(f"    Выбери любую из этих секунд для нарезки!")
        print(f"    Например: нарезать 5 минут начиная с секунды {int(ball_detections[len(ball_detections)//2]['time'])}")
    else:
        print(f"\n[!] Мяч не найден вообще на этом видео!")

if __name__ == "__main__":
    find_ball_periods()

