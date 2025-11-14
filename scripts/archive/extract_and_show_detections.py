#!/usr/bin/env python3
"""
Вырезаем 54-55 секунды и показываем детекцию объектов с моделью train10
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def extract_and_show_detections():
    model_path = "runs/segment/train10/weights/best.pt"
    video_path = "sources/GX010138.MP4"
    output_dir = "results/detections_54_55"
    
    if not os.path.exists(model_path):
        print(f"[!] Модель не найдена: {model_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"[!] Видео не найдено: {video_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Загружаю модель train10...")
    model = YOLO(model_path)
    
    print("[*] Открываю видео...")
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Переходим на 54 секунду
    start_time = 54
    end_time = 55
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    print(f"[*] Переходу на {start_time} сек (кадр {start_frame})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    detections_found = 0
    
    print(f"\n[*] Вырезаю кадры с {start_time}-{end_time} сек с детекцией...")
    
    for frame_idx in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Запускаем детекцию
        print(f"\n    Кадр {frame_idx + 1}: Запускаю детекцию...", end=" ")
        
        results = model.predict(frame, conf=0.1, verbose=False)
        
        # Рисуем боксы и маски
        annotated = frame.copy()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            print(f"[НАЙДЕНО {len(boxes)} объектов]")
            detections_found += len(boxes)
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Зеленый боx для мяча (класс 0)
                color = (0, 255, 0) if class_id == 0 else (0, 165, 255)
                thickness = 3 if class_id == 0 else 2
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                
                # Метка с уверенностью
                label = f"BALL {conf:.2f}" if class_id == 0 else f"Class {class_id} {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Центр объекта
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(annotated, (cx, cy), 5, color, -1)
        else:
            print("[НЕ НАЙДЕНО]")
        
        # Добавляем время кадра
        time_str = f"Time: {start_time + frame_idx/fps:.2f}s"
        cv2.putText(annotated, time_str, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Сохраняем кадр
        output_path = os.path.join(output_dir, f"detection_frame_{frame_idx:03d}.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"    Сохранено: {output_path}")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n[+] ГОТОВО!")
    print(f"    Всего кадров: {frame_count}")
    print(f"    Всего детекций: {detections_found}")
    print(f"    Папка с кадрами: {output_dir}")

if __name__ == "__main__":
    extract_and_show_detections()

