#!/usr/bin/env python3
"""
Вырезаем 06:35-06:36 и показываем детекцию со ВСЕМИ классами
ИСПРАВЛЕННАЯ КОДИРОВКА - русские названия класссов выводятся правильно!
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

# Исправляем кодировку для вывода русского текста
os.environ['PYTHONIOENCODING'] = 'utf-8'

def extract_and_show_detection():
    model_path = "runs/segment/train10/weights/best.pt"
    video_path = "sources/GX010138.MP4"
    output_dir = "results/detections_635_636"
    
    if not os.path.exists(model_path):
        print(f"[!] Модель не найдена: {model_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"[!] Видео не найдено: {video_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Загружаю модель train10...")
    model = YOLO(model_path)
    
    print(f"[*] Классы модели: {model.names}")
    
    # Цвета для каждого класса (BGR)
    class_colors = {
        0: (0, 255, 0),      # Ball - зеленый
        1: (255, 255, 0),    # Net - голубой
        2: (0, 255, 255),    # Player1 - желтый
        3: (255, 0, 255),    # Player2 - магента
        4: (255, 0, 0)       # Table - синий
    }
    
    # Русские названия классов
    class_names_ru = {
        0: "МЯЧ",
        1: "СЕТКА",
        2: "ИГРОК 1",
        3: "ИГРОК 2",
        4: "СТОЛ"
    }
    
    print("[*] Открываю видео...")
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Переходим на 06:35
    start_time = 6 * 60 + 35  # 395 секунд
    end_time = 6 * 60 + 36    # 396 секунд
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    print(f"[*] Переходу на {start_time} сек (кадр {start_frame})")
    print(f"[*] Вырезаю до {end_time} сек (кадр {end_frame})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    total_detections = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    print(f"\n[*] Вырезаю кадры с {start_time}-{end_time} сек...")
    print(f"    Будет примерно {(end_frame - start_frame)} кадров")
    print()
    
    for frame_idx in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Запускаем детекцию
        results = model.predict(frame, conf=0.1, verbose=False)
        
        annotated = frame.copy()
        
        current_time = start_time + frame_idx / fps
        current_minute = int(current_time // 60)
        current_second = int(current_time % 60)
        current_ms = int((current_time % 1) * 100)
        
        detections_this_frame = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Выбираем цвет по классу
                color = class_colors.get(class_id, (255, 255, 255))
                thickness = 3 if class_id == 0 else 2  # Толще для мяча
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                
                # Метка с классом на АНГЛИЙСКОМ (для избежания проблем)
                class_label = model.names[class_id]
                label = f"{class_label} {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Центр объекта
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(annotated, (cx, cy), 5, color, -1)
                
                total_detections[class_id] += 1
                detections_this_frame.append({
                    'class_id': class_id,
                    'class_name': class_names_ru.get(class_id, f"Class {class_id}"),
                    'confidence': conf
                })
        
        # Добавляем время кадра
        time_str = f"{current_minute:02d}:{current_second:02d}.{current_ms:02d}"
        cv2.putText(annotated, f"Time: {time_str}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Легенда классов в нижнем левом углу
        legend_y = annotated.shape[0] - 160
        for idx, (class_id, name) in enumerate(class_names_ru.items()):
            y = legend_y + idx * 25
            color = class_colors[class_id]
            cv2.rectangle(annotated, (10, y - 15), (30, y), color, -1)
            # Используем английские названия в легенде
            cv2.putText(annotated, model.names[class_id], (40, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Сохраняем кадр
        output_path = os.path.join(output_dir, f"det_{frame_idx:04d}_{current_minute:02d}_{current_second:02d}.jpg")
        cv2.imwrite(output_path, annotated)
        
        # Выводим информацию о кадре
        if frame_idx % 6 == 0:  # Каждые ~200мс (при 30fps)
            det_str = ", ".join([f"{d['class_name']}({d['confidence']:.2f})" for d in detections_this_frame]) if detections_this_frame else "NO DETECTIONS"
            print(f"    Кадр {frame_idx:3d} ({time_str}): {det_str}")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n[+] ГОТОВО!")
    print(f"\n[=] СТАТИСТИКА:")
    print(f"    Всего кадров: {frame_count}")
    print(f"    Временной отрезок: 06:35 - 06:36 (1 секунда)")
    print()
    print(f"    ДЕТЕКЦИИ ПО КЛАССАМ:")
    for class_id in range(5):
        count = total_detections[class_id]
        class_name = model.names[class_id]
        print(f"      {class_name:10} ({class_id}): {count:3d} детекций")
    print(f"    {'':10} ВСЕГО: {sum(total_detections.values())} детекций")
    print(f"\n    Папка с кадрами: {output_dir}")
    print(f"\n[*] Кадры сохранены! Ты можешь открыть их в папке.")

if __name__ == "__main__":
    extract_and_show_detection()

