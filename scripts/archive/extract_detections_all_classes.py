#!/usr/bin/env python3
"""
Вырезаем 54-55 секунды и показываем детекцию ВСЕХ объектов с правильными классами:
- Ball (зеленый)
- Net (голубой)
- Player1 (желтый)
- Player2 (фиолетовый)
- Table (синий)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def extract_and_show_all_classes():
    model_path = "runs/segment/train10/weights/best.pt"
    video_path = "sources/GX010138.MP4"
    output_dir = "results/detections_all_classes_54_55"
    
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
    
    # Цвета для каждого класса
    class_colors = {
        0: (0, 255, 0),      # Ball - зеленый
        1: (255, 255, 0),    # Net - голубой (B, G, R)
        2: (0, 255, 255),    # Player1 - желтый
        3: (255, 0, 255),    # Player2 - магента/фиолетовый
        4: (255, 0, 0)       # Table - синий
    }
    
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
    
    # Переходим на 54 секунду
    start_time = 54
    end_time = 55
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    print(f"[*] Переходу на {start_time} сек (кадр {start_frame})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    total_detections = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    print(f"\n[*] Вырезаю кадры с {start_time}-{end_time} сек с ПОЛНОЙ детекцией...")
    
    for frame_idx in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Запускаем детекцию со ВСЕМИ классами
        results = model.predict(frame, conf=0.1, verbose=False)
        
        annotated = frame.copy()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            print(f"\n    Кадр {frame_idx + 1}: Детекции:")
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Выбираем цвет по классу
                color = class_colors.get(class_id, (255, 255, 255))
                thickness = 3 if class_id == 0 else 2  # Толще для мяча
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                
                # Метка с классом и уверенностью
                label = f"{class_names_ru.get(class_id, f'Class {class_id}')} {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Центр объекта
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(annotated, (cx, cy), 5, color, -1)
                
                total_detections[class_id] += 1
                print(f"      - {class_names_ru.get(class_id, f'Class {class_id}')}: {conf:.3f}")
        else:
            print(f"\n    Кадр {frame_idx + 1}: Детекции не найдены")
        
        # Добавляем время кадра и легенду
        time_str = f"Time: {start_time + frame_idx/fps:.2f}s"
        cv2.putText(annotated, time_str, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Добавляем легенду классов в нижний левый угол
        legend_y = annotated.shape[0] - 150
        cv2.putText(annotated, "Legend:", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for idx, (class_id, name) in enumerate(class_names_ru.items()):
            y = legend_y + 25 + idx * 25
            color = class_colors[class_id]
            cv2.rectangle(annotated, (10, y - 15), (30, y), color, -1)
            cv2.putText(annotated, name, (40, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Сохраняем кадр
        output_path = os.path.join(output_dir, f"all_classes_{frame_idx:03d}.jpg")
        cv2.imwrite(output_path, annotated)
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n[+] ГОТОВО!")
    print(f"\n[=] СТАТИСТИКА ПО КЛАССАМ:")
    print(f"    Всего кадров: {frame_count}")
    for class_id, count in total_detections.items():
        class_name = class_names_ru.get(class_id, f"Class {class_id}")
        print(f"    {class_name}: {count} детекций")
    print(f"    Всего детекций: {sum(total_detections.values())}")
    print(f"\n    Папка с кадрами: {output_dir}")

if __name__ == "__main__":
    extract_and_show_all_classes()

