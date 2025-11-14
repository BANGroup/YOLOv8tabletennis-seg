#!/usr/bin/env python3
"""
Показываем МАСКИ СЕГМЕНТАЦИИ для стола и сетки
Модель обучена на сегментацию!
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def show_segmentation_masks():
    model_path = "runs/segment/train10/weights/best.pt"
    video_path = "sources/GX010138.MP4"
    output_dir = "results/segmentation_masks_635_636"
    
    if not os.path.exists(model_path):
        print(f"[!] Модель не найдена: {model_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"[!] Видео не найдено: {video_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Загружаю СЕГМЕНТАЦИОННУЮ модель train10...")
    model = YOLO(model_path)
    
    print(f"[*] Классы модели: {model.names}")
    
    # Цвета для каждого класса
    class_colors = {
        0: (0, 255, 0),      # Ball - зеленый
        1: (255, 255, 0),    # Net - голубой
        2: (0, 255, 255),    # Player1 - желтый
        3: (255, 0, 255),    # Player2 - магента
        4: (255, 0, 0)       # Table - синий
    }
    
    print("[*] Открываю видео...")
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Переходим на 06:35
    start_time = 6 * 60 + 35  # 395 секунд
    end_time = 6 * 60 + 36    # 396 секунд
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    print(f"[*] Вырезаю сегментационные маски с {start_time}-{end_time} сек...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    
    for frame_idx in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Запускаем СЕГМЕНТАЦИЮ
        results = model.predict(frame, conf=0.1, verbose=False)
        
        # Создаем 3 версии кадра
        annotated_boxes = frame.copy()
        annotated_masks = frame.copy()
        annotated_combined = frame.copy()
        
        current_time = start_time + frame_idx / fps
        current_minute = int(current_time // 60)
        current_second = int(current_time % 60)
        current_ms = int((current_time % 1) * 100)
        time_str = f"{current_minute:02d}:{current_second:02d}.{current_ms:02d}"
        
        has_masks = False
        
        if results[0].masks is not None:
            has_masks = True
            masks = results[0].masks.xy
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            print(f"    Кадр {frame_idx:3d} ({time_str}): Найдено {len(masks)} масок сегментации")
            
            # Отрисовываем маски
            for i, (mask_xy, box, conf, class_id) in enumerate(zip(masks, boxes, confidences, class_ids)):
                color = class_colors.get(class_id, (255, 255, 255))
                class_name = model.names[class_id]
                
                # Нормализуем координаты маски
                mask_points = np.array(mask_xy, dtype=np.int32)
                
                # На отдельной версии с масками
                cv2.polylines(annotated_masks, [mask_points], True, color, 2)
                cv2.fillPoly(annotated_masks, [mask_points], color)
                alpha = 0.3
                annotated_masks = cv2.addWeighted(annotated_masks, alpha, frame, 1 - alpha, 0)
                
                # На комбинированной версии
                cv2.polylines(annotated_combined, [mask_points], True, color, 2)
                
                # Боксы
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_boxes, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(annotated_combined, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {conf:.2f}"
                cv2.putText(annotated_boxes, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(annotated_combined, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if not has_masks and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            print(f"    Кадр {frame_idx:3d} ({time_str}): Боксы (маски не найдены)")
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                color = class_colors.get(class_id, (255, 255, 255))
                class_name = model.names[class_id]
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_boxes, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(annotated_boxes, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Добавляем время
        for img in [annotated_boxes, annotated_masks, annotated_combined]:
            cv2.putText(img, f"Time: {time_str}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Объединяем три версии в одну для наглядности (если нужно)
        # Сохраняем каждую версию отдельно
        if frame_idx % 12 == 0:  # Каждые ~400мс
            cv2.imwrite(os.path.join(output_dir, f"boxes_{frame_idx:04d}_{time_str.replace(':', '_')}.jpg"), annotated_boxes)
            if has_masks:
                cv2.imwrite(os.path.join(output_dir, f"masks_{frame_idx:04d}_{time_str.replace(':', '_')}.jpg"), annotated_masks)
                cv2.imwrite(os.path.join(output_dir, f"combined_{frame_idx:04d}_{time_str.replace(':', '_')}.jpg"), annotated_combined)
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n[+] ГОТОВО!")
    print(f"\n[=] СЕГМЕНТАЦИОННЫЕ МАСКИ:")
    print(f"    Модель: СЕГМЕНТАЦИОННАЯ (YOLOv8 Segment)")
    print(f"    Обработано кадров: {frame_count}")
    print(f"    Сохранены версии: boxes, masks, combined")
    print(f"    Папка: {output_dir}")
    print(f"\n[*] Маски сегментации УСПЕШНО ИЗВЛЕЧЕНЫ!")

if __name__ == "__main__":
    show_segmentation_masks()

