#!/usr/bin/env python3
"""
Простой тест трекинга мяча на тестовых изображениях
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

def test_ball_tracking():
    """Тестирование трекинга мяча на тестовых изображениях"""
    
    # Путь к обученной модели
    model_path = "runs/segment/train10/weights/best.pt"
    
    # Проверяем наличие модели
    if not os.path.exists(model_path):
        print(f"Ошибка: модель не найдена по пути {model_path}")
        return
    
    # Загружаем модель
    print("Загружаем обученную модель...")
    model = YOLO(model_path)
    
    # Путь к тестовым изображениям
    test_images_path = "dataset/test/images"
    
    if not os.path.exists(test_images_path):
        print(f"Ошибка: папка с тестовыми изображениями не найдена: {test_images_path}")
        return
    
    # Получаем список изображений
    image_files = list(Path(test_images_path).glob("*.jpg"))
    print(f"Найдено {len(image_files)} тестовых изображений")
    
    # Создаем папку для результатов трекинга мяча
    output_dir = "ball_tracking_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Цвет для мяча
    ball_color = (0, 255, 0)  # Зеленый
    
    print("Начинаем трекинг мяча...")
    
    ball_detections = 0
    trajectory = []
    
    # Обрабатываем каждое изображение
    for i, image_path in enumerate(image_files):
        print(f"Обрабатываем: {image_path.name}")
        
        # Загружаем изображение
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        
        # Выполняем трекинг только мяча
        results = model.track(
            frame, 
            persist=True, 
            tracker="botsort.yaml",
            conf=0.3,  # Более низкий порог для мяча
            iou=0.5,   # Более низкий IoU для мяча
            classes=[0],  # Только мяч (класс 0)
            verbose=False
        )
        
        # Создаем аннотированное изображение
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Получаем информацию о детекциях
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else None
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Ищем только мячи
            ball_indices = np.where(class_ids == 0)[0]  # Ball = класс 0
            
            if len(ball_indices) > 0:
                ball_detections += 1
                
                # Берем мяч с наибольшей уверенностью
                best_ball_idx = ball_indices[np.argmax(confidences[ball_indices])]
                
                box = boxes[best_ball_idx]
                conf = confidences[best_ball_idx]
                track_id = track_ids[best_ball_idx] if track_ids is not None else best_ball_idx
                
                # Получаем маску сегментации
                if results[0].masks is not None and best_ball_idx < len(results[0].masks.data):
                    mask = results[0].masks.data[best_ball_idx].cpu().numpy()
                    
                    # Преобразуем маску в нужный размер
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Создаем цветную маску
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask_binary == 1] = ball_color
                    
                    # Накладываем маску с прозрачностью
                    annotated_frame = cv2.addWeighted(annotated_frame, 0.7, colored_mask, 0.3, 0)
                
                # Отрисовываем bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), ball_color, 3)
                
                # Вычисляем центр мяча
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Отрисовываем центр
                cv2.circle(annotated_frame, (center_x, center_y), 5, ball_color, -1)
                
                # Добавляем текст
                label = f"Ball ID:{int(track_id)} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, ball_color, 2)
                
                # Добавляем координаты центра
                cv2.putText(annotated_frame, f"Center: ({center_x}, {center_y})", 
                           (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 2)
                
                # Добавляем точку в траекторию
                trajectory.append((center_x, center_y))
                
                print(f"  -> Мяч обнаружен! ID: {int(track_id)}, Уверенность: {conf:.2f}, Центр: ({center_x}, {center_y})")
            else:
                print(f"  -> Мяч не обнаружен")
        
        # Отрисовываем траекторию
        if len(trajectory) > 1:
            for j in range(1, len(trajectory)):
                # Градиент цвета от старых к новым точкам
                alpha = j / len(trajectory)
                color = tuple(int(c * alpha) for c in ball_color)
                cv2.line(annotated_frame, trajectory[j-1], trajectory[j], color, 2)
        
        # Добавляем информацию о кадре
        cv2.putText(annotated_frame, f"Frame: {i+1}/{len(image_files)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Добавляем статистику
        cv2.putText(annotated_frame, f"Ball detections: {ball_detections}/{i+1}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Сохраняем результат
        output_path = Path(output_dir) / f"ball_tracked_{image_path.name}"
        cv2.imwrite(str(output_path), annotated_frame)
        
        # Сохраняем результат (убираем показ окон)
        print(f"  -> Результат сохранен: {output_path}")
    print(f"\nТрекинг мяча завершен!")
    print(f"Результаты сохранены в папку: {output_dir}")
    print(f"Мяч обнаружен в {ball_detections}/{len(image_files)} кадрах")
    print(f"Процент обнаружения: {ball_detections/len(image_files)*100:.1f}%")

if __name__ == "__main__":
    test_ball_tracking()
