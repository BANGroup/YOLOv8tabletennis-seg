#!/usr/bin/env python3
"""
Простой скрипт для тестирования трекинга на изображениях из тестовой папки
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

def test_tracking_on_images():
    """Тестирование трекинга на изображениях из папки test"""
    
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
    
    # Создаем папку для результатов трекинга
    output_dir = "tracking_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Цвета для разных классов
    colors = {
        0: (0, 255, 0),      # Ball - Зеленый
        1: (255, 0, 0),      # Net - Синий  
        2: (0, 0, 255),      # Player1 - Красный
        3: (255, 255, 0),    # Player2 - Голубой
        4: (128, 0, 128)     # Table - Фиолетовый
    }
    
    class_names = ['Ball', 'Net', 'Player1', 'Player2', 'Table']
    
    print("Начинаем трекинг...")
    
    # Обрабатываем каждое изображение
    for i, image_path in enumerate(image_files):
        print(f"Обрабатываем: {image_path.name}")
        
        # Загружаем изображение
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        
        # Выполняем трекинг с сегментацией
        results = model.track(
            frame, 
            persist=True, 
            tracker="botsort.yaml",
            conf=0.5,  # Минимальная уверенность
            iou=0.7,   # IoU threshold
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
            
            # Получаем маски сегментации
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                
                for j, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    # Получаем информацию о треке
                    track_id = track_ids[j] if track_ids is not None else j
                    class_name = class_names[class_id]
                    color = colors.get(class_id, (255, 255, 255))
                    
                    # Отрисовываем bounding box
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Добавляем текст с ID трека и классом
                    label = f"ID:{int(track_id)} {class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Отрисовываем маску сегментации
                    if j < len(masks):
                        mask = masks[j]
                        # Преобразуем маску в нужный размер
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)
                        
                        # Создаем цветную маску
                        colored_mask = np.zeros_like(frame)
                        colored_mask[mask_binary == 1] = color
                        
                        # Накладываем маску с прозрачностью
                        annotated_frame = cv2.addWeighted(annotated_frame, 0.8, colored_mask, 0.2, 0)
        
        # Добавляем информацию о кадре
        cv2.putText(annotated_frame, f"Frame: {i+1}/{len(image_files)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Сохраняем результат
        output_path = Path(output_dir) / f"tracked_{image_path.name}"
        cv2.imwrite(str(output_path), annotated_frame)
        
        # Показываем результат
        cv2.imshow('Segmentation Tracking', annotated_frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):  # Показываем каждое изображение 0.5 секунды
            break
    
    cv2.destroyAllWindows()
    print(f"Трекинг завершен! Результаты сохранены в папку: {output_dir}")

if __name__ == "__main__":
    test_tracking_on_images()

