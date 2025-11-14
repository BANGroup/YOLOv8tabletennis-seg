#!/usr/bin/env python3
"""
Скрипт для трекинга объектов с сегментацией на основе обученной модели YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
from pathlib import Path

class SegmentationTracker:
    def __init__(self, model_path, tracker_config="botsort.yaml"):
        """
        Инициализация трекера сегментации
        
        Args:
            model_path: путь к обученной модели
            tracker_config: конфигурация трекера
        """
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        
        # Цвета для разных классов объектов
        self.colors = {
            'Ball': (0, 255, 0),      # Зеленый
            'Net': (255, 0, 0),       # Синий  
            'Player1': (0, 0, 255),   # Красный
            'Player2': (255, 255, 0), # Голубой
            'Table': (128, 0, 128)    # Фиолетовый
        }
        
        # Названия классов
        self.class_names = ['Ball', 'Net', 'Player1', 'Player2', 'Table']
        
    def track_video(self, video_path, output_path=None, show_video=True):
        """
        Трекинг объектов в видео
        
        Args:
            video_path: путь к входному видео
            output_path: путь для сохранения результата
            show_video: показывать видео в реальном времени
        """
        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {video_path}")
            return
            
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Видео: {width}x{height}, {fps} FPS, {total_frames} кадров")
        
        # Настраиваем выходное видео
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Выполняем трекинг с сегментацией
            results = self.model.track(
                frame, 
                persist=True, 
                tracker=self.tracker_config,
                conf=0.5,  # Минимальная уверенность
                iou=0.7,   # IoU threshold
                verbose=False
            )
            
            # Обрабатываем результаты
            annotated_frame = self.draw_tracking_results(frame, results[0])
            
            # Добавляем информацию о кадре
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Показываем видео
            if show_video:
                cv2.imshow('Segmentation Tracking', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Сохраняем результат
            if output_path:
                out.write(annotated_frame)
            
            # Прогресс
            if frame_count % 30 == 0:
                print(f"Обработано кадров: {frame_count}/{total_frames}")
        
        # Освобождаем ресурсы
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Трекинг завершен! Обработано {frame_count} кадров")
        
    def draw_tracking_results(self, frame, results):
        """
        Отрисовка результатов трекинга с сегментацией
        
        Args:
            frame: исходный кадр
            results: результаты детекции и трекинга
            
        Returns:
            аннотированный кадр
        """
        annotated_frame = frame.copy()
        
        if results.boxes is not None and len(results.boxes) > 0:
            # Получаем информацию о детекциях
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else None
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            # Получаем маски сегментации
            if results.masks is not None:
                masks = results.masks.data.cpu().numpy()
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    # Получаем информацию о треке
                    track_id = track_ids[i] if track_ids is not None else i
                    class_name = self.class_names[class_id]
                    color = self.colors.get(class_name, (255, 255, 255))
                    
                    # Отрисовываем bounding box
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Добавляем текст с ID трека и классом
                    label = f"ID:{int(track_id)} {class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Отрисовываем маску сегментации
                    if i < len(masks):
                        mask = masks[i]
                        # Преобразуем маску в нужный размер
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)
                        
                        # Создаем цветную маску
                        colored_mask = np.zeros_like(frame)
                        colored_mask[mask_binary == 1] = color
                        
                        # Накладываем маску с прозрачностью
                        annotated_frame = cv2.addWeighted(annotated_frame, 0.8, colored_mask, 0.2, 0)
        
        return annotated_frame
    
    def track_image_sequence(self, image_folder, output_folder=None, show_images=True):
        """
        Трекинг объектов в последовательности изображений
        
        Args:
            image_folder: папка с изображениями
            output_folder: папка для сохранения результатов
            show_images: показывать изображения
        """
        image_paths = sorted(Path(image_folder).glob("*.jpg")) + sorted(Path(image_folder).glob("*.png"))
        
        if not image_paths:
            print(f"Изображения не найдены в папке {image_folder}")
            return
            
        print(f"Найдено {len(image_paths)} изображений")
        
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            print(f"Обрабатываем: {image_path.name}")
            
            # Загружаем изображение
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
                
            # Выполняем трекинг
            results = self.model.track(
                frame, 
                persist=True, 
                tracker=self.tracker_config,
                conf=0.5,
                iou=0.7,
                verbose=False
            )
            
            # Отрисовываем результаты
            annotated_frame = self.draw_tracking_results(frame, results[0])
            
            # Добавляем информацию о кадре
            cv2.putText(annotated_frame, f"Frame: {i+1}/{len(image_paths)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Показываем изображение
            if show_images:
                cv2.imshow('Segmentation Tracking', annotated_frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            
            # Сохраняем результат
            if output_folder:
                output_path = Path(output_folder) / f"tracked_{image_path.name}"
                cv2.imwrite(str(output_path), annotated_frame)
        
        cv2.destroyAllWindows()
        print("Трекинг последовательности изображений завершен!")

def main():
    parser = argparse.ArgumentParser(description='Трекинг объектов с сегментацией')
    parser.add_argument('--model', type=str, default='runs/segment/train10/weights/best.pt',
                       help='Путь к обученной модели')
    parser.add_argument('--video', type=str, help='Путь к входному видео')
    parser.add_argument('--images', type=str, help='Папка с изображениями')
    parser.add_argument('--output', type=str, help='Путь для сохранения результата')
    parser.add_argument('--tracker', type=str, default='botsort.yaml',
                       help='Конфигурация трекера')
    parser.add_argument('--no-show', action='store_true', help='Не показывать видео/изображения')
    
    args = parser.parse_args()
    
    # Проверяем наличие модели
    if not os.path.exists(args.model):
        print(f"Ошибка: модель не найдена по пути {args.model}")
        return
    
    # Создаем трекер
    tracker = SegmentationTracker(args.model, args.tracker)
    
    # Выполняем трекинг
    if args.video:
        print(f"Трекинг видео: {args.video}")
        tracker.track_video(args.video, args.output, not args.no_show)
    elif args.images:
        print(f"Трекинг изображений: {args.images}")
        tracker.track_image_sequence(args.images, args.output, not args.no_show)
    else:
        print("Укажите --video или --images для трекинга")

if __name__ == "__main__":
    main()

