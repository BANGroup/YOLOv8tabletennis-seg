#!/usr/bin/env python3
"""
Специализированный трекинг только мяча с сегментацией
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
from pathlib import Path
import json

class BallTracker:
    def __init__(self, model_path, tracker_config="botsort.yaml"):
        """
        Инициализация трекера мяча
        
        Args:
            model_path: путь к обученной модели
            tracker_config: конфигурация трекера
        """
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        
        # Цвет для мяча
        self.ball_color = (0, 255, 0)  # Зеленый
        
        # Названия классов (Ball = индекс 0)
        self.class_names = ['Ball', 'Net', 'Player1', 'Player2', 'Table']
        self.ball_class_id = 0  # Индекс класса Ball
        
        # История позиций мяча для траектории
        self.ball_trajectory = []
        self.max_trajectory_length = 50
        
    def track_ball_video(self, video_path, output_path=None, show_video=True, save_trajectory=True):
        """
        Трекинг мяча в видео
        
        Args:
            video_path: путь к входному видео
            output_path: путь для сохранения результата
            show_video: показывать видео в реальном времени
            save_trajectory: сохранять траекторию мяча
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
        
        # Данные для сохранения траектории
        trajectory_data = []
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
                conf=0.3,  # Более низкий порог для мяча
                iou=0.5,   # Более низкий IoU для мяча
                classes=[self.ball_class_id],  # Только мяч
                verbose=False
            )
            
            # Обрабатываем результаты
            annotated_frame, ball_data = self.draw_ball_tracking(frame, results[0], frame_count)
            
            # Сохраняем данные о мяче
            if ball_data:
                trajectory_data.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'center': ball_data['center'],
                    'bbox': ball_data['bbox'],
                    'confidence': ball_data['confidence'],
                    'track_id': ball_data['track_id']
                })
            
            # Добавляем информацию о кадре
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Показываем видео
            if show_video:
                cv2.imshow('Ball Tracking', annotated_frame)
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
        
        # Сохраняем траекторию
        if save_trajectory and trajectory_data:
            trajectory_file = output_path.replace('.mp4', '_trajectory.json') if output_path else 'ball_trajectory.json'
            with open(trajectory_file, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            print(f"Траектория мяча сохранена в: {trajectory_file}")
        
        print(f"Трекинг мяча завершен! Обработано {frame_count} кадров")
        print(f"Найдено {len(trajectory_data)} кадров с мячом")
        
    def draw_ball_tracking(self, frame, results, frame_number):
        """
        Отрисовка трекинга мяча с сегментацией
        
        Args:
            frame: исходный кадр
            results: результаты детекции и трекинга
            frame_number: номер кадра
            
        Returns:
            аннотированный кадр и данные о мяче
        """
        annotated_frame = frame.copy()
        ball_data = None
        
        if results.boxes is not None and len(results.boxes) > 0:
            # Получаем информацию о детекциях
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else None
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            # Ищем только мячи
            ball_indices = np.where(class_ids == self.ball_class_id)[0]
            
            if len(ball_indices) > 0:
                # Берем мяч с наибольшей уверенностью
                best_ball_idx = ball_indices[np.argmax(confidences[ball_indices])]
                
                box = boxes[best_ball_idx]
                conf = confidences[best_ball_idx]
                track_id = track_ids[best_ball_idx] if track_ids is not None else best_ball_idx
                
                # Получаем маску сегментации
                if results.masks is not None and best_ball_idx < len(results.masks.data):
                    mask = results.masks.data[best_ball_idx].cpu().numpy()
                    
                    # Преобразуем маску в нужный размер
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Создаем цветную маску
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask_binary == 1] = self.ball_color
                    
                    # Накладываем маску с прозрачностью
                    annotated_frame = cv2.addWeighted(annotated_frame, 0.7, colored_mask, 0.3, 0)
                
                # Отрисовываем bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.ball_color, 3)
                
                # Вычисляем центр мяча
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Отрисовываем центр
                cv2.circle(annotated_frame, (center_x, center_y), 5, self.ball_color, -1)
                
                # Добавляем текст
                label = f"Ball ID:{int(track_id)} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ball_color, 2)
                
                # Добавляем координаты центра
                cv2.putText(annotated_frame, f"Center: ({center_x}, {center_y})", 
                           (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ball_color, 2)
                
                # Сохраняем данные о мяче
                ball_data = {
                    'center': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(conf),
                    'track_id': int(track_id)
                }
                
                # Добавляем точку в траекторию
                self.ball_trajectory.append((center_x, center_y))
                if len(self.ball_trajectory) > self.max_trajectory_length:
                    self.ball_trajectory.pop(0)
        
        # Отрисовываем траекторию
        if len(self.ball_trajectory) > 1:
            for i in range(1, len(self.ball_trajectory)):
                # Градиент цвета от старых к новым точкам
                alpha = i / len(self.ball_trajectory)
                color = tuple(int(c * alpha) for c in self.ball_color)
                cv2.line(annotated_frame, self.ball_trajectory[i-1], self.ball_trajectory[i], color, 2)
        
        return annotated_frame, ball_data
    
    def track_ball_images(self, image_folder, output_folder=None, show_images=True):
        """
        Трекинг мяча в последовательности изображений
        
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
        
        ball_detections = 0
        
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
                conf=0.3,
                iou=0.5,
                classes=[self.ball_class_id],  # Только мяч
                verbose=False
            )
            
            # Отрисовываем результаты
            annotated_frame, ball_data = self.draw_ball_tracking(frame, results[0], i+1)
            
            if ball_data:
                ball_detections += 1
            
            # Добавляем информацию о кадре
            cv2.putText(annotated_frame, f"Frame: {i+1}/{len(image_paths)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Показываем изображение
            if show_images:
                cv2.imshow('Ball Tracking', annotated_frame)
                if cv2.waitKey(500) & 0xFF == ord('q'):  # Показываем 0.5 секунды
                    break
            
            # Сохраняем результат
            if output_folder:
                output_path = Path(output_folder) / f"ball_tracked_{image_path.name}"
                cv2.imwrite(str(output_path), annotated_frame)
        
        cv2.destroyAllWindows()
        print(f"Трекинг мяча завершен! Мяч обнаружен в {ball_detections}/{len(image_paths)} кадрах")

def main():
    parser = argparse.ArgumentParser(description='Трекинг мяча с сегментацией')
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
    
    # Создаем трекер мяча
    tracker = BallTracker(args.model, args.tracker)
    
    # Выполняем трекинг
    if args.video:
        print(f"Трекинг мяча в видео: {args.video}")
        tracker.track_ball_video(args.video, args.output, not args.no_show)
    elif args.images:
        print(f"Трекинг мяча в изображениях: {args.images}")
        tracker.track_ball_images(args.images, args.output, not args.no_show)
    else:
        print("Укажите --video или --images для трекинга мяча")

if __name__ == "__main__":
    main()

