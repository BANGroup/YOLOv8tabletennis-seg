#!/usr/bin/env python3
"""
Умный трекинг мяча с правильной траекторией полета
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import time
from collections import deque
import math
import json

class SmartBallTracker:
    """Умный трекер мяча с правильной траекторией"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Параметры детекции
        self.confidence_threshold = 0.25
        
        # Фильтры размера мяча
        self.min_ball_area = 100
        self.max_ball_area = 3000
        self.expected_ball_size = 500
        
        # Физические параметры
        self.max_speed = 150  # пикселей/кадр
        self.max_jump = 100   # максимальный скачок между кадрами
        
        # Текущая траектория (сбрасывается при потере мяча)
        self.current_trajectory = []
        self.max_trajectory_length = 30  # Максимум 30 точек для текущей траектории
        
        # История всех траекторий
        self.all_trajectories = []
        
        # Информация о розыгрышах
        self.rallies = []  # Список розыгрышей с метаданными
        self.current_rally_start_frame = None
        self.current_rally_start_time = None
        self.rally_counter = 0
        
        # Параметры розыгрыша
        self.max_rally_duration = 2.0  # Максимальная длительность розыгрыша в секундах
        self.max_rally_frames = int(60 * self.max_rally_duration)  # ~120 кадров при 60 FPS
        
        # Счетчики
        self.frames_without_ball = 0
        self.max_frames_gap = 5  # Максимальный разрыв для сохранения траектории
        
        # Последняя надежная позиция
        self.last_position = None
        self.last_confidence = 0
        
        # Направление движения
        self.velocity = None  # (vx, vy) - вектор скорости
        self.direction_history = []  # История направлений
        self.max_direction_change = 90  # Максимальное изменение угла (градусы)
        
        # Статистика
        self.total_detections = 0
        self.total_trajectories = 0
        
        # Цвета
        self.ball_color = (0, 255, 0)        # Зеленый
        self.trajectory_color = (0, 140, 255) # Оранжевый
        self.old_trajectory_color = (128, 128, 128)  # Серый
        self.velocity_color = (255, 0, 255)  # Пурпурный для вектора скорости
    
    def calculate_velocity(self, current_pos, last_pos):
        """Вычисление вектора скорости"""
        if last_pos is None:
            return None
        
        vx = current_pos[0] - last_pos[0]
        vy = current_pos[1] - last_pos[1]
        
        return (vx, vy)
    
    def get_direction_angle(self, velocity):
        """Получить угол направления в градусах"""
        if velocity is None:
            return None
        
        vx, vy = velocity
        if vx == 0 and vy == 0:
            return None
        
        angle = math.degrees(math.atan2(vy, vx))
        return angle
    
    def is_direction_consistent(self, new_velocity):
        """Проверка согласованности направления движения"""
        if self.velocity is None or new_velocity is None:
            return True
        
        old_angle = self.get_direction_angle(self.velocity)
        new_angle = self.get_direction_angle(new_velocity)
        
        if old_angle is None or new_angle is None:
            return True
        
        # Вычисляем разницу углов
        angle_diff = abs(new_angle - old_angle)
        
        # Нормализуем разницу к диапазону [0, 180]
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # Проверяем, не превышает ли изменение максимально допустимое
        return angle_diff <= self.max_direction_change
    
    def is_rally_start(self, frame_number):
        """Определение начала розыгрыша - первая детекция после паузы"""
        # Начало розыгрыша = нет активной траектории (пауза между розыгрышами)
        return len(self.current_trajectory) == 0 and self.current_rally_start_frame is None
    
    def start_new_rally(self, frame_number, video_time):
        """Начать новый розыгрыш"""
        self.rally_counter += 1
        self.current_rally_start_frame = frame_number
        self.current_rally_start_time = video_time
        
        print(f"\n🎾 НАЧАЛО РОЗЫГРЫША #{self.rally_counter}")
        if video_time:
            minutes = int(video_time // 60)
            seconds = video_time % 60
            print(f"   Время: {minutes:02d}:{seconds:05.2f}")
        print(f"   Кадр: {frame_number}")
    
    def end_rally(self, frame_number, video_time):
        """Завершить текущий розыгрыш"""
        if self.current_rally_start_frame is None:
            return
        
        duration_frames = frame_number - self.current_rally_start_frame
        duration_seconds = (video_time - self.current_rally_start_time) if video_time and self.current_rally_start_time else None
        
        rally_info = {
            'rally_number': self.rally_counter,
            'start_frame': self.current_rally_start_frame,
            'end_frame': frame_number,
            'duration_frames': duration_frames,
            'start_time': self.current_rally_start_time,
            'end_time': video_time,
            'duration_seconds': duration_seconds,
            'trajectory_points': len(self.current_trajectory)
        }
        
        self.rallies.append(rally_info)
        
        print(f"\n✓ КОНЕЦ РОЗЫГРЫША #{self.rally_counter}")
        if duration_seconds:
            print(f"   Длительность: {duration_seconds:.2f} сек ({duration_frames} кадров)")
        print(f"   Точек траектории: {len(self.current_trajectory)}")
    
    def validate_ball_detection(self, detection, frame_shape):
        """Валидация детекции мяча"""
        if not detection:
            return False
        
        box = detection['box']
        center = detection['center']
        size = detection['size']
        
        # Проверка размера
        if size < self.min_ball_area or size > self.max_ball_area:
            return False
        
        # Проверка формы (круглость)
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            return False
        
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
        
        # Проверка скорости и направления (если есть предыдущая позиция)
        if self.last_position:
            dx = center[0] - self.last_position[0]
            dy = center[1] - self.last_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Слишком большой скачок - возможно другой объект
            if distance > self.max_jump:
                return False
            
            # Проверка согласованности направления движения
            new_velocity = (dx, dy)
            if not self.is_direction_consistent(new_velocity):
                # Направление изменилось слишком резко - возможно ложная детекция
                return False
        
        # Проверка границ изображения
        h, w = frame_shape[:2]
        margin = 20
        if (center[0] < margin or center[0] > w - margin or 
            center[1] < margin or center[1] > h - margin):
            return False
        
        return True
    
    def detect_ball(self, frame):
        """Детекция мяча"""
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=0.4,
            classes=[0],  # Только мяч
            verbose=False
        )
        
        detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                if class_id == 0:
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    size = (x2 - x1) * (y2 - y1)
                    
                    detection = {
                        'box': box,
                        'center': (center_x, center_y),
                        'confidence': conf,
                        'size': size
                    }
                    
                    if self.validate_ball_detection(detection, frame.shape):
                        detections.append(detection)
        
        # Выбираем лучшую детекцию
        if detections:
            # Если есть предыдущая позиция, выбираем ближайшую
            if self.last_position:
                def distance_score(det):
                    dx = det['center'][0] - self.last_position[0]
                    dy = det['center'][1] - self.last_position[1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    # Комбинируем расстояние и уверенность
                    return det['confidence'] * 0.6 - distance * 0.001
                
                best_detection = max(detections, key=distance_score)
            else:
                # Выбираем с наибольшей уверенностью
                best_detection = max(detections, key=lambda x: x['confidence'])
            
            return best_detection
        
        return None
    
    def update_trajectory(self, detection, frame_number=None, video_time=None):
        """Обновление траектории с правильной логикой"""
        if detection:
            # Мяч обнаружен
            center = detection['center']
            
            # Вычисляем вектор скорости
            new_velocity = self.calculate_velocity(center, self.last_position)
            if new_velocity:
                self.velocity = new_velocity
                # Сохраняем в историю направлений
                angle = self.get_direction_angle(new_velocity)
                if angle is not None:
                    self.direction_history.append(angle)
                    # Ограничиваем историю
                    if len(self.direction_history) > 10:
                        self.direction_history.pop(0)
            
            # Добавляем в текущую траекторию
            self.current_trajectory.append({
                'position': center,
                'confidence': detection['confidence'],
                'frame_time': time.time(),
                'velocity': self.velocity
            })
            
            # Ограничиваем длину траектории
            if len(self.current_trajectory) > self.max_trajectory_length:
                self.current_trajectory.pop(0)
            
            # Обновляем последнюю позицию
            self.last_position = center
            self.last_confidence = detection['confidence']
            
            # Сбрасываем счетчик пропусков
            self.frames_without_ball = 0
            self.total_detections += 1
            
        else:
            # Мяч не обнаружен
            self.frames_without_ball += 1
            
            # Если мяч потерян надолго, сохраняем траекторию и начинаем новую
            if self.frames_without_ball > self.max_frames_gap:
                if len(self.current_trajectory) > 5:  # Сохраняем только если траектория значимая
                    self.all_trajectories.append(list(self.current_trajectory))
                    self.total_trajectories += 1
                    
                    # Завершаем розыгрыш
                    if frame_number:
                        self.end_rally(frame_number, video_time)
                
                # Сбрасываем текущую траекторию и направление
                self.current_trajectory = []
                self.last_position = None
                self.last_confidence = 0
                self.velocity = None
                self.direction_history = []
                self.current_rally_start_frame = None
                self.current_rally_start_time = None
    
    def track_ball(self, frame, frame_number=None, video_time=None):
        """Основной метод трекинга"""
        # Детекция мяча
        detection = self.detect_ball(frame)
        
        # Проверка начала розыгрыша (первая детекция после паузы)
        if detection and frame_number:
            if self.is_rally_start(frame_number):
                self.start_new_rally(frame_number, video_time)
        
        # Проверка превышения максимальной длительности розыгрыша
        if detection and self.current_rally_start_frame and frame_number:
            rally_duration_frames = frame_number - self.current_rally_start_frame
            if rally_duration_frames >= self.max_rally_frames:
                # Розыгрыш длится слишком долго - завершаем его
                self.end_rally(frame_number, video_time)
                # Сохраняем траекторию
                if len(self.current_trajectory) > 5:
                    self.all_trajectories.append(list(self.current_trajectory))
                    self.total_trajectories += 1
                # Сбрасываем состояние
                self.current_trajectory = []
                self.last_position = None
                self.last_confidence = 0
                self.velocity = None
                self.direction_history = []
                self.current_rally_start_frame = None
                self.current_rally_start_time = None
                self.frames_without_ball = 0
        
        # Обновление траектории
        self.update_trajectory(detection, frame_number, video_time)
        
        return detection
    
    def draw_tracking(self, frame, detection, frame_number, video_time=None):
        """Отрисовка результатов трекинга"""
        annotated_frame = frame.copy()
        
        # Отрисовываем старые траектории серым цветом
        for old_trajectory in self.all_trajectories:
            if len(old_trajectory) > 1:
                points = [(int(p['position'][0]), int(p['position'][1])) for p in old_trajectory]
                for i in range(1, len(points)):
                    cv2.line(annotated_frame, points[i-1], points[i], 
                            self.old_trajectory_color, 1)
        
        # Отрисовываем текущую траекторию
        if len(self.current_trajectory) > 1:
            trajectory_points = [(int(p['position'][0]), int(p['position'][1])) 
                                for p in self.current_trajectory]
            
            # Линии траектории с градиентом
            for i in range(1, len(trajectory_points)):
                # Градиент от старых к новым точкам
                alpha = i / len(trajectory_points)
                
                # Оранжевый градиент
                b = int(255 * (1 - alpha))
                g = int(140 * alpha)
                r = 0
                color = (b, g, r)
                
                # Толщина увеличивается к новым точкам
                thickness = int(2 + 3 * alpha)
                
                cv2.line(annotated_frame, trajectory_points[i-1], trajectory_points[i], 
                        color, thickness)
            
            # Точки траектории
            for i, point in enumerate(trajectory_points):
                alpha = i / len(trajectory_points)
                radius = int(3 + 4 * alpha)
                
                b = int(255 * (1 - alpha))
                g = int(140 * alpha)
                r = 0
                color = (b, g, r)
                
                cv2.circle(annotated_frame, point, radius, color, -1)
                cv2.circle(annotated_frame, point, radius+2, (255, 255, 255), 1)
        
        # Отрисовываем текущую детекцию поверх траектории
        if detection:
            box = detection['box']
            center = detection['center']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = map(int, box)
            
            # Bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.ball_color, 3)
            
            # Центр мяча
            center_x, center_y = map(int, center)
            cv2.circle(annotated_frame, (center_x, center_y), 10, self.ball_color, -1)
            cv2.circle(annotated_frame, (center_x, center_y), 14, self.ball_color, 3)
            cv2.circle(annotated_frame, (center_x, center_y), 18, (255, 255, 255), 1)
            
            # Вектор скорости (стрелка направления)
            if self.velocity:
                vx, vy = self.velocity
                # Масштабируем вектор для визуализации
                scale = 5
                end_x = int(center_x + vx * scale)
                end_y = int(center_y + vy * scale)
                
                # Рисуем стрелку
                cv2.arrowedLine(annotated_frame, (center_x, center_y), (end_x, end_y), 
                               self.velocity_color, 3, tipLength=0.3)
                
                # Вычисляем угол направления
                angle = self.get_direction_angle(self.velocity)
                if angle is not None:
                    # Показываем угол и скорость
                    speed = math.sqrt(vx*vx + vy*vy)
                    direction_text = f"Dir: {angle:.0f}deg, Speed: {speed:.1f}"
                    cv2.putText(annotated_frame, direction_text, (x1, y1-45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.velocity_color, 2)
            
            # Информация о мяче
            label = f"Ball {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.ball_color, 3)
            
            # Координаты
            cv2.putText(annotated_frame, f"({center_x}, {center_y})", 
                       (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ball_color, 2)
        
        # Статистика (черный текст для лучшей видимости)
        text_color = (0, 0, 0)  # Черный
        y_pos = 40
        
        # Время видео (если передано)
        if video_time:
            minutes = int(video_time // 60)
            seconds = video_time % 60
            time_text = f"Time: {minutes:02d}:{seconds:05.2f}"
            cv2.putText(annotated_frame, time_text, 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
            y_pos += 40
        
        cv2.putText(annotated_frame, f"Frame: {frame_number}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        y_pos += 40
        
        # Информация о розыгрыше
        if self.current_rally_start_frame:
            rally_text = f"Rally #{self.rally_counter} (Active)"
            cv2.putText(annotated_frame, rally_text, 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Красный для активного
            y_pos += 40
        else:
            rally_text = f"Rallies: {self.rally_counter}"
            cv2.putText(annotated_frame, rally_text, 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
            y_pos += 40
        
        cv2.putText(annotated_frame, f"Detections: {self.total_detections}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        y_pos += 40
        cv2.putText(annotated_frame, f"Current trajectory: {len(self.current_trajectory)}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        y_pos += 40
        cv2.putText(annotated_frame, f"Total trajectories: {self.total_trajectories}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        
        # Вычисляем скорость
        if len(self.current_trajectory) > 1:
            last_two = self.current_trajectory[-2:]
            dx = last_two[1]['position'][0] - last_two[0]['position'][0]
            dy = last_two[1]['position'][1] - last_two[0]['position'][1]
            speed = math.sqrt(dx*dx + dy*dy)
            
            y_pos += 40
            cv2.putText(annotated_frame, f"Speed: {speed:.1f} px/frame", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        
        return annotated_frame

def test_smart_tracking(image_folder="extracted_frames_end"):
    """Тестирование умного трекинга"""
    
    model_path = "runs/segment/train10/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Ошибка: модель не найдена: {model_path}")
        return
    
    if not os.path.exists(image_folder):
        print(f"Ошибка: папка не найдена: {image_folder}")
        return
    
    # Инициализация трекера
    print("Инициализация умного трекера мяча...")
    tracker = SmartBallTracker(model_path)
    
    # Получаем список изображений
    image_files = sorted(list(Path(image_folder).glob("*.jpg")))
    
    if not image_files:
        print(f"Изображения не найдены в папке {image_folder}")
        return
    
    print(f"Найдено {len(image_files)} изображений")
    
    # Создаем папку для результатов
    output_dir = "smart_tracking_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Очищаем папку
    for old_file in Path(output_dir).glob("*.jpg"):
        old_file.unlink()
    
    print("\nНачинаем умный трекинг...")
    print("=" * 60)
    
    # FPS видео
    VIDEO_FPS = 59.94
    
    # Обрабатываем каждое изображение
    for i, image_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"Обрабатываем кадр {i+1}/{len(image_files)}: {image_path.name}")
        
        # Загружаем изображение
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        
        # Извлекаем реальный номер кадра из имени файла (например, frame_039589.jpg -> 39589)
        import re
        match = re.search(r'frame_(\d+)', image_path.name)
        if match:
            real_frame_number = int(match.group(1))
            video_time = real_frame_number / VIDEO_FPS
        else:
            real_frame_number = i + 1
            video_time = None
        
        # Трекинг мяча
        detection = tracker.track_ball(frame, real_frame_number, video_time)
        
        # Отрисовка результатов
        annotated_frame = tracker.draw_tracking(frame, detection, real_frame_number, video_time)
        
        # Сохраняем каждое 2-е изображение
        if i % 2 == 0:
            output_path = Path(output_dir) / f"smart_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_frame)
    
    print("\n" + "=" * 60)
    print("Умный трекинг завершен!")
    print("=" * 60)
    
    print(f"\nИТОГОВАЯ СТАТИСТИКА:")
    print(f"   Всего детекций: {tracker.total_detections}")
    print(f"   Текущая траектория: {len(tracker.current_trajectory)} точек")
    print(f"   Всего траекторий: {tracker.total_trajectories}")
    print(f"   Результаты сохранены в: {output_dir}")
    
    if tracker.total_detections > 0:
        detection_rate = tracker.total_detections / len(image_files) * 100
        print(f"   Процент обнаружения: {detection_rate:.1f}%")
    
    # Сохраняем все траектории
    all_trajectories_data = []
    
    # Текущая траектория
    if tracker.current_trajectory:
        all_trajectories_data.append({
            'type': 'current',
            'points': [
                {
                    'x': float(p['position'][0]),
                    'y': float(p['position'][1]),
                    'confidence': float(p['confidence'])
                }
                for p in tracker.current_trajectory
            ]
        })
    
    # Старые траектории
    for idx, traj in enumerate(tracker.all_trajectories):
        all_trajectories_data.append({
            'type': 'completed',
            'trajectory_id': idx,
            'points': [
                {
                    'x': float(p['position'][0]),
                    'y': float(p['position'][1]),
                    'confidence': float(p['confidence'])
                }
                for p in traj
            ]
        })
    
    with open('smart_trajectories.json', 'w', encoding='utf-8') as f:
        json.dump(all_trajectories_data, f, indent=2, ensure_ascii=False)
    
    print(f"   Траектории сохранены в: smart_trajectories.json")
    
    # Сохраняем информацию о розыгрышах
    if tracker.rallies:
        rallies_data = []
        for rally in tracker.rallies:
            rally_info = {
                'rally_number': rally['rally_number'],
                'start_frame': rally['start_frame'],
                'end_frame': rally['end_frame'],
                'duration_frames': rally['duration_frames'],
                'duration_seconds': float(rally['duration_seconds']) if rally['duration_seconds'] else None,
                'trajectory_points': rally['trajectory_points']
            }
            if rally['start_time']:
                rally_info['start_time'] = float(rally['start_time'])
            if rally['end_time']:
                rally_info['end_time'] = float(rally['end_time'])
            rallies_data.append(rally_info)
        
        with open('rallies.json', 'w', encoding='utf-8') as f:
            json.dump(rallies_data, f, indent=2, ensure_ascii=False)
        
        print(f"   Розыгрыши сохранены в: rallies.json")
        print(f"\n   ОБНАРУЖЕНО РОЗЫГРЫШЕЙ: {len(tracker.rallies)}")
    
    # Открываем папку с результатами
    print("\nОткрываю папку с результатами...")
    os.system(f'explorer "{output_dir}"')

if __name__ == "__main__":
    test_smart_tracking("extracted_frames_end")


