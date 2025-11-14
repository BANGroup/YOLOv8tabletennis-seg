#!/usr/bin/env python3
"""
Профессиональный трекинг мяча на основе лучших практик
Использует Kalman фильтр, адаптивную детекцию и валидацию траектории
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

class KalmanFilter:
    """Kalman фильтр для предсказания позиции мяча"""
    
    def __init__(self):
        # Состояние: [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float32)
        
        # Ковариационная матрица состояния
        self.P = np.eye(4, dtype=np.float32) * 1000
        
        # Матрица перехода состояния (константная скорость)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Матрица наблюдения (наблюдаем только позицию)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Процессная шумовая ковариация
        self.Q = np.eye(4, dtype=np.float32) * 0.1
        
        # Шумовая ковариация измерений
        self.R = np.eye(2, dtype=np.float32) * 10
        
        self.initialized = False
    
    def predict(self):
        """Предсказание следующего состояния"""
        if not self.initialized:
            return None
        
        # Предсказание состояния
        self.state = self.F @ self.state
        
        # Предсказание ковариации
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return (self.state[0], self.state[1])
    
    def update(self, measurement):
        """Обновление на основе измерения"""
        x, y = measurement
        
        if not self.initialized:
            # Инициализация
            self.state = np.array([x, y, 0, 0], dtype=np.float32)
            self.initialized = True
            return
        
        # Измерение
        z = np.array([x, y], dtype=np.float32)
        
        # Инновация (разница между предсказанием и измерением)
        y_innov = z - self.H @ self.state
        
        # Инновационная ковариация
        S = self.H @ self.P @ self.H.T + self.R
        
        # Коэффициент усиления Kalman
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Обновление состояния
        self.state = self.state + K @ y_innov
        
        # Обновление ковариации
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return (self.state[0], self.state[1])

class ProfessionalBallTracker:
    """Профессиональный трекер мяча с Kalman фильтром"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Kalman фильтр
        self.kalman = KalmanFilter()
        
        # Адаптивные параметры детекции
        self.base_confidence = 0.25
        self.min_confidence = 0.1
        self.max_confidence = 0.6
        self.current_confidence = self.base_confidence
        
        # Фильтры размера мяча (адаптивные)
        self.min_ball_area = 30
        self.max_ball_area = 2500
        self.expected_sizes = deque(maxlen=10)  # История размеров
        
        # Физические ограничения
        self.max_speed = 300  # пикселей/кадр
        self.max_acceleration = 80
        self.gravity = 0.8
        
        # История для анализа
        self.trajectory = deque(maxlen=100)
        self.velocities = deque(maxlen=15)
        self.confidences = deque(maxlen=20)
        self.detection_history = deque(maxlen=10)
        
        # Статистика
        self.total_detections = 0
        self.missed_frames = 0
        self.consecutive_detections = 0
        self.kalman_predictions_used = 0
        
        # Цвета
        self.colors = {
            'ball': (0, 255, 0),
            'trajectory': (255, 0, 255),
            'prediction': (0, 255, 255),
            'kalman': (255, 255, 0),
            'velocity': (255, 0, 0)
        }
    
    def adaptive_confidence(self):
        """Адаптивная настройка порога уверенности"""
        if len(self.confidences) < 3:
            return self.current_confidence
        
        recent_conf = list(self.confidences)[-5:]
        avg_conf = np.mean(recent_conf)
        
        # Повышаем порог при стабильных детекциях
        if self.consecutive_detections > 5 and avg_conf > 0.4:
            self.current_confidence = min(self.max_confidence, self.current_confidence + 0.03)
        # Снижаем при пропусках
        elif self.missed_frames > 2:
            self.current_confidence = max(self.min_confidence, self.current_confidence - 0.05)
        
        return self.current_confidence
    
    def update_expected_size(self, size):
        """Обновление ожидаемого размера мяча"""
        self.expected_sizes.append(size)
        if len(self.expected_sizes) > 3:
            self.expected_sizes.popleft()
    
    def get_expected_size(self):
        """Получить ожидаемый размер мяча"""
        if len(self.expected_sizes) == 0:
            return 400  # Дефолтное значение
        
        return np.median(list(self.expected_sizes))
    
    def validate_detection(self, detection, frame_shape):
        """Валидация детекции по множеству критериев"""
        if not detection:
            return False
        
        box = detection['box']
        center = detection['center']
        confidence = detection['confidence']
        size = detection['size']
        
        # 1. Проверка размера
        expected_size = self.get_expected_size()
        size_ratio = size / expected_size if expected_size > 0 else 1.0
        
        if size < self.min_ball_area or size > self.max_ball_area:
            return False
        
        # Если есть история размеров, проверяем согласованность
        if len(self.expected_sizes) > 0:
            if size_ratio < 0.3 or size_ratio > 3.0:
                return False
        
        # 2. Проверка формы (соотношение сторон)
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            return False
        
        aspect_ratio = width / height
        if aspect_ratio < 0.6 or aspect_ratio > 1.4:
            return False
        
        # 3. Проверка скорости (если есть предыдущие позиции)
        if len(self.trajectory) > 0:
            last_pos = self.trajectory[-1]['position']
            dx = center[0] - last_pos[0]
            dy = center[1] - last_pos[1]
            speed = math.sqrt(dx*dx + dy*dy)
            
            if speed > self.max_speed:
                return False
        
        # 4. Проверка через Kalman фильтр (если инициализирован)
        if self.kalman.initialized:
            predicted = self.kalman.predict()
            if predicted:
                pred_x, pred_y = predicted
                distance = math.sqrt((center[0] - pred_x)**2 + (center[1] - pred_y)**2)
                
                # Если детекция далеко от предсказания, возможно ложная
                if distance > 150:
                    return False
        
        return True
    
    def detect_ball(self, frame):
        """Детекция мяча с адаптивными параметрами"""
        conf_threshold = self.adaptive_confidence()
        
        results = self.model.predict(
            frame,
            conf=conf_threshold,
            iou=0.3,
            classes=[0],
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
                        'size': size,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                    
                    if self.validate_detection(detection, frame.shape):
                        detections.append(detection)
        
        # Выбираем лучшую детекцию
        if detections:
            def score_detection(det):
                expected_size = self.get_expected_size()
                size_score = 1.0 - min(abs(det['size'] - expected_size) / expected_size, 1.0)
                
                # Приоритет ближайшим к предсказанию Kalman
                kalman_score = 1.0
                if self.kalman.initialized:
                    predicted = self.kalman.predict()
                    if predicted:
                        pred_x, pred_y = predicted
                        distance = math.sqrt((det['center'][0] - pred_x)**2 + 
                                           (det['center'][1] - pred_y)**2)
                        kalman_score = 1.0 / (1.0 + distance / 50.0)
                
                return det['confidence'] * 0.5 + size_score * 0.3 + kalman_score * 0.2
            
            best_detection = max(detections, key=score_detection)
            return best_detection
        
        return None
    
    def track_ball(self, frame):
        """Основной метод трекинга с Kalman фильтром"""
        # 1. Детекция мяча
        detection = self.detect_ball(frame)
        
        # 2. Если мяч не найден, используем предсказание Kalman
        if detection is None:
            if self.kalman.initialized:
                predicted_pos = self.kalman.predict()
                if predicted_pos:
                    # Создаем "виртуальную" детекцию из предсказания
                    pred_x, pred_y = predicted_pos
                    expected_size = self.get_expected_size()
                    size = int(math.sqrt(expected_size))
                    
                    detection = {
                        'box': (pred_x - size/2, pred_y - size/2, 
                               pred_x + size/2, pred_y + size/2),
                        'center': (pred_x, pred_y),
                        'confidence': 0.5,  # Средняя уверенность для предсказания
                        'size': expected_size,
                        'width': size,
                        'height': size,
                        'is_prediction': True
                    }
                    self.kalman_predictions_used += 1
        
        # 3. Обновляем статистику
        if detection:
            self.total_detections += 1
            self.missed_frames = 0
            self.consecutive_detections += 1
            
            center = detection['center']
            
            # Обновляем Kalman фильтр
            if not detection.get('is_prediction', False):
                self.kalman.update(center)
                self.update_expected_size(detection['size'])
            
            # Добавляем в траекторию
            self.trajectory.append({
                'position': center,
                'confidence': detection['confidence'],
                'timestamp': time.time(),
                'size': detection['size'],
                'is_prediction': detection.get('is_prediction', False)
            })
            
            self.confidences.append(detection['confidence'])
            
            # Вычисляем скорость
            if len(self.trajectory) > 1:
                last_pos = self.trajectory[-2]['position']
                dx = center[0] - last_pos[0]
                dy = center[1] - last_pos[1]
                speed = math.sqrt(dx*dx + dy*dy)
                self.velocities.append(speed)
            
            self.detection_history.append(True)
        else:
            self.missed_frames += 1
            self.consecutive_detections = 0
            self.detection_history.append(False)
            
            # Предсказываем следующую позицию даже при пропуске
            if self.kalman.initialized:
                self.kalman.predict()
        
        return detection
    
    def draw_tracking(self, frame, detection):
        """Профессиональная отрисовка результатов трекинга"""
        annotated_frame = frame.copy()
        
        # Отрисовываем текущую детекцию
        if detection:
            box = detection['box']
            center = detection['center']
            confidence = detection['confidence']
            is_prediction = detection.get('is_prediction', False)
            
            # Выбираем цвет в зависимости от типа
            color = self.colors['kalman'] if is_prediction else self.colors['ball']
            
            x1, y1, x2, y2 = map(int, box)
            
            # Bounding box
            thickness = 2 if is_prediction else 3
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Центр мяча
            center_x, center_y = map(int, center)
            radius = 6 if is_prediction else 8
            cv2.circle(annotated_frame, (center_x, center_y), radius, color, -1)
            cv2.circle(annotated_frame, (center_x, center_y), radius+3, color, 2)
            
            # Метка
            label_type = "PRED" if is_prediction else "BALL"
            label = f"{label_type} {confidence:.3f}"
            cv2.putText(annotated_frame, label, (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Отрисовываем траекторию
        if len(self.trajectory) > 1:
            trajectory_points = []
            for i, point in enumerate(self.trajectory):
                pos = point['position']
                trajectory_points.append((int(pos[0]), int(pos[1])))
            
            # Линии траектории с градиентом
            for i in range(1, len(trajectory_points)):
                alpha = i / len(trajectory_points)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                
                # Толщина по скорости
                thickness = 2
                if i-1 < len(self.velocities):
                    speed_factor = min(self.velocities[i-1] / 100, 1.5)
                    thickness = int(2 + speed_factor * 2)
                
                cv2.line(annotated_frame, trajectory_points[i-1], trajectory_points[i], 
                        color, thickness)
            
            # Точки траектории
            for i, point in enumerate(trajectory_points):
                is_pred = self.trajectory[i].get('is_prediction', False)
                alpha = i / len(trajectory_points)
                radius = 2 if is_pred else int(2 + 2 * alpha)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                
                cv2.circle(annotated_frame, point, radius, color, -1)
        
        # Предсказание Kalman
        if self.kalman.initialized:
            predicted = self.kalman.predict()
            if predicted:
                pred_x, pred_y = map(int, predicted)
                cv2.circle(annotated_frame, (pred_x, pred_y), 5, self.colors['prediction'], 2)
        
        # Вектор скорости
        if len(self.velocities) > 0 and detection:
            center_x, center_y = map(int, detection['center'])
            speed = self.velocities[-1]
            
            if len(self.trajectory) > 1:
                last_pos = self.trajectory[-2]['position']
                dx = center_x - last_pos[0]
                dy = center_y - last_pos[1]
                
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx_norm = dx / length
                    dy_norm = dy / length
                    
                    end_x = int(center_x + dx_norm * min(speed, 50))
                    end_y = int(center_y + dy_norm * min(speed, 50))
                    
                    cv2.arrowedLine(annotated_frame, (center_x, center_y), 
                                  (end_x, end_y), self.colors['velocity'], 3)
        
        # Статистика
        y_pos = 30
        cv2.putText(annotated_frame, f"Detections: {self.total_detections}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(annotated_frame, f"Missed: {self.missed_frames}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(annotated_frame, f"Conf: {self.current_confidence:.2f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(annotated_frame, f"Trajectory: {len(self.trajectory)}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(self.velocities) > 0:
            avg_speed = np.mean(list(self.velocities)[-5:])
            y_pos += 30
            cv2.putText(annotated_frame, f"Speed: {avg_speed:.1f} px/frame", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.kalman_predictions_used > 0:
            y_pos += 30
            cv2.putText(annotated_frame, f"Kalman: {self.kalman_predictions_used}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame

def test_professional_tracking(image_folder="extracted_frames_middle"):
    """Тестирование профессионального трекинга"""
    
    model_path = "runs/segment/train10/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Ошибка: модель не найдена: {model_path}")
        return
    
    if not os.path.exists(image_folder):
        print(f"Ошибка: папка не найдена: {image_folder}")
        return
    
    # Инициализация трекера
    print("Инициализация профессионального трекера мяча...")
    tracker = ProfessionalBallTracker(model_path)
    
    # Получаем список изображений
    image_files = sorted(list(Path(image_folder).glob("*.jpg")) + 
                        list(Path(image_folder).glob("*.png")))
    
    if not image_files:
        print(f"Изображения не найдены в папке {image_folder}")
        return
    
    print(f"Найдено {len(image_files)} изображений")
    
    # Создаем папку для результатов (отдельную для этого запуска)
    base_output_dir = Path("professional_tracking_results")
    output_dir = base_output_dir / f"tracking_end_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nНачинаем профессиональный трекинг...")
    print("=" * 60)
    
    # Обрабатываем каждое изображение
    for i, image_path in enumerate(image_files):
        if i % 30 == 0:
            print(f"Обрабатываем кадр {i+1}/{len(image_files)}: {image_path.name}")
        
        # Загружаем изображение
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        
        # Трекинг мяча
        detection = tracker.track_ball(frame)
        
        # Отрисовка результатов
        annotated_frame = tracker.draw_tracking(frame, detection)
        
        # Сохраняем результат (каждый 5-й кадр для экономии места)
        if i % 5 == 0:
            output_path = Path(output_dir) / f"tracked_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_frame)
    
    print("\n" + "=" * 60)
    print("Профессиональный трекинг завершен!")
    print("=" * 60)
    
    print(f"\nИТОГОВАЯ СТАТИСТИКА:")
    print(f"   Всего детекций: {tracker.total_detections}")
    print(f"   Пропущено кадров: {tracker.missed_frames}")
    print(f"   Точки траектории: {len(tracker.trajectory)}")
    print(f"   Использовано предсказаний Kalman: {tracker.kalman_predictions_used}")
    
    if tracker.total_detections > 0:
        detection_rate = tracker.total_detections / len(image_files) * 100
        print(f"   Процент обнаружения: {detection_rate:.1f}%")
    
    if tracker.velocities:
        print(f"   Средняя скорость: {np.mean(list(tracker.velocities)):.1f} px/frame")
        print(f"   Максимальная скорость: {np.max(list(tracker.velocities)):.1f} px/frame")
    
    print(f"   Результаты сохранены в: {output_dir}")
    
    # Сохраняем траекторию
    if tracker.trajectory:
        trajectory_data = []
        for point in tracker.trajectory:
            trajectory_data.append({
                'x': float(point['position'][0]),
                'y': float(point['position'][1]),
                'confidence': float(point['confidence']),
                'size': float(point['size']),
                'is_prediction': point.get('is_prediction', False),
                'timestamp': point['timestamp']
            })
        
        with open('professional_trajectory.json', 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
        
        print(f"   Траектория сохранена в: professional_trajectory.json")

if __name__ == "__main__":
    test_professional_tracking("extracted_frames")
