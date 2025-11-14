#!/usr/bin/env python3
"""
Применение профессионального трекера к видео:
1. Вырезает 5 минут из середины видео
2. Обрабатывает каждый кадр с профессиональным трекингом
3. Создает видео с полной длиной и всеми кадрами
4. Добавляет время и информацию на изображение
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
from datetime import timedelta

class KalmanFilter:
    """Kalman фильтр для предсказания позиции мяча"""
    
    def __init__(self):
        self.state = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1000
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 0.1
        self.R = np.eye(2, dtype=np.float32) * 10
        self.initialized = False
    
    def predict(self):
        if not self.initialized:
            return None
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (self.state[0], self.state[1])
    
    def update(self, measurement):
        x, y = measurement
        if not self.initialized:
            self.state = np.array([x, y, 0, 0], dtype=np.float32)
            self.initialized = True
            return
        z = np.array([x, y], dtype=np.float32)
        y_innov = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_innov
        self.P = (np.eye(4) - K @ self.H) @ self.P

class ProfessionalBallTracker:
    """Профессиональный трекер мяча с Kalman фильтром"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.kalman = KalmanFilter()
        self.base_confidence = 0.25
        self.min_confidence = 0.1
        self.max_confidence = 0.6
        self.current_confidence = self.base_confidence
        self.min_ball_area = 30
        self.max_ball_area = 2500
        self.expected_sizes = deque(maxlen=10)
        self.max_speed = 300
        self.max_acceleration = 80
        self.gravity = 0.8
        self.trajectory = deque(maxlen=100)
        self.velocities = deque(maxlen=15)
        self.confidences = deque(maxlen=20)
        self.detection_history = deque(maxlen=10)
        self.total_detections = 0
        self.missed_frames = 0
        self.consecutive_detections = 0
        self.kalman_predictions_used = 0
        self.colors = {
            'ball': (0, 255, 0),
            'trajectory': (255, 0, 255),
            'prediction': (0, 255, 255),
            'kalman': (255, 255, 0),
            'velocity': (255, 0, 0)
        }
    
    def adaptive_confidence(self):
        if len(self.confidences) < 3:
            return self.current_confidence
        recent_conf = list(self.confidences)[-5:]
        avg_conf = np.mean(recent_conf)
        if self.consecutive_detections > 5 and avg_conf > 0.4:
            self.current_confidence = min(self.max_confidence, self.current_confidence + 0.03)
        elif self.missed_frames > 2:
            self.current_confidence = max(self.min_confidence, self.current_confidence - 0.05)
        return self.current_confidence
    
    def update_expected_size(self, size):
        self.expected_sizes.append(size)
        if len(self.expected_sizes) > 3:
            self.expected_sizes.popleft()
    
    def get_expected_size(self):
        if len(self.expected_sizes) == 0:
            return 400
        return np.median(list(self.expected_sizes))
    
    def validate_detection(self, detection, frame_shape):
        if not detection:
            return False
        box = detection['box']
        center = detection['center']
        confidence = detection['confidence']
        size = detection['size']
        expected_size = self.get_expected_size()
        size_ratio = size / expected_size if expected_size > 0 else 1.0
        if size < self.min_ball_area or size > self.max_ball_area:
            return False
        if len(self.expected_sizes) > 0:
            if size_ratio < 0.3 or size_ratio > 3.0:
                return False
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return False
        aspect_ratio = width / height
        if aspect_ratio < 0.6 or aspect_ratio > 1.4:
            return False
        if len(self.trajectory) > 0:
            last_pos = self.trajectory[-1]['position']
            dx = center[0] - last_pos[0]
            dy = center[1] - last_pos[1]
            speed = math.sqrt(dx*dx + dy*dy)
            if speed > self.max_speed:
                return False
        if self.kalman.initialized:
            predicted = self.kalman.predict()
            if predicted:
                pred_x, pred_y = predicted
                distance = math.sqrt((center[0] - pred_x)**2 + (center[1] - pred_y)**2)
                if distance > 150:
                    return False
        return True
    
    def detect_ball(self, frame):
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
        if detections:
            def score_detection(det):
                expected_size = self.get_expected_size()
                size_score = 1.0 - min(abs(det['size'] - expected_size) / expected_size, 1.0)
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
        detection = self.detect_ball(frame)
        if detection is None:
            if self.kalman.initialized:
                predicted_pos = self.kalman.predict()
                if predicted_pos:
                    pred_x, pred_y = predicted_pos
                    expected_size = self.get_expected_size()
                    size = int(math.sqrt(expected_size))
                    detection = {
                        'box': (pred_x - size/2, pred_y - size/2, 
                               pred_x + size/2, pred_y + size/2),
                        'center': (pred_x, pred_y),
                        'confidence': 0.5,
                        'size': expected_size,
                        'width': size,
                        'height': size,
                        'is_prediction': True
                    }
                    self.kalman_predictions_used += 1
        if detection:
            self.total_detections += 1
            self.missed_frames = 0
            self.consecutive_detections += 1
            center = detection['center']
            if not detection.get('is_prediction', False):
                self.kalman.update(center)
                self.update_expected_size(detection['size'])
            self.trajectory.append({
                'position': center,
                'confidence': detection['confidence'],
                'timestamp': time.time(),
                'size': detection['size'],
                'is_prediction': detection.get('is_prediction', False)
            })
            self.confidences.append(detection['confidence'])
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
            if self.kalman.initialized:
                self.kalman.predict()
        return detection
    
    def draw_tracking(self, frame, detection, frame_info=None):
        """Профессиональная отрисовка с информацией на столе"""
        annotated_frame = frame.copy()
        
        # Текущая детекция
        if detection:
            box = detection['box']
            center = detection['center']
            confidence = detection['confidence']
            is_prediction = detection.get('is_prediction', False)
            color = self.colors['kalman'] if is_prediction else self.colors['ball']
            x1, y1, x2, y2 = map(int, box)
            thickness = 2 if is_prediction else 3
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            center_x, center_y = map(int, center)
            radius = 6 if is_prediction else 8
            cv2.circle(annotated_frame, (center_x, center_y), radius, color, -1)
            cv2.circle(annotated_frame, (center_x, center_y), radius+3, color, 2)
            label_type = "PRED" if is_prediction else "BALL"
            label = f"{label_type} {confidence:.3f}"
            cv2.putText(annotated_frame, label, (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Траектория
        if len(self.trajectory) > 1:
            trajectory_points = []
            for i, point in enumerate(self.trajectory):
                pos = point['position']
                trajectory_points.append((int(pos[0]), int(pos[1])))
            for i in range(1, len(trajectory_points)):
                alpha = i / len(trajectory_points)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                thickness = 2
                if i-1 < len(self.velocities):
                    speed_factor = min(self.velocities[i-1] / 100, 1.5)
                    thickness = int(2 + speed_factor * 2)
                cv2.line(annotated_frame, trajectory_points[i-1], trajectory_points[i], 
                        color, thickness)
            for i, point in enumerate(trajectory_points):
                is_pred = self.trajectory[i].get('is_prediction', False)
                alpha = i / len(trajectory_points)
                radius = 2 if is_pred else int(2 + 2 * alpha)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                cv2.circle(annotated_frame, point, radius, color, -1)
        
        # Информация на столе (на черной полосе внизу)
        h, w = annotated_frame.shape[:2]
        info_height = 60
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, h - info_height), (w, h), (0, 0, 0), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0)
        
        # Время
        if frame_info and 'time' in frame_info:
            time_str = frame_info['time']
            cv2.putText(annotated_frame, f"Time: {time_str}", 
                       (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Статистика
        y_pos = h - 20
        cv2.putText(annotated_frame, f"Det: {self.total_detections} | Frame: {len(self.trajectory)}", 
                   (w - 350, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame

def extract_middle_5min_from_video(input_video, output_dir):
    """Вырезает 5 минут из середины видео"""
    print(f"[*] Открываю видео: {input_video}")
    cap = cv2.VideoCapture(input_video)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    
    print(f"    FPS: {fps:.2f}")
    print(f"    Всего кадров: {total_frames}")
    print(f"    Длительность: {total_duration:.1f} сек ({total_duration/60:.1f} мин)")
    
    # Найти начало середины (5 минут = 300 секунд)
    middle_start = (total_duration - 300) / 2  # Начало 5-минутного отрезка
    start_frame = int(middle_start * fps)
    end_frame = start_frame + int(300 * fps)
    
    print(f"    Вырезаю с {middle_start:.1f} до {middle_start + 300:.1f} сек")
    print(f"    Кадры: {start_frame} - {end_frame}")
    
    # Читаем кадры
    extracted_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for i in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        extracted_frames.append(frame)
        if (i + 1) % 100 == 0:
            print(f"    Прочитано {i + 1}/{end_frame - start_frame} кадров")
    
    cap.release()
    print(f"[+] Загружено {len(extracted_frames)} кадров")
    return extracted_frames, fps

def process_and_create_video(frames, fps, output_dir, model_path):
    """Обрабатывает кадры с трекингом и создает видео"""
    
    if not os.path.exists(model_path):
        print(f"[!] Ошибка: модель не найдена: {model_path}")
        return
    
    print(f"[*] Инициализирую профессиональный трекер...")
    tracker = ProfessionalBallTracker(model_path)
    
    print(f"[*] Обрабатываю {len(frames)} кадров с трекингом...")
    
    h, w = frames[0].shape[:2]
    processed_frames = []
    
    for idx, frame in enumerate(frames):
        # Трекинг мяча
        detection = tracker.track_ball(frame)
        
        # Вычисляем время в видео
        frame_time = idx / fps
        minutes = int(frame_time // 60)
        seconds = int(frame_time % 60)
        milliseconds = int((frame_time % 1) * 1000)
        time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        
        frame_info = {'time': time_str}
        
        # Отрисовка с информацией
        annotated = tracker.draw_tracking(frame, detection, frame_info)
        processed_frames.append(annotated)
        
        if (idx + 1) % 50 == 0:
            print(f"    Обработано {idx + 1}/{len(frames)} кадров")
    
    # Создаем видео
    print(f"[*] Создаю видео ({len(processed_frames)} кадров)...")
    output_video = os.path.join(output_dir, "tracked_video_5min.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    for idx, frame in enumerate(processed_frames):
        out.write(frame)
        if (idx + 1) % 50 == 0:
            print(f"    Написано {idx + 1}/{len(processed_frames)} кадров")
    
    out.release()
    print(f"[+] Видео создано: {output_video}")
    
    # Статистика
    print(f"\n[=] ИТОГОВАЯ СТАТИСТИКА:")
    print(f"    Всего детекций: {tracker.total_detections}")
    print(f"    Пропущено кадров: {tracker.missed_frames}")
    print(f"    Точки траектории: {len(tracker.trajectory)}")
    print(f"    Использовано предсказаний Kalman: {tracker.kalman_predictions_used}")
    if tracker.total_detections > 0:
        detection_rate = tracker.total_detections / len(frames) * 100
        print(f"    Процент обнаружения: {detection_rate:.1f}%")
    if tracker.velocities:
        print(f"    Средняя скорость: {np.mean(list(tracker.velocities)):.1f} px/frame")
    
    return output_video

def main():
    # Пути
    input_video = "sources/GX010138.MP4"
    output_base_dir = "results/tracked_5min_professional"
    model_path = "runs/segment/train9/weights/best.pt"
    
    # Проверяем файлы
    if not os.path.exists(input_video):
        print(f"[!] Ошибка: видео не найдено: {input_video}")
        return
    
    if not os.path.exists(model_path):
        print(f"[!] Ошибка: модель не найдена: {model_path}")
        print(f"    Ищу альтернативные модели...")
        alternatives = [
            "runs/segment/train8/weights/best.pt",
            "runs/segment/train/weights/best.pt"
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                model_path = alt
                print(f"    Найдена: {model_path}")
                break
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    print("=" * 70)
    print("ПРИМЕНЕНИЕ ПРОФЕССИОНАЛЬНОГО ТРЕКЕРА К ВИДЕО")
    print("=" * 70)
    
    # Шаг 1: Вырезаем 5 минут из середины
    print("\n[ЭТАП 1] Вырезание 5 минут из середины видео...")
    frames, fps = extract_middle_5min_from_video(input_video, output_base_dir)
    
    # Шаг 2: Обрабатываем и создаем видео
    print("\n[ЭТАП 2] Обработка кадров и создание видео...")
    output_video = process_and_create_video(frames, fps, output_base_dir, model_path)
    
    print("\n" + "=" * 70)
    print(f"[+] ГОТОВО! Видео сохранено в: {output_video}")
    print("=" * 70)

if __name__ == "__main__":
    main()

