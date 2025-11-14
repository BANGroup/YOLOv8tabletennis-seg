#!/usr/bin/env python3
"""
Профессиональный трекинг мяча с потоковой обработкой
Использует модель train10 и оригинальный алгоритм, который работал
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from collections import deque
import math

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
            'velocity': (0, 0, 255)
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
    
    def draw_tracking(self, frame, detection):
        annotated_frame = frame.copy()
        
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
        
        # КРАСНАЯ СТРЕЛОЧКА ТРАЕКТОРИИ
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
                    
                    end_x = int(center_x + dx_norm * min(speed, 80))
                    end_y = int(center_y + dy_norm * min(speed, 80))
                    
                    cv2.arrowedLine(annotated_frame, (center_x, center_y), 
                                  (end_x, end_y), self.colors['velocity'], 3, tipLength=0.3)
        
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

def add_time_to_frame(frame, elapsed_seconds):
    """Добавить время черным цветом"""
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    time_str = f"{minutes:02d}:{seconds:02d}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(time_str, font, font_scale, thickness)[0]
    
    x = 10
    y = frame.shape[0] - 20
    
    cv2.rectangle(frame, (x - 5, y - text_size[1] - 10), 
                  (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, time_str, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    return frame

def main():
    # Используем модель train10 - та, которая работала!
    model_path = "runs/segment/train10/weights/best.pt"
    video_path = "sources/GX010138.MP4"
    output_dir = "results/tracked_5min_working"
    
    if not os.path.exists(model_path):
        print(f"[!] Модель не найдена: {model_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"[!] Видео не найдено: {video_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Открываю видео...")
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"    FPS: {fps:.2f}")
    print(f"    Кадров: {total_frames}")
    print(f"    Длительность: {total_duration:.1f} сек ({total_duration/60:.1f} мин)")
    
    # Вырезаем 5 минут из середины
    middle_start = (total_duration - 300) / 2
    start_frame = int(middle_start * fps)
    end_frame = start_frame + int(300 * fps)
    
    print(f"    Вырезаю: {middle_start:.1f} - {middle_start + 300:.1f} сек")
    
    print(f"\n[*] Инициализирую трекер с моделью train10...")
    tracker = ProfessionalBallTracker(model_path)
    
    print(f"[*] Инициализирую VideoWriter...")
    output_video = os.path.join(output_dir, "tracked_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("[!] Ошибка при открытии VideoWriter")
        return
    
    print(f"\n[*] Обрабатываю видео потоком (train10 модель)...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    processed_count = 0
    start_time = time.time()
    
    for frame_idx in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        detection = tracker.track_ball(frame)
        annotated = tracker.draw_tracking(frame, detection)
        elapsed = (frame_idx / fps)
        annotated = add_time_to_frame(annotated, elapsed)
        out.write(annotated)
        
        processed_count += 1
        
        if (processed_count) % 500 == 0:
            elapsed_time = time.time() - start_time
            print(f"    Обработано {processed_count}/{end_frame - start_frame} кадров ({elapsed_time:.1f}сек)")
    
    cap.release()
    out.release()
    
    print(f"\n[+] ВИДЕО СОЗДАНО: {output_video}")
    print(f"\n[=] СТАТИСТИКА:")
    print(f"    Детекций: {tracker.total_detections}")
    print(f"    Пропущено: {tracker.missed_frames}")
    print(f"    Траектория: {len(tracker.trajectory)}")
    print(f"    Kalman предсказаний: {tracker.kalman_predictions_used}")
    
    if tracker.total_detections > 0:
        detection_rate = tracker.total_detections / processed_count * 100
        print(f"    Процент обнаружения: {detection_rate:.1f}%")
    
    if tracker.velocities:
        print(f"    Средняя скорость: {np.mean(list(tracker.velocities)):.1f} px/frame")
    
    total_time = time.time() - start_time
    print(f"\n[*] Общее время обработки: {total_time:.1f} сек")

if __name__ == "__main__":
    main()

