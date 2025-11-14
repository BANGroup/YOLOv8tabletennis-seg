#!/usr/bin/env python3
"""
Трекинг мяча 2 сек (06:35-06:37):
- Детектирует перелом траектории (резкое изменение направления)
- Применяет Kalman фильтр для сглаживания
- Показывает статистику в левом верхнем углу (как было)
- Выводит телеметрию с анализом переломов
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import math
from collections import deque
import json

class KalmanFilter:
    """Kalman фильтр для сглаживания и предсказания"""
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

class TrajectoryBreakDetector:
    """Детектор переломов траектории"""
    
    def __init__(self, angle_threshold=30, speed_threshold=10):
        """
        angle_threshold: минимальный угол для обнаружения перелома (градусы)
        speed_threshold: минимальная разница скоростей (px/frame)
        """
        self.angle_threshold = angle_threshold
        self.speed_threshold = speed_threshold
        self.direction_history = deque(maxlen=5)
        self.speed_history = deque(maxlen=5)
        self.breaks_detected = []
    
    def detect_break(self, pos1, pos2, pos3):
        """
        Анализирует три последовательные позиции для определения перелома
        pos1 -> pos2 -> pos3
        """
        if not pos1 or not pos2 or not pos3:
            return None
        
        # Вектор 1: pos1 -> pos2
        v1 = (pos2[0] - pos1[0], pos2[1] - pos1[1])
        # Вектор 2: pos2 -> pos3
        v2 = (pos3[0] - pos2[0], pos3[1] - pos2[1])
        
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 < 0.1 or len2 < 0.1:
            return None
        
        # Нормализуем векторы
        v1_norm = (v1[0] / len1, v1[1] / len1)
        v2_norm = (v2[0] / len2, v2[1] / len2)
        
        # Скалярное произведение
        dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Угол между векторами
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)
        
        # Разница скоростей
        speed_diff = abs(len2 - len1)
        
        # Перелом - если угол > threshold или резкое изменение скорости
        is_break = angle_deg > self.angle_threshold or speed_diff > self.speed_threshold
        
        return {
            'angle': angle_deg,
            'speed_before': len1,
            'speed_after': len2,
            'speed_diff': speed_diff,
            'is_break': is_break
        }

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.kalman = KalmanFilter()
        self.break_detector = TrajectoryBreakDetector(angle_threshold=25, speed_threshold=5)
        
        self.trajectory = deque(maxlen=100)
        self.velocities = deque(maxlen=15)
        self.accelerations = deque(maxlen=15)
        self.total_detections = 0
        self.missed_frames = 0
        self.consecutive_detections = 0
        self.kalman_predictions_used = 0
        self.breaks_count = 0
        
    def detect_ball(self, frame):
        results = self.model.predict(frame, conf=0.1, verbose=False)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if class_id == 0:  # Ball
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    return {'center': center, 'conf': conf}
        return None
    
    def track_ball(self, frame):
        detection = self.detect_ball(frame)
        
        if detection:
            self.total_detections += 1
            self.missed_frames = 0
            self.consecutive_detections += 1
            
            center = detection['center']
            self.kalman.update(center)
            self.trajectory.append(center)
            
            # Вычисляем скорость
            if len(self.trajectory) > 1:
                last_pos = self.trajectory[-2]
                dx = center[0] - last_pos[0]
                dy = center[1] - last_pos[1]
                speed = math.sqrt(dx*dx + dy*dy)
                self.velocities.append(speed)
                
                # Ускорение
                if len(self.velocities) > 1:
                    accel = self.velocities[-1] - self.velocities[-2]
                    self.accelerations.append(accel)
                
                # Анализируем перелом (если есть 3+ точки)
                if len(self.trajectory) >= 3:
                    pos1 = self.trajectory[-3]
                    pos2 = self.trajectory[-2]
                    pos3 = self.trajectory[-1]
                    
                    break_info = self.break_detector.detect_break(pos1, pos2, pos3)
                    if break_info and break_info['is_break']:
                        self.breaks_count += 1
                        self.break_detector.breaks_detected.append({
                            'frame': self.total_detections,
                            'angle': break_info['angle'],
                            'speed_diff': break_info['speed_diff']
                        })
        else:
            self.missed_frames += 1
            self.consecutive_detections = 0
            
            if self.kalman.initialized:
                pred = self.kalman.predict()
                if pred:
                    self.kalman_predictions_used += 1
        
        return detection
    
    def draw_tracking(self, frame, detection):
        annotated = frame.copy()
        
        # Траектория с градиентом
        if len(self.trajectory) > 1:
            points = [(int(p[0]), int(p[1])) for p in self.trajectory]
            for i in range(1, len(points)):
                alpha = i / len(points)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                cv2.line(annotated, points[i-1], points[i], color, 2)
        
        # Текущая позиция мяча
        if detection:
            cx, cy = map(int, detection['center'])
            cv2.circle(annotated, (cx, cy), 8, (0, 255, 0), -1)
            cv2.circle(annotated, (cx, cy), 11, (0, 255, 0), 2)
        
        # Стрелочка скорости
        if len(self.velocities) > 0 and detection:
            cx, cy = map(int, detection['center'])
            speed = self.velocities[-1]
            if len(self.trajectory) > 1:
                last_pos = self.trajectory[-2]
                dx = cx - last_pos[0]
                dy = cy - last_pos[1]
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx_norm = dx / length
                    dy_norm = dy / length
                    end_x = int(cx + dx_norm * min(speed, 80))
                    end_y = int(cy + dy_norm * min(speed, 80))
                    cv2.arrowedLine(annotated, (cx, cy), (end_x, end_y), (0, 0, 255), 3, tipLength=0.3)
        
        # СТАТИСТИКА В ЛЕВОМ ВЕРХНЕМ УГЛУ (как было)
        y_pos = 30
        cv2.putText(annotated, f"Detections: {self.total_detections}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(annotated, f"Missed: {self.missed_frames}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(annotated, f"Trajectory: {len(self.trajectory)}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(annotated, f"Breaks: {self.breaks_count}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        if len(self.velocities) > 0:
            avg_speed = np.mean(list(self.velocities)[-5:])
            y_pos += 30
            cv2.putText(annotated, f"Speed: {avg_speed:.1f} px/frame", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.kalman_predictions_used > 0:
            y_pos += 30
            cv2.putText(annotated, f"Kalman: {self.kalman_predictions_used}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated

def main():
    model_path = "runs/segment/train10/weights/best.pt"
    video_path = "sources/GX010138.MP4"
    output_dir = "results/track_2sec_breaks"
    
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print("[!] Ошибка: модель или видео не найдены")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Инициализирую трекер с детектором переломов...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 2 сек (06:35 - 06:37)
    start_time = 6 * 60 + 35
    end_time = 6 * 60 + 37
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    tracker = BallTracker(model_path)
    
    print("[*] Создаю видео с анализом переломов траектории...")
    output_video = os.path.join(output_dir, "tracked_2sec_breaks.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("[!] Ошибка VideoWriter")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed_count = 0
    
    for frame_idx in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        # ТРЕКИНГ С ПРИМЕНЕНИЕМ KALMAN ФИЛЬТРА
        detection = tracker.track_ball(frame)
        
        # Добавляем время
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        ms = int(((frame_idx / fps) % 1) * 100)
        time_str = f"{minutes:02d}:{seconds:02d}.{ms:02d}"
        
        # Рисуем трекинг
        annotated = tracker.draw_tracking(frame, detection)
        
        # Добавляем время
        cv2.putText(annotated, f"Time: {time_str}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Записываем
        out.write(annotated)
        
        processed_count += 1
        
        if (processed_count) % 30 == 0:
            print(f"    Обработано {processed_count} кадров, переломов: {tracker.breaks_count}")
    
    cap.release()
    out.release()
    
    print(f"\n[+] ВИДЕО: {output_video}")
    print(f"\n[=] АНАЛИЗ ТРАЕКТОРИИ:")
    print(f"    Кадров обработано: {processed_count}")
    print(f"    Детекций мяча: {tracker.total_detections}")
    print(f"    Пропущено кадров: {tracker.missed_frames}")
    print(f"    ПЕРЕЛОМОВ ТРАЕКТОРИИ ОБНАРУЖЕНО: {tracker.breaks_count}")
    print(f"    Kalman предсказаний: {tracker.kalman_predictions_used}")
    
    # Сохраняем анализ
    analysis = {
        'total_detections': tracker.total_detections,
        'missed_frames': tracker.missed_frames,
        'trajectory_breaks': tracker.breaks_count,
        'kalman_predictions': tracker.kalman_predictions_used,
        'breaks_details': tracker.break_detector.breaks_detected,
        'tracking_method': 'Kalman Filter + Trajectory Break Detection'
    }
    
    json_path = os.path.join(output_dir, "trajectory_analysis.json")
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"    Анализ сохранен: {json_path}")
    print(f"\n[*] ПРИМЕНЕН ТРЕКИНГ: Kalman фильтр + детектор переломов")

if __name__ == "__main__":
    main()

