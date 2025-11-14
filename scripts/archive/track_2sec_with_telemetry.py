#!/usr/bin/env python3
"""
Трекинг мяча 2 сек (06:35-06:37) с ТЕЛЕМЕТРИЕЙ:
- Позиция (X, Y)
- Скорость (px/frame)
- Ускорение
- Траектория
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import math
from collections import deque

class KalmanFilter:
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

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.kalman = KalmanFilter()
        self.trajectory = deque(maxlen=100)
        self.velocities = deque(maxlen=15)
        self.accelerations = deque(maxlen=15)
        self.total_detections = 0
        self.telemetry_data = []
        
    def detect_ball(self, frame):
        results = self.model.predict(frame, conf=0.1, verbose=False)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if class_id == 0:
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    return {'center': center, 'conf': conf, 'box': box}
        return None
    
    def track_ball(self, frame):
        detection = self.detect_ball(frame)
        
        if detection:
            self.total_detections += 1
            center = detection['center']
            self.kalman.update(center)
            self.trajectory.append(center)
            
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
        else:
            if self.kalman.initialized:
                self.kalman.predict()
        
        return detection
    
    def get_telemetry(self, frame_idx, fps, start_time):
        """Получить телеметрию мяча"""
        if not self.trajectory:
            return None
        
        current_time = start_time + frame_idx / fps
        pos = self.trajectory[-1]
        
        telemetry = {
            'frame': frame_idx,
            'time': current_time,
            'position_x': pos[0],
            'position_y': pos[1],
            'velocity': self.velocities[-1] if self.velocities else 0,
            'acceleration': self.accelerations[-1] if self.accelerations else 0,
            'avg_velocity': np.mean(list(self.velocities)) if self.velocities else 0
        }
        
        return telemetry
    
    def draw_tracking_with_telemetry(self, frame, detection, telemetry):
        annotated = frame.copy()
        
        # Траектория
        if len(self.trajectory) > 1:
            points = [(int(p[0]), int(p[1])) for p in self.trajectory]
            for i in range(1, len(points)):
                alpha = i / len(points)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                cv2.line(annotated, points[i-1], points[i], color, 2)
        
        # Текущая позиция
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
        
        # ТЕЛЕМЕТРИЯ НА ЭКРАНЕ
        if telemetry:
            y_pos = 40
            
            # Время
            minutes = int(telemetry['time'] // 60)
            seconds = int(telemetry['time'] % 60)
            ms = int((telemetry['time'] % 1) * 100)
            time_str = f"{minutes:02d}:{seconds:02d}.{ms:02d}"
            cv2.putText(annotated, f"Time: {time_str}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Позиция
            y_pos += 30
            cv2.putText(annotated, f"Pos: X={telemetry['position_x']:.1f} Y={telemetry['position_y']:.1f}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Скорость
            y_pos += 30
            cv2.putText(annotated, f"Speed: {telemetry['velocity']:.2f} px/f (avg: {telemetry['avg_velocity']:.2f})", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Ускорение
            y_pos += 30
            accel_str = f"{telemetry['acceleration']:+.2f}"
            cv2.putText(annotated, f"Accel: {accel_str} px/f2", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Детекций
            y_pos += 30
            cv2.putText(annotated, f"Detections: {self.total_detections}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated

def main():
    model_path = "runs/segment/train10/weights/best.pt"
    video_path = "sources/GX010138.MP4"
    output_dir = "results/track_2sec_telemetry"
    
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print("[!] Модель или видео не найдены")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_time = 6 * 60 + 35  # 395 сек
    end_time = 6 * 60 + 37    # 397 сек
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    print("[*] Инициализирую трекер...")
    tracker = BallTracker(model_path)
    
    print("[*] Создаю видео...")
    output_video = os.path.join(output_dir, "tracked_2sec_telemetry.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("[!] Ошибка VideoWriter")
        return
    
    print(f"[*] Обрабатываю 2 сек (06:35-06:37) с телеметрией...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    processed_count = 0
    start_processing = time.time()
    
    for frame_idx in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Трекинг
        detection = tracker.track_ball(frame)
        
        # Телеметрия
        telemetry = tracker.get_telemetry(frame_idx, fps, start_time)
        
        # Рисуем
        annotated = tracker.draw_tracking_with_telemetry(frame, detection, telemetry)
        
        # Записываем
        out.write(annotated)
        
        processed_count += 1
        
        if (processed_count) % 30 == 0:
            print(f"    Обработано {processed_count} кадров...")
    
    cap.release()
    out.release()
    
    print(f"\n[+] ВИДЕО: {output_video}")
    print(f"\n[=] СТАТИСТИКА:")
    print(f"    Кадров: {processed_count}")
    print(f"    Детекций мяча: {tracker.total_detections}")
    print(f"    Время: {time.time() - start_processing:.1f}сек")
    
    # Сохраняем телеметрию в JSON
    import json
    telemetry_log = []
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    tracker = BallTracker(model_path)
    
    for frame_idx in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        detection = tracker.track_ball(frame)
        telemetry = tracker.get_telemetry(frame_idx, fps, start_time)
        if telemetry:
            telemetry_log.append(telemetry)
    
    cap.release()
    
    json_path = os.path.join(output_dir, "telemetry.json")
    with open(json_path, 'w') as f:
        json.dump(telemetry_log, f, indent=2)
    
    print(f"    Телеметрия: {json_path}")

if __name__ == "__main__":
    main()

