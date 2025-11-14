#!/usr/bin/env python3
"""
Трекинг мяча для 2 секунд (06:35-06:37) с масками сегментации
Потоковая обработка + OpenCV запись
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import math
from collections import deque

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

class BallTracker:
    """Трекер мяча с Kalman фильтром"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.kalman = KalmanFilter()
        self.trajectory = deque(maxlen=100)
        self.velocities = deque(maxlen=15)
        self.total_detections = 0
        
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
        else:
            if self.kalman.initialized:
                self.kalman.predict()
        
        return detection
    
    def draw_tracking(self, frame, detection):
        annotated = frame.copy()
        
        # Траектория
        if len(self.trajectory) > 1:
            points = [(int(p[0]), int(p[1])) for p in self.trajectory]
            for i in range(1, len(points)):
                cv2.line(annotated, points[i-1], points[i], (255, 0, 255), 2)
        
        # Текущая позиция мяча
        if detection:
            cx, cy = map(int, detection['center'])
            cv2.circle(annotated, (cx, cy), 8, (0, 255, 0), -1)
            cv2.circle(annotated, (cx, cy), 11, (0, 255, 0), 2)
        
        # Стрелочка предсказания
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
        
        return annotated
    
    def draw_masks(self, frame, detection_data):
        """Рисует маски сегментации"""
        annotated = frame.copy()
        
        class_colors = {
            0: (0, 255, 0),      # Ball
            1: (255, 255, 0),    # Net
            2: (0, 255, 255),    # Player1
            3: (255, 0, 255),    # Player2
            4: (255, 0, 0)       # Table
        }
        
        if detection_data['masks'] is not None:
            for mask_xy, class_id in zip(detection_data['masks'], detection_data['class_ids']):
                color = class_colors.get(class_id, (255, 255, 255))
                mask_points = np.array(mask_xy, dtype=np.int32)
                cv2.polylines(annotated, [mask_points], True, color, 2)
                cv2.fillPoly(annotated, [mask_points], color)
        
        alpha = 0.3
        annotated = cv2.addWeighted(annotated, alpha, frame, 1 - alpha, 0)
        return annotated

def add_time_to_frame(frame, elapsed_seconds):
    """Добавить время на видео"""
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
    model_path = "runs/segment/train10/weights/best.pt"
    video_path = "sources/GX010138.MP4"
    output_dir = "results/track_2sec_635_637"
    
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
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 2 секунды (06:35 - 06:37)
    start_time = 6 * 60 + 35  # 395 сек
    end_time = 6 * 60 + 37    # 397 сек
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    print(f"[*] Инициализирую трекер...")
    tracker = BallTracker(model_path)
    
    print(f"[*] Создаю видео для вывода...")
    output_video = os.path.join(output_dir, "tracked_2sec.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("[!] Ошибка при открытии VideoWriter")
        return
    
    print(f"\n[*] Обрабатываю видео (2 сек: 06:35-06:37)...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    processed_count = 0
    start_processing = time.time()
    
    for frame_idx in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Трекинг мяча
        detection = tracker.track_ball(frame)
        
        # Запускаем сегментацию для масок
        results = tracker.model.predict(frame, conf=0.1, verbose=False)
        
        # Рисуем маски
        annotated = frame.copy()
        if results[0].masks is not None:
            masks = results[0].masks.xy
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            class_colors = {
                0: (0, 255, 0),
                1: (255, 255, 0),
                2: (0, 255, 255),
                3: (255, 0, 255),
                4: (255, 0, 0)
            }
            
            for mask_xy, class_id in zip(masks, class_ids):
                color = class_colors.get(class_id, (255, 255, 255))
                mask_points = np.array(mask_xy, dtype=np.int32)
                
                # Проверяем валидность маски
                if len(mask_points) > 2:
                    try:
                        cv2.polylines(annotated, [mask_points], True, color, 2)
                        cv2.fillPoly(annotated, [mask_points], color)
                    except:
                        pass  # Пропускаем невалидные маски
            
            alpha = 0.3
            annotated = cv2.addWeighted(annotated, alpha, frame, 1 - alpha, 0)
        
        # Рисуем трекинг
        annotated = tracker.draw_tracking(annotated, detection)
        
        # Добавляем время
        elapsed = (frame_idx / fps)
        annotated = add_time_to_frame(annotated, 6 * 60 + 35 + elapsed)
        
        # Записываем в видео
        out.write(annotated)
        
        processed_count += 1
        
        if (processed_count) % 30 == 0:
            elapsed_time = time.time() - start_processing
            print(f"    Обработано {processed_count} кадров ({elapsed_time:.1f}сек)")
    
    cap.release()
    out.release()
    
    print(f"\n[+] ВИДЕО СОЗДАНО: {output_video}")
    print(f"\n[=] СТАТИСТИКА:")
    print(f"    Кадров: {processed_count}")
    print(f"    Детекций мяча: {tracker.total_detections}")
    print(f"    Время обработки: {time.time() - start_processing:.1f}сек")

if __name__ == "__main__":
    main()

