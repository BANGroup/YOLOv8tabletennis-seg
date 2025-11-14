#!/usr/bin/env python3
"""
Трекинг мяча с УМНЫМ обнаружением ударов об стол
Использует логику из smart_tracking анализа:
- max_direction_change = 180° (обратные удары)
- min_rally_duration = 0.3 сек (быстрые пассы)
- min_velocity = 2.0 пикс/кадр (фильтр шума)
- min_consecutive_detections = 3 (фильтр YOLO шума)
- Требуется вертикальная смена направления (отскок)
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

class SmartImpactDetector:
    """Умный детектор ударов об стол с параметрами из smart_tracking анализа"""
    
    def __init__(self, frame_height, fps=59.94):
        self.frame_height = frame_height
        self.fps = fps
        self.table_zone_start = int(frame_height * 0.65)  # Нижние 35% = стол
        
        # ПАРАМЕТРЫ ИЗ SMART_TRACKING АНАЛИЗА
        self.max_direction_change = 180  # Градусы (разрешить обратные удары)
        self.min_rally_duration = 0.3    # Секунды (быстрые пассы)
        self.min_velocity = 2.0          # Пиксели/кадр (фильтр шума)
        self.min_consecutive_detections = 3  # Детекции подряд
        
        self.impacts = []
        self.impact_count = 0
        self.rally_start = None
        self.consecutive_detections = 0
    
    def detect_impact(self, pos1, pos2, pos3, pos4, frame_idx):
        """
        Обнаруживает удар об стол используя 4 последовательные точки
        Критерии:
        1. Точка 2 или 3 в зоне стола (низ кадра)
        2. Вертикальная смена направления (отскок вверх)
        3. Угол перелома >= 20° (НЕ 180°, как в старой логике)
        4. Минимум движение (не шум)
        """
        if not all([pos1, pos2, pos3, pos4]):
            return False
        
        # Проверяем, что есть точка в зоне стола
        in_table_zone = (pos2[1] > self.table_zone_start or pos3[1] > self.table_zone_start)
        if not in_table_zone:
            return False
        
        # Вектор 1: pos2 -> pos3 (приход к столу)
        v1 = (pos3[0] - pos2[0], pos3[1] - pos2[1])
        # Вектор 2: pos3 -> pos4 (отскок от стола)
        v2 = (pos4[0] - pos3[0], pos4[1] - pos3[1])
        
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        # Фильтруем очень малые движения (шум)
        if len1 < self.min_velocity or len2 < self.min_velocity:
            return False
        
        # Нормализуем
        v1_norm = (v1[0] / len1, v1[1] / len1)
        v2_norm = (v2[0] / len2, v2[1] / len2)
        
        # КЛЮЧЕВАЯ ПРОВЕРКА: вертикальная смена направления
        # Приход вниз (v1[1] > 0.2) + отскок вверх (v2[1] < -0.2) = удар об стол!
        is_bounce = (v1[1] > 0.15 and v2[1] < -0.15)
        
        if not is_bounce:
            return False
        
        # Скалярное произведение для угла
        dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)
        
        # Требуем хотя бы 20° перелома (исключаем очень малые отскоки)
        if angle_deg < 20:
            return False
        
        return True
    
    def process_detection(self, position, frame_idx, fps_frame_time):
        """Обработка детекции с фильтром ложных срабатываний"""
        # Подсчитываем последовательные детекции
        self.consecutive_detections += 1
        
        # Если это только первая или вторая детекция - еще не начинаем
        if self.consecutive_detections < self.min_consecutive_detections:
            return False
        
        # Если это ровно минимум детекций - начинаем ралли
        if self.consecutive_detections == self.consecutive_detections:
            self.rally_start = (frame_idx, fps_frame_time)
            return True
        
        return False
    
    def finalize_rally(self, frame_idx, fps_frame_time):
        """Завершить ралли и проверить минимальную длительность"""
        if not self.rally_start:
            self.consecutive_detections = 0
            return False
        
        start_frame, start_time = self.rally_start
        duration = fps_frame_time - start_time
        
        # Проверяем минимальную длительность (0.3 сек)
        if duration >= self.min_rally_duration:
            self.impact_count += 1
            self.impacts.append({
                'frame': start_frame,
                'time': start_time,
                'duration': duration,
                'end_frame': frame_idx
            })
            self.rally_start = None
            self.consecutive_detections = 0
            return True
        
        self.rally_start = None
        self.consecutive_detections = 0
        return False

class BallTracker:
    def __init__(self, model_path, frame_height, fps):
        self.model = YOLO(model_path)
        self.kalman = KalmanFilter()
        self.impact_detector = SmartImpactDetector(frame_height, fps)
        
        self.trajectory = deque(maxlen=100)
        self.velocities = deque(maxlen=15)
        self.total_detections = 0
        self.missed_frames = 0
        self.kalman_predictions_used = 0
        self.fps = fps
        
    def detect_ball(self, frame):
        results = self.model.predict(frame, conf=0.1, verbose=False)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            for box, class_id in zip(boxes, class_ids):
                if class_id == 0:  # Ball
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    return center
        return None
    
    def track_ball(self, frame, frame_idx, fps_time):
        detection = self.detect_ball(frame)
        
        if detection:
            self.total_detections += 1
            self.missed_frames = 0
            self.kalman.update(detection)
            self.trajectory.append(detection)
            
            # Вычисляем скорость
            if len(self.trajectory) > 1:
                last_pos = self.trajectory[-2]
                dx = detection[0] - last_pos[0]
                dy = detection[1] - last_pos[1]
                speed = math.sqrt(dx*dx + dy*dy)
                self.velocities.append(speed)
                
                # Анализируем удар об стол (нужна история из 4+ точек)
                if len(self.trajectory) >= 4:
                    pos1 = self.trajectory[-4]
                    pos2 = self.trajectory[-3]
                    pos3 = self.trajectory[-2]
                    pos4 = self.trajectory[-1]
                    
                    if self.impact_detector.detect_impact(pos1, pos2, pos3, pos4, frame_idx):
                        # Это потенциальный удар
                        self.impact_detector.process_detection(detection, frame_idx, fps_time)
        else:
            self.missed_frames += 1
            
            # Если пропустили > 5 кадров - завершаем ралли
            if self.missed_frames > 5:
                fps_time = (frame_idx / self.fps) if self.fps > 0 else 0
                self.impact_detector.finalize_rally(frame_idx, fps_time)
            
            if self.kalman.initialized:
                pred = self.kalman.predict()
                if pred:
                    self.kalman_predictions_used += 1
        
        return detection
    
    def draw_tracking(self, frame, detection):
        annotated = frame.copy()
        
        # Тонкие линии траектории
        if len(self.trajectory) > 1:
            points = [(int(p[0]), int(p[1])) for p in self.trajectory]
            
            for i in range(1, len(points)):
                alpha = i / len(points)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                cv2.line(annotated, points[i-1], points[i], color, 1)
            
            # Маленькие точки
            for i, point in enumerate(points):
                is_impact_point = any(imp['frame'] == i for imp in self.impact_detector.impacts)
                
                if is_impact_point:
                    # КРАСНАЯ точка - удар об стол
                    cv2.circle(annotated, point, 4, (0, 0, 255), -1)
                    cv2.circle(annotated, point, 6, (0, 0, 255), 1)
                else:
                    alpha = i / len(points)
                    radius = int(1 + alpha * 1.5)
                    cv2.circle(annotated, point, radius, (0, 255, 0), -1)
        
        # Текущая позиция
        if detection:
            cx, cy = map(int, detection)
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(annotated, (cx, cy), 7, (0, 255, 0), 1)
        
        # Стрелочка скорости
        if len(self.velocities) > 0 and detection:
            cx, cy = map(int, detection)
            speed = self.velocities[-1]
            if len(self.trajectory) > 1:
                last_pos = self.trajectory[-2]
                dx = cx - last_pos[0]
                dy = cy - last_pos[1]
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx_norm = dx / length
                    dy_norm = dy / length
                    end_x = int(cx + dx_norm * min(speed, 60))
                    end_y = int(cy + dy_norm * min(speed, 60))
                    cv2.arrowedLine(annotated, (cx, cy), (end_x, end_y), (0, 0, 255), 2, tipLength=0.25)
        
        # Зона стола
        table_zone = self.impact_detector.table_zone_start
        cv2.line(annotated, (0, table_zone), (annotated.shape[1], table_zone), (200, 200, 200), 1)
        
        # СТАТИСТИКА
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
        cv2.putText(annotated, f"[RED] IMPACTS (SMART): {self.impact_detector.impact_count}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
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
    output_dir = "results/track_2sec_smart_impacts"
    
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print("[!] Ошибка: модель или видео не найдены")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Инициализирую SMART трекер (smart_tracking логика)...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"    Разрешение: {frame_width}x{frame_height}, FPS: {fps:.2f}")
    
    # 2 сек (06:35 - 06:37)
    start_time = 6 * 60 + 35
    end_time = 6 * 60 + 37
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    tracker = BallTracker(model_path, frame_height, fps)
    
    print("[*] Создаю видео со SMART обнаружением ударов...")
    output_video = os.path.join(output_dir, "tracked_2sec_smart_impacts.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed_count = 0
    
    for frame_idx in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        fps_time = start_time + (frame_idx / fps)
        detection = tracker.track_ball(frame, frame_idx, fps_time)
        
        # Добавляем время
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        ms = int(((frame_idx / fps) % 1) * 100)
        time_str = f"{minutes:02d}:{seconds:02d}.{ms:02d}"
        
        annotated = tracker.draw_tracking(frame, detection)
        cv2.putText(annotated, f"Time: {time_str}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(annotated)
        
        processed_count += 1
        
        if (processed_count) % 30 == 0:
            print(f"    Обработано {processed_count} кадров, SMART ударов: {tracker.impact_detector.impact_count}")
    
    cap.release()
    out.release()
    
    print(f"\n[+] ВИДЕО: {output_video}")
    print(f"\n[=] РЕЗУЛЬТАТЫ (SMART ЛОГИКА из smart_tracking):")
    print(f"    Кадров: {processed_count}")
    print(f"    Детекций мяча: {tracker.total_detections}")
    print(f"    Пропущено: {tracker.missed_frames}")
    print(f"    SMART УДАРОВ об СТОЛ: {tracker.impact_detector.impact_count}")
    print(f"    Kalman предсказаний: {tracker.kalman_predictions_used}")
    
    print(f"\n[*] ИСПОЛЬЗУЕМЫЕ ПАРАМЕТРЫ из smart_tracking анализа:")
    print(f"    max_direction_change: {tracker.impact_detector.max_direction_change}°")
    print(f"    min_rally_duration: {tracker.impact_detector.min_rally_duration} сек")
    print(f"    min_velocity: {tracker.impact_detector.min_velocity} пикс/кадр")
    print(f"    min_consecutive_detections: {tracker.impact_detector.min_consecutive_detections}")
    
    # Сохраняем анализ
    analysis = {
        'total_detections': tracker.total_detections,
        'missed_frames': tracker.missed_frames,
        'smart_impacts': tracker.impact_detector.impact_count,
        'impacts_details': tracker.impact_detector.impacts,
        'parameters': {
            'max_direction_change': tracker.impact_detector.max_direction_change,
            'min_rally_duration': tracker.impact_detector.min_rally_duration,
            'min_velocity': tracker.impact_detector.min_velocity,
            'min_consecutive_detections': tracker.impact_detector.min_consecutive_detections
        }
    }
    
    json_path = os.path.join(output_dir, "smart_impacts_analysis.json")
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"    Анализ: {json_path}")

if __name__ == "__main__":
    main()

