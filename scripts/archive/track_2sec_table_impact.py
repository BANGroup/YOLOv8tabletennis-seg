#!/usr/bin/env python3
"""
Трекинг мяча с обнаружением УДАРА об СТОЛ
- Тонкие линии траектории
- Маленькие точки
- Считает только РЕАЛЬНЫЕ изломы (удар об стол)
- Логика: резкое изменение направления В НИЖНЕЙ ЧАСТИ КАДРА = удар об стол
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

class TableImpactDetector:
    """Детектор удара об стол"""
    
    def __init__(self, frame_height, angle_threshold=25, speed_threshold=5):
        """
        frame_height: высота кадра (для определения низа - стола)
        angle_threshold: минимальный угол для обнаружения отскока (в градусах)
        speed_threshold: минимальная разница скоростей
        """
        self.frame_height = frame_height
        self.table_zone_start = int(frame_height * 0.65)  # Нижние 35% кадра = зона стола
        self.angle_threshold = angle_threshold
        self.speed_threshold = speed_threshold
        
        self.impacts = []  # Точки ударов об стол
        self.impact_count = 0
    
    def detect_impact(self, pos1, pos2, pos3, frame_idx):
        """
        Обнаруживает удар об стол:
        1. Точка 2 должна быть в зоне стола (внизу кадра)
        2. Резкое изменение направления (перелом)
        3. Вертикальная компонента меняет направление (отскок вверх)
        """
        if not pos1 or not pos2 or not pos3:
            return False
        
        # Проверяем, что pos2 в зоне стола (низ кадра)
        if pos2[1] < self.table_zone_start:
            return False  # Не в зоне стола
        
        # Вектор 1: pos1 -> pos2 (движение вниз к столу)
        v1 = (pos2[0] - pos1[0], pos2[1] - pos1[1])
        # Вектор 2: pos2 -> pos3 (отскок от стола)
        v2 = (pos3[0] - pos2[0], pos3[1] - pos2[1])
        
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 < 0.5 or len2 < 0.5:
            return False
        
        # Нормализуем
        v1_norm = (v1[0] / len1, v1[1] / len1)
        v2_norm = (v2[0] / len2, v2[1] / len2)
        
        # Проверяем вертикальное изменение направления
        # Если v1 идет вниз (v1[1] > 0) а v2 идет вверх (v2[1] < 0) - это отскок!
        is_bounce = (v1[1] > 0.1 and v2[1] < -0.1)  # Более чувствительный отскок
        
        # Скалярное произведение
        dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Угол между векторами
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)
        
        # Разница скоростей
        speed_diff = abs(len2 - len1)
        
        # УДАР об стол - только при большом угле перелома
        # Это исключает ложные срабатывания от небольших колебаний
        is_impact = angle_deg > self.angle_threshold and speed_diff > self.speed_threshold
        
        if is_impact:
            self.impact_count += 1
            self.impacts.append({
                'frame': frame_idx,
                'position': pos2,
                'angle': angle_deg,
                'speed_before': len1,
                'speed_after': len2,
                'speed_diff': speed_diff
            })
            return True
        
        return False

class BallTracker:
    def __init__(self, model_path, frame_height):
        self.model = YOLO(model_path)
        self.kalman = KalmanFilter()
        self.impact_detector = TableImpactDetector(frame_height, angle_threshold=45, speed_threshold=3)
        
        self.trajectory = deque(maxlen=100)
        self.velocities = deque(maxlen=15)
        self.total_detections = 0
        self.missed_frames = 0
        self.kalman_predictions_used = 0
        
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
    
    def track_ball(self, frame):
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
                
                # Анализируем удар об стол (если есть 3+ точки)
                if len(self.trajectory) >= 3:
                    pos1 = self.trajectory[-3]
                    pos2 = self.trajectory[-2]
                    pos3 = self.trajectory[-1]
                    
                    self.impact_detector.detect_impact(pos1, pos2, pos3, self.total_detections)
        else:
            self.missed_frames += 1
            
            if self.kalman.initialized:
                pred = self.kalman.predict()
                if pred:
                    self.kalman_predictions_used += 1
        
        return detection
    
    def draw_tracking(self, frame, detection):
        annotated = frame.copy()
        
        # ТОНКИЕ ЛИНИИ траектории
        if len(self.trajectory) > 1:
            points = [(int(p[0]), int(p[1])) for p in self.trajectory]
            
            # Рисуем тонкие линии между точками
            for i in range(1, len(points)):
                alpha = i / len(points)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                cv2.line(annotated, points[i-1], points[i], color, 1)  # Толщина 1
            
            # Маленькие точки
            for i, point in enumerate(points):
                is_impact_point = any(imp['frame'] == i for imp in self.impact_detector.impacts)
                
                if is_impact_point:
                    # КРАСНАЯ точка - удар об стол
                    cv2.circle(annotated, point, 4, (0, 0, 255), -1)
                    cv2.circle(annotated, point, 6, (0, 0, 255), 1)
                else:
                    # Маленькая зеленая точка - нормальная траектория
                    alpha = i / len(points)
                    radius = int(1 + alpha * 1.5)
                    cv2.circle(annotated, point, radius, (0, 255, 0), -1)
        
        # Текущая позиция мяча
        if detection:
            cx, cy = map(int, detection)
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(annotated, (cx, cy), 7, (0, 255, 0), 1)
        
        # Стрелочка скорости (тонкая)
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
        
        # Зона стола (полупрозрачная)
        table_zone = self.impact_detector.table_zone_start
        cv2.line(annotated, (0, table_zone), (annotated.shape[1], table_zone), (200, 200, 200), 1)
        cv2.putText(annotated, "TABLE ZONE", (10, table_zone - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # СТАТИСТИКА В ЛЕВОМ ВЕРХНЕМ УГЛУ
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
        cv2.putText(annotated, f"[RED] TABLE IMPACTS: {self.impact_detector.impact_count}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
    output_dir = "results/track_2sec_table_impacts"
    
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print("[!] Ошибка: модель или видео не найдены")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Инициализирую трекер с обнаружением ударов об стол...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"    Разрешение: {frame_width}x{frame_height}")
    print(f"    FPS: {fps:.2f}")
    
    # 2 сек (06:35 - 06:37)
    start_time = 6 * 60 + 35
    end_time = 6 * 60 + 37
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    tracker = BallTracker(model_path, frame_height)
    
    print("[*] Создаю видео с обнаружением ударов об стол...")
    output_video = os.path.join(output_dir, "tracked_2sec_table_impacts.mp4")
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
        
        # Трекинг
        detection = tracker.track_ball(frame)
        
        # Добавляем время
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        ms = int(((frame_idx / fps) % 1) * 100)
        time_str = f"{minutes:02d}:{seconds:02d}.{ms:02d}"
        
        # Рисуем трекинг с обнаружением ударов
        annotated = tracker.draw_tracking(frame, detection)
        
        # Добавляем время
        cv2.putText(annotated, f"Time: {time_str}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Записываем
        out.write(annotated)
        
        processed_count += 1
        
        if (processed_count) % 30 == 0:
            print(f"    Обработано {processed_count} кадров, ударов об стол: {tracker.impact_detector.impact_count}")
    
    cap.release()
    out.release()
    
    print(f"\n[+] ВИДЕО: {output_video}")
    print(f"\n[=] РЕЗУЛЬТАТЫ:")
    print(f"    Кадров: {processed_count}")
    print(f"    Детекций мяча: {tracker.total_detections}")
    print(f"    Пропущено: {tracker.missed_frames}")
    print(f"    УДАРОВ об СТОЛ: {tracker.impact_detector.impact_count}")
    print(f"    Kalman предсказаний: {tracker.kalman_predictions_used}")
    
    # Сохраняем анализ
    analysis = {
        'total_detections': tracker.total_detections,
        'missed_frames': tracker.missed_frames,
        'table_impacts': tracker.impact_detector.impact_count,
        'kalman_predictions': tracker.kalman_predictions_used,
        'impacts_details': tracker.impact_detector.impacts
    }
    
    json_path = os.path.join(output_dir, "impacts_analysis.json")
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"    Анализ: {json_path}")
    print(f"\n[*] ВИЗУАЛИЗАЦИЯ:")
    print(f"    Тонкие линии = траектория")
    print(f"    Маленькие точки = позиции мяча")
    print(f"    [RED] Красные точки = УДАРЫ об стол")
    print(f"    Серая линия = зона стола")

if __name__ == "__main__":
    main()

