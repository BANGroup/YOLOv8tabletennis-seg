#!/usr/bin/env python3
"""
Трекинг мяча v4 - копия оригинального main_processor.py с улучшениями.
Базируется на проверенном скрипте с 11 ударами.

УЛУЧШЕНИЯ:
1. Более низкий confidence YOLO (0.03 вместо 0.1)
2. Расширенный диапазон ball_area (20-6000)  
3. Фильтр максимальной скорости (отсечение артефактов)
4. Сохранена оригинальная визуализация

НЕ ИЗМЕНЯТЬ ОРИГИНАЛ main_processor.py!
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import math
from collections import deque
import json
import argparse
from pathlib import Path


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


class SmartImpactDetectorV4:
    """Улучшенный детектор ударов v4"""
    
    def __init__(self, frame_height, fps=59.94):
        self.frame_height = frame_height
        self.fps = fps
        self.table_zone_start = int(frame_height * 0.65)
        
        # ПАРАМЕТРЫ (можно переопределить из конфига)
        self.max_direction_change = 180
        self.min_rally_duration = 0.05
        self.min_velocity = 2.0
        self.max_velocity = 80.0  # НОВОЕ: фильтр артефактов
        self.min_consecutive_detections = 3
        self.min_frame_gap = 8  # Увеличено с 6
        self.curvature_threshold = 10  # Порог кривизны
        self.y_range_threshold = 15  # Порог Y-диапазона
        
        self.table_polygon = None
        self.table_polygon_cv = None
        
        self.trajectory_history = deque(maxlen=10)
        self.pixel_history = deque(maxlen=10)
        self.y_history = deque(maxlen=10)
        self.live_impacts = []
        self.last_impact_frame = None
        
        self.impacts = []
        self.impact_count = 0
        self.rally_start = None
        self.consecutive_detections = 0
    
    def calculate_curvature(self, points):
        if len(points) < 3:
            return 0.0
        
        p1 = points[0]
        p2 = points[len(points) // 2]
        p3 = points[-1]
        
        curvature = p2[1] - max(p1[1], p3[1])
        
        if p2[1] < min(p1[1], p3[1]):
            curvature = min(p1[1], p3[1]) - p2[1]
        
        return abs(curvature)
    
    def detect_impact(self, position, frame_idx, pixel_position=None):
        """Обнаруживает удар по ВОГНУТОЙ траектории"""
        if not position:
            return False
        
        if self.table_polygon_cv is None:
            return False
        
        self.trajectory_history.append(position)
        self.pixel_history.append(pixel_position if pixel_position is not None else position)
        self.y_history.append(position[1])
        
        if len(self.trajectory_history) < 5:
            return False
        
        points_list = list(self.trajectory_history)
        first_y = points_list[0][1]
        middle_y = points_list[len(points_list)//2][1]
        last_y = points_list[-1][1]
        
        # ПРОВЕРКА 1: Паттерн вниз-вверх
        goes_down = (first_y < middle_y)
        goes_up = (middle_y > last_y)
        
        if not (goes_down and goes_up):
            return False
        
        # ПРОВЕРКА 2: Y-диапазон
        y_range = max(self.y_history) - min(self.y_history)
        if y_range < self.y_range_threshold:
            return False
        
        # ПРОВЕРКА 3: Кривизна
        curvature = self.calculate_curvature(points_list)
        if curvature < self.curvature_threshold:
            return False
        
        mid_idx = len(points_list) // 2
        bounce_point = points_list[mid_idx]
        pixel_points = list(self.pixel_history)
        bounce_pixel = pixel_points[mid_idx] if len(pixel_points) > mid_idx else bounce_point
        
        # ПРОВЕРКА 4: Точка на столе
        if cv2.pointPolygonTest(self.table_polygon_cv, (bounce_pixel[0], bounce_pixel[1]), False) < 0:
            return False
        
        prev_point = points_list[max(0, mid_idx - 1)]
        next_point = points_list[min(len(points_list) - 1, mid_idx + 1)]
        
        incoming_vec = (bounce_point[0] - prev_point[0], bounce_point[1] - prev_point[1])
        outgoing_vec = (next_point[0] - bounce_point[0], next_point[1] - bounce_point[1])
        incoming_speed = math.sqrt(incoming_vec[0]**2 + incoming_vec[1]**2)
        outgoing_speed = math.sqrt(outgoing_vec[0]**2 + outgoing_vec[1]**2)
        
        # ПРОВЕРКА 5: Скорость в реалистичном диапазоне
        if incoming_speed > self.max_velocity or outgoing_speed > self.max_velocity:
            return False
        
        return {
            'frame': frame_idx,
            'bounce_point': bounce_point,
            'bounce_pixel': bounce_pixel,
            'curvature': curvature,
            'y_range': y_range,
            'incoming_vector': incoming_vec,
            'outgoing_vector': outgoing_vec,
            'incoming_speed': incoming_speed,
            'outgoing_speed': outgoing_speed
        }
    
    def process_detection(self, position, frame_idx, fps_frame_time, impact_meta):
        """Обработка детекции"""
        self.consecutive_detections += 1
        
        if self.consecutive_detections < self.min_consecutive_detections:
            return False
        
        if not self.rally_start:
            self.rally_start = (frame_idx, fps_frame_time)
        
        live_event = {
            'frame': frame_idx,
            'time': fps_frame_time,
            'x': float(impact_meta['bounce_pixel'][0]),
            'y': float(impact_meta['bounce_pixel'][1]),
            'curvature': float(impact_meta['curvature']),
            'y_range': float(impact_meta['y_range']),
            'incoming_vector': [float(v) for v in impact_meta['incoming_vector']],
            'outgoing_vector': [float(v) for v in impact_meta['outgoing_vector']],
            'incoming_speed': float(impact_meta['incoming_speed']),
            'outgoing_speed': float(impact_meta['outgoing_speed'])
        }
        
        should_add_new = (
            self.last_impact_frame is None or
            (frame_idx - self.last_impact_frame) >= self.min_frame_gap
        )
        
        if should_add_new:
            self.live_impacts.append(live_event)
            self.last_impact_frame = frame_idx
        elif self.live_impacts:
            self.live_impacts[-1].update(live_event)
        
        self.impact_count = len(self.live_impacts)
        self.consecutive_detections = 0
        return should_add_new
    
    def finalize_rally(self, frame_idx, fps_frame_time):
        if not self.rally_start:
            self.consecutive_detections = 0
            self.trajectory_history.clear()
            self.y_history.clear()
            self.pixel_history.clear()
            return False
        
        start_frame, start_time = self.rally_start
        duration = fps_frame_time - start_time
        
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


class BallTrackerV4:
    def __init__(self, model_path, frame_width, frame_height, fps, output_dir, params=None):
        self.model = YOLO(model_path)
        self.kalman = KalmanFilter()
        self.impact_detector = SmartImpactDetectorV4(frame_height, fps)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.output_dir = output_dir
        
        # ПАРАМЕТРЫ YOLO (улучшенные)
        self.confidence = 0.03  # Было 0.1
        self.ball_min_area = 20  # Было 80
        self.ball_max_area = 6000  # Было 3000
        
        # Применяем параметры из конфига
        if params:
            # YOLO
            if 'confidence' in params:
                self.confidence = params['confidence']
            if 'ball_min_area' in params:
                self.ball_min_area = params['ball_min_area']
            if 'ball_max_area' in params:
                self.ball_max_area = params['ball_max_area']
            
            # Impact detector
            if 'min_rally_duration' in params:
                self.impact_detector.min_rally_duration = params['min_rally_duration']
            if 'min_velocity' in params:
                self.impact_detector.min_velocity = params['min_velocity']
            if 'max_velocity' in params:
                self.impact_detector.max_velocity = params['max_velocity']
            if 'min_consecutive_detections' in params:
                self.impact_detector.min_consecutive_detections = params['min_consecutive_detections']
            if 'min_frame_gap' in params:
                self.impact_detector.min_frame_gap = params['min_frame_gap']
            if 'curvature_threshold' in params:
                self.impact_detector.curvature_threshold = params['curvature_threshold']
            if 'y_range_threshold' in params:
                self.impact_detector.y_range_threshold = params['y_range_threshold']
        
        self.trajectory = deque(maxlen=100)
        self.velocities = deque(maxlen=15)
        self.total_detections = 0
        self.missed_frames = 0
        self.kalman_predictions_used = 0
        self.fps = fps
        self.table_polygon = None
        self.table_class_id = params.get('table_class_id', 4) if params else 4
    
    def detect_ball(self, frame):
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        result = results[0]
        self._maybe_update_table_polygon(result, frame)
        
        if result.boxes is not None and len(result.boxes) > 0:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            boxes = result.boxes.xyxy.cpu().numpy()
            
            best_ball = None
            best_area = 0
            
            for box, class_id in zip(boxes, class_ids):
                if class_id == 0:  # Ball
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Фильтр по размеру
                    if not (self.ball_min_area <= area <= self.ball_max_area):
                        continue
                    
                    if area > best_area:
                        best_area = area
                        best_ball = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            return best_ball
        return None
    
    def _maybe_update_table_polygon(self, result, frame):
        if self.table_polygon is not None:
            return
        if result.masks is None or result.boxes is None:
            return
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        if not hasattr(result.masks, 'xy') or result.masks.xy is None:
            return
            
        polygons = result.masks.xy
        candidates = []
        for idx, cls in enumerate(class_ids):
            if cls == self.table_class_id and idx < len(polygons):
                poly = np.array(polygons[idx], dtype=np.float32)
                if poly.shape[0] >= 4:
                    area = abs(cv2.contourArea(poly))
                    candidates.append((area, poly))
        if not candidates:
            return
        _, best_poly = max(candidates, key=lambda x: x[0])
        epsilon = 0.01 * cv2.arcLength(best_poly, True)
        approx = cv2.approxPolyDP(best_poly, epsilon, True)
        if approx.shape[0] < 4:
            approx = cv2.convexHull(best_poly)
        polygon = approx.reshape(-1, 2)
        self.table_polygon = polygon
        self.impact_detector.table_polygon = polygon
        self.impact_detector.table_polygon_cv = polygon.reshape(-1, 1, 2).astype(np.float32)
        self.impact_detector.table_zone_start = int(np.mean(polygon[:, 1]))
        
        # Сохраняем debug изображение
        debug_img = frame.copy()
        cv2.polylines(debug_img, [self.impact_detector.table_polygon_cv.astype(np.int32)], True, (0, 0, 255), 2)
        overlay = debug_img.copy()
        cv2.fillPoly(overlay, [self.impact_detector.table_polygon_cv.astype(np.int32)], (0, 0, 255))
        debug_img = cv2.addWeighted(overlay, 0.2, debug_img, 0.8, 0)
        if self.output_dir:
            debug_path = os.path.join(self.output_dir, "table_polygon_debug_v4.png")
            cv2.imwrite(debug_path, debug_img)
            print(f"[+] Table polygon → {debug_path}")
    
    def track_ball(self, frame, frame_idx, fps_time):
        detection = self.detect_ball(frame)
        
        if detection:
            self.total_detections += 1
            self.missed_frames = 0
            self.kalman.update(detection)
            self.trajectory.append(detection)
            
            if len(self.trajectory) > 1:
                last_pos = self.trajectory[-2]
                dx = detection[0] - last_pos[0]
                dy = detection[1] - last_pos[1]
                speed = math.sqrt(dx*dx + dy*dy)
                self.velocities.append(speed)
                
                impact_meta = self.impact_detector.detect_impact(detection, frame_idx, pixel_position=detection)
                if impact_meta:
                    self.impact_detector.process_detection(detection, frame_idx, fps_time, impact_meta)
        else:
            self.missed_frames += 1
            
            if self.missed_frames > 5:
                self.impact_detector.finalize_rally(frame_idx, fps_time)
            
            if self.kalman.initialized:
                pred = self.kalman.predict()
                if pred:
                    self.kalman_predictions_used += 1
        
        return detection
    
    def draw_tracking(self, frame, detection):
        """ОРИГИНАЛЬНАЯ визуализация из main_processor.py"""
        annotated = frame.copy()
        
        # Тонкие линии траектории
        if len(self.trajectory) > 1:
            points = [(int(p[0]), int(p[1])) for p in self.trajectory]
            
            for i in range(1, len(points)):
                alpha = i / len(points)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                cv2.line(annotated, points[i-1], points[i], color, 1)
            
            for i, point in enumerate(points):
                alpha = i / len(points)
                radius = int(1 + alpha * 1.5)
                cv2.circle(annotated, point, radius, (0, 255, 0), -1)
        
        if detection:
            cx, cy = map(int, detection)
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(annotated, (cx, cy), 7, (0, 255, 0), 1)
        
        # Стрелка скорости
        if len(self.velocities) > 0 and detection and len(self.trajectory) > 1:
            cx, cy = map(int, detection)
            speed = self.velocities[-1]
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
        
        # Полигон стола
        if self.impact_detector.table_polygon_cv is not None:
            poly = self.impact_detector.table_polygon_cv.astype(np.int32)
            cv2.polylines(annotated, [poly], True, (120, 120, 255), 2)
        
        # Статистика
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
        
        impacts_count = len(self.impact_detector.live_impacts)
        cv2.putText(annotated, f"IMPACTS: {impacts_count}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_pos += 30
        
        # Точки ударов
        if impacts_count > 0:
            for event in self.impact_detector.live_impacts[-8:]:
                hit_x = int(event['x'])
                hit_y = int(event['y'])
                cv2.circle(annotated, (hit_x, hit_y), 6, (0, 0, 255), 2)
            
            last = self.impact_detector.live_impacts[-1]
            cv2.putText(annotated,
                f"#{impacts_count}: in {last['incoming_speed']:.1f} out {last['outgoing_speed']:.1f}",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            y_pos += 25
            cv2.putText(annotated,
                f"Pos ({last['x']:.0f}, {last['y']:.0f}) Curv {last['curvature']:.1f}",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            y_pos += 30
        
        if len(self.velocities) > 0:
            avg_speed = np.mean(list(self.velocities)[-5:])
            cv2.putText(annotated, f"Speed: {avg_speed:.1f} px/frame", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
        
        if self.kalman_predictions_used > 0:
            cv2.putText(annotated, f"Kalman: {self.kalman_predictions_used}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated


def main():
    parser = argparse.ArgumentParser(description='Трекинг мяча v4 - улучшенная версия')
    parser.add_argument('--video', required=True, help='Путь к видео')
    parser.add_argument('--start', type=int, required=True, help='Начало (сек)')
    parser.add_argument('--end', type=int, required=True, help='Конец (сек)')
    parser.add_argument('--output', default='results/v4', help='Папка результатов')
    
    args = parser.parse_args()
    
    model_path = "runs/segment/train10/weights/best.pt"
    video_path = args.video
    start_time = args.start
    end_time = args.end
    output_dir = args.output
    
    if not os.path.exists(model_path):
        print(f"[!] Model not found: {model_path}")
        return
    if not os.path.exists(video_path):
        print(f"[!] Video not found: {video_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[*] Ball Tracker v4 (improved)")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"    Video: {frame_width}x{frame_height} @ {fps:.2f} FPS")
    print(f"    Segment: {start_time}-{end_time}s")
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Параметры v4
    params = {
        'confidence': 0.03,
        'ball_min_area': 20,
        'ball_max_area': 6000,
        'max_velocity': 80,
        'min_frame_gap': 8,
        'curvature_threshold': 10,
        'y_range_threshold': 15,
        'min_consecutive_detections': 3
    }
    
    tracker = BallTrackerV4(model_path, frame_width, frame_height, fps, output_dir, params)
    
    print(f"[*] Parameters: conf={params['confidence']}, area={params['ball_min_area']}-{params['ball_max_area']}, max_vel={params['max_velocity']}")
    
    output_video = os.path.join(output_dir, "tracked_v4.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed = 0
    total = end_frame - start_frame
    t_start = time.perf_counter()
    
    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        
        fps_time = start_time + (frame_idx / fps)
        detection = tracker.track_ball(frame, frame_idx, fps_time)
        
        # Время
        minutes = int(fps_time // 60)
        seconds = int(fps_time % 60)
        ms = int((fps_time % 1) * 100)
        time_str = f"{minutes:02d}:{seconds:02d}.{ms:02d}"
        
        annotated = tracker.draw_tracking(frame, detection)
        cv2.putText(annotated, f"Time: {time_str}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(annotated)
        processed += 1
        
        if processed % 60 == 0 or processed == 1:
            elapsed = time.perf_counter() - t_start
            eff_fps = processed / elapsed if elapsed > 0 else 0
            progress = (processed / total) * 100
            impacts = len(tracker.impact_detector.live_impacts)
            print(f"[{progress:5.1f}%] Frame {processed}/{total} | IMPACTS: {impacts} | FPS: {eff_fps:.1f}")
    
    # Финализация
    final_time = start_time + (processed / fps)
    tracker.impact_detector.finalize_rally(processed, final_time)
    
    cap.release()
    out.release()
    
    t_total = time.perf_counter() - t_start
    
    print(f"\n{'='*60}")
    print(f"RESULTS v4:")
    print(f"  Frames: {processed}")
    print(f"  Detections: {tracker.total_detections}")
    print(f"  IMPACTS: {len(tracker.impact_detector.live_impacts)}")
    print(f"  Kalman: {tracker.kalman_predictions_used}")
    print(f"  Time: {t_total:.2f}s | FPS: {processed/t_total:.1f}")
    print(f"  Output: {output_video}")
    print(f"{'='*60}")
    
    # Сохраняем JSON
    json_path = os.path.join(output_dir, "impacts_v4.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tracker.impact_detector.live_impacts, f, ensure_ascii=False, indent=2)
    print(f"  Impacts JSON: {json_path}")


if __name__ == "__main__":
    main()
