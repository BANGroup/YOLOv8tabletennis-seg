#!/usr/bin/env python3
"""
Трекинг мяча с УЛУЧШЕННЫМ обнаружением ударов об стол (v2)
КОПИЯ main_processor.py с оптимизированными параметрами для достижения 14 ударов

Изменения относительно v1:
- min_consecutive_detections: 3 → 2 (ловить быстрые удары)
- min_frame_gap: 6 → 4 (не объединять близкие удары)
- curvature_threshold: 10 → 5 (ловить слабые отскоки)
- y_range_threshold: 15 → 10 (ловить плоские траектории)
- trajectory_history: 10 → 8 (быстрая реакция)
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

class SmartImpactDetectorV2:
    """Улучшенный детектор ударов об стол v2 - оптимизация для 14 ударов"""
    
    def __init__(self, frame_height, fps=59.94, params=None):
        self.frame_height = frame_height
        self.fps = fps
        self.table_zone_start = int(frame_height * 0.65)
        
        # ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ v2
        params = params or {}
        self.max_direction_change = params.get('max_direction_change', 180)
        self.min_rally_duration = params.get('min_rally_duration', 0.03)  # 30ms
        self.min_velocity = params.get('min_velocity', 1.5)  # Снижено с 2.0
        self.min_consecutive_detections = params.get('min_consecutive_detections', 2)  # Снижено с 3
        self.min_frame_gap = params.get('min_frame_gap', 4)  # Снижено с 6
        self.curvature_threshold = params.get('curvature_threshold', 5)  # Снижено с 10
        self.y_range_threshold = params.get('y_range_threshold', 10)  # Снижено с 15
        trajectory_len = params.get('trajectory_history_len', 8)  # Снижено с 10
        
        self.table_polygon = None
        self.table_polygon_cv = None
        
        # ИСТОРИЯ ТРАЕКТОРИИ
        self.trajectory_history = deque(maxlen=trajectory_len)
        self.pixel_history = deque(maxlen=trajectory_len)
        self.y_history = deque(maxlen=trajectory_len)
        self.live_impacts = []
        self.last_impact_frame = None
        
        self.impacts = []
        self.impact_count = 0
        self.rally_start = None
        self.consecutive_detections = 0
    
    def calculate_curvature(self, points):
        """Вычисляет кривизну траектории (вогнутость)"""
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
        """Обнаруживает удар об стол по ВОГНУТОЙ траектории (v2 - более чувствительный)"""
        if not position:
            return False
        
        if self.table_polygon_cv is None:
            return False
        
        self.trajectory_history.append(position)
        self.pixel_history.append(pixel_position if pixel_position is not None else position)
        self.y_history.append(position[1])
        
        # v2: Снижено минимум точек с 5 до 4
        if len(self.trajectory_history) < 4:
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
        
        # ПРОВЕРКА 2: Движение по Y (v2: снижено с 15 до y_range_threshold)
        y_min = min(self.y_history)
        y_max = max(self.y_history)
        y_range = y_max - y_min
        
        if y_range < self.y_range_threshold:
            return False
        
        # ПРОВЕРКА 3: Вогнутость (v2: снижено с 10 до curvature_threshold)
        curvature = self.calculate_curvature(points_list)
        
        if curvature < self.curvature_threshold:
            return False
        
        mid_idx = len(points_list) // 2
        bounce_point = points_list[mid_idx]
        pixel_points = list(self.pixel_history)
        bounce_pixel = pixel_points[mid_idx] if len(pixel_points) > mid_idx else bounce_point
        
        # Проверка на столе
        if cv2.pointPolygonTest(self.table_polygon_cv, (bounce_pixel[0], bounce_pixel[1]), False) < 0:
            return False
        
        prev_point = points_list[max(0, mid_idx - 1)]
        next_point = points_list[min(len(points_list) - 1, mid_idx + 1)]

        incoming_vec = (bounce_point[0] - prev_point[0], bounce_point[1] - prev_point[1])
        outgoing_vec = (next_point[0] - bounce_point[0], next_point[1] - bounce_point[1])
        incoming_speed = math.sqrt(incoming_vec[0] ** 2 + incoming_vec[1] ** 2)
        outgoing_speed = math.sqrt(outgoing_vec[0] ** 2 + outgoing_vec[1] ** 2)

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
        """Обработка детекции (v2: снижен min_consecutive_detections)"""
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
        
        # v2: Снижен min_frame_gap с 6 до 4
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
        """Завершить ралли"""
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

class BallTrackerV2:
    def __init__(self, model_path, frame_width, frame_height, fps, output_dir, impact_params=None):
        self.model = YOLO(model_path)
        self.kalman = KalmanFilter()
        self.impact_detector = SmartImpactDetectorV2(frame_height, fps, impact_params)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.output_dir = output_dir
        
        self.trajectory = deque(maxlen=100)
        self.velocities = deque(maxlen=15)
        self.total_detections = 0
        self.missed_frames = 0
        self.kalman_predictions_used = 0
        self.fps = fps
        self.table_polygon = None
        self.table_class_id = impact_params.get('table_class_id', 4) if impact_params else 4
        self.conf_threshold = impact_params.get('confidence', 0.1) if impact_params else 0.1
        
    def detect_ball(self, frame):
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        result = results[0]
        self._maybe_update_table_polygon(result, frame)
        
        if result.boxes is not None and len(result.boxes) > 0:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            boxes = result.boxes.xyxy.cpu().numpy()
            
            for box, class_id in zip(boxes, class_ids):
                if class_id == 0:  # Ball
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    return center
        return None

    def _maybe_update_table_polygon(self, result, frame):
        if self.table_polygon is not None:
            return
        if result.masks is None or result.boxes is None:
            return
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
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
        debug_img = frame.copy()
        cv2.polylines(debug_img, [self.impact_detector.table_polygon_cv.astype(np.int32)], True, (0, 0, 255), 2)
        overlay = debug_img.copy()
        cv2.fillPoly(overlay, [self.impact_detector.table_polygon_cv.astype(np.int32)], (0, 0, 255))
        debug_img = cv2.addWeighted(overlay, 0.2, debug_img, 0.8, 0)
        if self.output_dir:
            debug_path = os.path.join(self.output_dir, "table_polygon_debug.png")
            cv2.imwrite(debug_path, debug_img)
            print(f"[+] Table polygon captured → {debug_path}")
    
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
        annotated = frame.copy()
        
        if len(self.trajectory) > 1:
            points = [(int(p[0]), int(p[1])) for p in self.trajectory]
            
            for i in range(1, len(points)):
                alpha = i / len(points)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                cv2.line(annotated, points[i-1], points[i], color, 1)
            
            for i, point in enumerate(points):
                is_impact_point = any(imp['frame'] == i for imp in self.impact_detector.impacts)
                
                if is_impact_point:
                    cv2.circle(annotated, point, 4, (0, 0, 255), -1)
                    cv2.circle(annotated, point, 6, (0, 0, 255), 1)
                else:
                    alpha = i / len(points)
                    radius = int(1 + alpha * 1.5)
                    cv2.circle(annotated, point, radius, (0, 255, 0), -1)
        
        if detection:
            cx, cy = map(int, detection)
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(annotated, (cx, cy), 7, (0, 255, 0), 1)
        
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
        
        table_zone = self.impact_detector.table_zone_start
        cv2.line(annotated, (0, table_zone), (annotated.shape[1], table_zone), (200, 200, 200), 1)
        if self.impact_detector.table_polygon_cv is not None:
            poly = self.impact_detector.table_polygon_cv.astype(np.int32)
            cv2.polylines(annotated, [poly], True, (120, 120, 255), 2)
        
        # СТАТИСТИКА v2
        y_pos = 30
        cv2.putText(annotated, f"[v2] Detections: {self.total_detections}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(annotated, f"Missed: {self.missed_frames}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(annotated, f"Trajectory: {len(self.trajectory)}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        impacts_logged = len(self.impact_detector.live_impacts)
        cv2.putText(annotated, f"[RED] IMPACTS v2: {impacts_logged}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_pos += 30

        last_event = None
        if impacts_logged > 0:
            last_event = self.impact_detector.live_impacts[-1]
            for event in self.impact_detector.live_impacts[-8:]:
                hit_x = int(event['x'])
                hit_y = int(event['y'])
                cv2.circle(annotated, (hit_x, hit_y), 6, (0, 0, 255), 2)

        if last_event:
            cv2.putText(
                annotated,
                f"Impact#{impacts_logged}: in {last_event['incoming_speed']:.1f}px out {last_event['outgoing_speed']:.1f}px",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )
            y_pos += 25
            cv2.putText(
                annotated,
                f"Pos ({last_event['x']:.0f}, {last_event['y']:.0f}) Curv {last_event['curvature']:.1f}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )
        
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

def load_config(config_path):
    """Загружает конфиг из JSON файла"""
    if not os.path.exists(config_path):
        print(f"[!] Ошибка: конфиг не найден: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Трекинг мяча v2 - оптимизация для 14 ударов')
    parser.add_argument('--video', required=True, help='Путь к видео файлу')
    parser.add_argument('--start', type=int, required=True, help='Начало сегмента в секундах')
    parser.add_argument('--end', type=int, required=True, help='Конец сегмента в секундах')
    parser.add_argument('--config', default='scripts/config/detection_params.json', 
                       help='Путь к конфиг файлу')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    if not config:
        return
    
    video_path = args.video
    start_time = args.start
    end_time = args.end
    
    model_path = config['yolo']['model_path']
    output_dir = config['output']['output_dir']
    
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print("[!] Ошибка: модель или видео не найдены")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Инициализирую SMART трекер v2 (оптимизация для 14 ударов)...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"    Разрешение: {frame_width}x{frame_height}, FPS: {fps:.2f}")
    print(f"[*] Сегмент: {start_time}-{end_time} сек")
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    impact_params = config.get('impact_detection', {})
    impact_params['table_class_id'] = config.get('yolo', {}).get('table_class_id', 4)
    impact_params['confidence'] = config.get('yolo', {}).get('confidence', 0.1)
    
    tracker = BallTrackerV2(model_path, frame_width, frame_height, fps, output_dir, impact_params)
    
    print("[*] Создаю видео v2...")
    output_video = os.path.join(output_dir, "tracked_v2.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed_count = 0
    total_frames = max(1, end_frame - start_frame)
    processing_start = time.perf_counter()
    frame_durations = []
    
    for frame_idx in range(end_frame - start_frame):
        frame_loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        
        fps_time = start_time + (frame_idx / fps)
        detection = tracker.track_ball(frame, frame_idx, fps_time)
        
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        ms = int(((frame_idx / fps) % 1) * 100)
        time_str = f"{minutes:02d}:{seconds:02d}.{ms:02d}"
        
        annotated = tracker.draw_tracking(frame, detection)
        cv2.putText(annotated, f"Time: {time_str}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(annotated)
        
        processed_count += 1
        frame_duration = time.perf_counter() - frame_loop_start
        frame_durations.append(frame_duration)
        
        if processed_count % 30 == 0 or processed_count == 1:
            elapsed = time.perf_counter() - processing_start
            effective_fps = processed_count / elapsed if elapsed > 0 else 0
            progress_pct = (processed_count / total_frames) * 100
            print(f"[PROGRESS] {processed_count}/{total_frames} "
                  f"({progress_pct:.1f}%) | IMPACTS v2: {tracker.impact_detector.impact_count} | "
                  f"FPS: {effective_fps:.2f}")
    
    cap.release()
    out.release()
    processing_total = time.perf_counter() - processing_start
    
    final_frame_in_video = start_frame + processed_count
    final_fps_time = (final_frame_in_video / tracker.fps) if tracker.fps > 0 else 0
    tracker.impact_detector.finalize_rally(final_frame_in_video, final_fps_time)
    
    print(f"\n[+] ВИДЕО v2: {output_video}")
    print(f"\n[=] РЕЗУЛЬТАТЫ v2 (оптимизация для 14 ударов):")
    print(f"    Кадров: {processed_count}")
    print(f"    Детекций мяча: {tracker.total_detections}")
    print(f"    Пропущено: {tracker.missed_frames}")
    print(f"    SMART УДАРОВ v2: {tracker.impact_detector.impact_count}")
    print(f"    Kalman предсказаний: {tracker.kalman_predictions_used}")
    
    print(f"\n[*] ПАРАМЕТРЫ v2:")
    print(f"    min_consecutive_detections: {tracker.impact_detector.min_consecutive_detections}")
    print(f"    min_frame_gap: {tracker.impact_detector.min_frame_gap}")
    print(f"    curvature_threshold: {tracker.impact_detector.curvature_threshold}")
    print(f"    y_range_threshold: {tracker.impact_detector.y_range_threshold}")
    
    avg_frame_time = (processing_total / processed_count) if processed_count > 0 else 0
    min_frame_time = min(frame_durations) if frame_durations else 0
    max_frame_time = max(frame_durations) if frame_durations else 0
    effective_fps_total = processed_count / processing_total if processing_total > 0 else 0
    
    print(f"\n[TIMING] Total: {processing_total:.2f}s | "
          f"Avg/frame: {avg_frame_time*1000:.2f} ms | "
          f"Effective FPS: {effective_fps_total:.2f}")
    
    analysis = {
        'version': 'v2',
        'total_detections': tracker.total_detections,
        'missed_frames': tracker.missed_frames,
        'smart_impacts': tracker.impact_detector.impact_count,
        'impacts_details': tracker.impact_detector.impacts,
        'parameters_v2': {
            'min_consecutive_detections': tracker.impact_detector.min_consecutive_detections,
            'min_frame_gap': tracker.impact_detector.min_frame_gap,
            'curvature_threshold': tracker.impact_detector.curvature_threshold,
            'y_range_threshold': tracker.impact_detector.y_range_threshold,
            'min_velocity': tracker.impact_detector.min_velocity,
            'min_rally_duration': tracker.impact_detector.min_rally_duration
        },
        'timing': {
            'total_seconds': processing_total,
            'avg_ms_per_frame': avg_frame_time * 1000,
            'effective_fps': effective_fps_total,
            'processed_frames': processed_count
        },
        'segment': {
            'video': video_path,
            'start': start_time,
            'end': end_time
        }
    }
    
    json_path = os.path.join(output_dir, "impacts_v2.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    live_impacts_path = os.path.join(output_dir, "live_impacts_v2.json")
    with open(live_impacts_path, 'w', encoding='utf-8') as f:
        json.dump(tracker.impact_detector.live_impacts, f, ensure_ascii=False, indent=2)
    
    print(f"    Анализ: {json_path}")
    print(f"    Live impacts: {live_impacts_path}")

if __name__ == "__main__":
    main()







