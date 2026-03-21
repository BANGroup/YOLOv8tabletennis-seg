"""
Чистый визуализатор - копия оригинального из main_processor.py
Минимум шума, тонкие линии, маленькие точки.
"""

import cv2
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any

from .ball_tracker import BallTracker


class CleanVisualizer:
    """
    Чистый визуализатор как в оригинальном main_processor.py
    Минимум графики, максимум информативности.
    """
    
    def __init__(self):
        pass
    
    def draw(self, frame: np.ndarray, tracker: BallTracker,
             detection: Optional[Tuple[float, float]], 
             time_str: Optional[str] = None) -> np.ndarray:
        """
        Рисует визуализацию на кадре.
        
        Args:
            frame: Исходный BGR кадр
            tracker: Трекер мяча
            detection: Текущая детекция (x, y) или None
            time_str: Строка времени для отображения
            
        Returns:
            Кадр с визуализацией
        """
        annotated = frame.copy()
        
        # === ТРАЕКТОРИЯ ===
        if len(tracker.trajectory) > 1:
            points = [(int(p[0]), int(p[1])) for p in tracker.trajectory]
            
            # Тонкие линии траектории с градиентом
            for i in range(1, len(points)):
                alpha = i / len(points)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                cv2.line(annotated, points[i-1], points[i], color, 1)  # Толщина 1!
            
            # Маленькие точки
            for i, point in enumerate(points):
                alpha = i / len(points)
                radius = int(1 + alpha * 1.5)  # Радиус 1-2.5
                cv2.circle(annotated, point, radius, (0, 255, 0), -1)
        
        # === ТЕКУЩАЯ ПОЗИЦИЯ ===
        if detection:
            cx, cy = int(detection[0]), int(detection[1])
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(annotated, (cx, cy), 7, (0, 255, 0), 1)
        
        # === СТРЕЛКА СКОРОСТИ ===
        if len(tracker.velocities) > 0 and detection and len(tracker.trajectory) > 1:
            cx, cy = int(detection[0]), int(detection[1])
            speed = tracker.velocities[-1]
            last_pos = tracker.trajectory[-2]
            dx = cx - last_pos[0]
            dy = cy - last_pos[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx_norm = dx / length
                dy_norm = dy / length
                end_x = int(cx + dx_norm * min(speed, 60))
                end_y = int(cy + dy_norm * min(speed, 60))
                cv2.arrowedLine(annotated, (cx, cy), (end_x, end_y), (0, 0, 255), 2, tipLength=0.25)
        
        # === ПОЛИГОН СТОЛА ===
        if tracker.impact_detector.table_polygon_cv is not None:
            poly = tracker.impact_detector.table_polygon_cv.astype(np.int32)
            cv2.polylines(annotated, [poly], True, (120, 120, 255), 2)
        
        # === СТАТИСТИКА ===
        y_pos = 30
        cv2.putText(annotated, f"Detections: {tracker.total_detections}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(annotated, f"Missed: {tracker.missed_frames}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(annotated, f"Trajectory: {len(tracker.trajectory)}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        
        # Удары
        impacts_count = len(tracker.impact_detector.live_impacts)
        cv2.putText(annotated, f"IMPACTS: {impacts_count}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_pos += 30
        
        # Рисуем точки ударов
        if impacts_count > 0:
            for event in tracker.impact_detector.live_impacts[-8:]:
                hit_x = int(event.x)
                hit_y = int(event.y)
                cv2.circle(annotated, (hit_x, hit_y), 6, (0, 0, 255), 2)
            
            # Информация о последнем ударе
            last = tracker.impact_detector.live_impacts[-1]
            cv2.putText(annotated,
                f"Impact#{impacts_count}: in {last.incoming_speed:.1f}px out {last.outgoing_speed:.1f}px",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            y_pos += 25
            cv2.putText(annotated,
                f"Pos ({last.x:.0f}, {last.y:.0f}) Curv {last.curvature:.1f}",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            y_pos += 30
        
        # Скорость
        if len(tracker.velocities) > 0:
            avg_speed = np.mean(list(tracker.velocities)[-5:])
            cv2.putText(annotated, f"Speed: {avg_speed:.1f} px/frame", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
        
        # Калман
        if tracker.kalman_predictions_used > 0:
            cv2.putText(annotated, f"Kalman: {tracker.kalman_predictions_used}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # === ВРЕМЯ ===
        if time_str:
            cv2.putText(annotated, f"Time: {time_str}", 
                       (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated
