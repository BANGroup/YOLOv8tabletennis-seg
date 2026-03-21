"""
Модуль визуализации трекинга.

Отделён от логики трекинга для соблюдения принципа единственной ответственности.
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

from .ball_tracker import BallTracker
from .impact_detector import ImpactEvent


@dataclass 
class VisualizerConfig:
    """Конфигурация визуализации."""
    # Траектория
    trajectory_line_thickness: int = 1
    trajectory_point_min_radius: int = 1
    trajectory_point_max_radius: int = 3
    
    # Текущая позиция мяча
    ball_radius: int = 5
    ball_outline_radius: int = 7
    ball_color: Tuple[int, int, int] = (0, 255, 0)  # BGR: зелёный
    
    # Удары
    impact_radius: int = 6
    impact_color: Tuple[int, int, int] = (0, 0, 255)  # BGR: красный
    impact_history_count: int = 8  # Сколько последних ударов показывать
    
    # Стрелка скорости
    velocity_arrow_color: Tuple[int, int, int] = (0, 0, 255)
    velocity_arrow_thickness: int = 2
    velocity_arrow_max_length: int = 60
    
    # Полигон стола
    table_outline_color: Tuple[int, int, int] = (120, 120, 255)
    table_outline_thickness: int = 2
    table_zone_line_color: Tuple[int, int, int] = (200, 200, 200)
    
    # Текст статистики
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.7
    font_thickness: int = 2
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_highlight_color: Tuple[int, int, int] = (0, 0, 255)
    text_info_color: Tuple[int, int, int] = (0, 255, 255)
    text_start_y: int = 30
    text_line_height: int = 30
    
    # Время
    time_font_scale: float = 0.8
    time_position_offset: int = 20  # От нижнего края
    
    # Включение/выключение элементов
    show_trajectory: bool = True
    show_impacts: bool = True
    show_velocity_arrow: bool = True
    show_table_polygon: bool = True
    show_stats: bool = True
    show_time: bool = True


class TrackingVisualizer:
    """
    Визуализатор результатов трекинга.
    
    Рисует:
    - Траекторию мяча (градиент цвета)
    - Текущую позицию мяча
    - Точки ударов об стол
    - Стрелку скорости
    - Полигон стола
    - Статистику
    - Время
    """
    
    def __init__(self, config: Optional[VisualizerConfig] = None):
        """
        Args:
            config: Конфигурация визуализации
        """
        self.config = config or VisualizerConfig()
    
    def draw(self, frame: np.ndarray, tracker: BallTracker,
             detection: Optional[Tuple[float, float]], 
             time_str: Optional[str] = None,
             extra_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Рисует все элементы визуализации на кадре.
        
        Args:
            frame: Исходный BGR кадр
            tracker: Трекер мяча
            detection: Текущая детекция (x, y) или None
            time_str: Строка времени для отображения
            extra_info: Дополнительная информация для HUD
            
        Returns:
            Кадр с визуализацией
        """
        annotated = frame.copy()
        
        try:
            if self.config.show_table_polygon:
                self._draw_table(annotated, tracker)
            
            if self.config.show_trajectory:
                self._draw_trajectory(annotated, tracker)
            
            if self.config.show_impacts:
                self._draw_impacts(annotated, tracker)
            
            if detection and self.config.show_velocity_arrow:
                self._draw_velocity_arrow(annotated, detection, tracker)
            
            if detection:
                self._draw_current_position(annotated, detection)
            
            if self.config.show_stats:
                self._draw_stats(annotated, tracker, extra_info)
            
            if self.config.show_time and time_str:
                self._draw_time(annotated, time_str)
        except Exception as e:
            # Если визуализация упала - вернуть кадр без аннотаций
            cv2.putText(annotated, f"Viz error: {str(e)[:50]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return annotated
    
    def _draw_trajectory(self, frame: np.ndarray, tracker: BallTracker) -> None:
        """Рисует траекторию мяча с градиентом цвета."""
        points = tracker.get_trajectory_points()
        if len(points) < 2:
            return
        
        impacts = tracker.impact_detector.live_impacts
        impact_frames = {e.frame for e in impacts}
        
        for i in range(1, len(points)):
            alpha = i / len(points)
            color_intensity = int(255 * alpha)
            # Градиент от синего к красному
            color = (color_intensity, 0, 255 - color_intensity)
            cv2.line(frame, points[i-1], points[i], color, 
                    self.config.trajectory_line_thickness)
        
        # Точки траектории
        for i, point in enumerate(points):
            # Проверяем не точка ли удара
            is_impact = i in impact_frames
            
            if is_impact:
                cv2.circle(frame, point, 4, self.config.impact_color, -1)
                cv2.circle(frame, point, 6, self.config.impact_color, 1)
            else:
                alpha = i / len(points)
                radius = int(self.config.trajectory_point_min_radius + 
                           alpha * (self.config.trajectory_point_max_radius - 
                                   self.config.trajectory_point_min_radius))
                cv2.circle(frame, point, radius, self.config.ball_color, -1)
    
    def _draw_current_position(self, frame: np.ndarray, 
                                detection: Tuple[float, float]) -> None:
        """Рисует текущую позицию мяча."""
        cx, cy = int(detection[0]), int(detection[1])
        cv2.circle(frame, (cx, cy), self.config.ball_radius, 
                  self.config.ball_color, -1)
        cv2.circle(frame, (cx, cy), self.config.ball_outline_radius, 
                  self.config.ball_color, 1)
    
    def _draw_impacts(self, frame: np.ndarray, tracker: BallTracker) -> None:
        """Рисует точки ударов об стол."""
        impacts = tracker.impact_detector.live_impacts
        if not impacts:
            return
        
        # Показываем последние N ударов
        recent = impacts[-self.config.impact_history_count:]
        
        for event in recent:
            x, y = int(event.x), int(event.y)
            cv2.circle(frame, (x, y), self.config.impact_radius, 
                      self.config.impact_color, 2)
    
    def _draw_velocity_arrow(self, frame: np.ndarray,
                              detection: Tuple[float, float],
                              tracker: BallTracker) -> None:
        """Рисует стрелку скорости."""
        if len(tracker.velocities) == 0 or len(tracker.trajectory) < 2:
            return
        
        cx, cy = int(detection[0]), int(detection[1])
        speed = tracker.velocities[-1]
        
        last_pos = tracker.trajectory[-2]
        dx = cx - last_pos[0]
        dy = cy - last_pos[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            dx_norm = dx / length
            dy_norm = dy / length
            arrow_len = min(speed, self.config.velocity_arrow_max_length)
            end_x = int(cx + dx_norm * arrow_len)
            end_y = int(cy + dy_norm * arrow_len)
            
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y),
                           self.config.velocity_arrow_color,
                           self.config.velocity_arrow_thickness,
                           tipLength=0.25)
    
    def _draw_table(self, frame: np.ndarray, tracker: BallTracker) -> None:
        """Рисует полигон стола и зону."""
        # Горизонтальная линия зоны стола
        if hasattr(tracker.impact_detector, '_fallback_zone'):
            zone_y = int(tracker.impact_detector._fallback_zone.get('top', 0))
            if zone_y > 0:
                cv2.line(frame, (0, zone_y), (frame.shape[1], zone_y),
                        self.config.table_zone_line_color, 1)
        
        # Полигон стола
        if tracker.impact_detector.table_polygon_cv is not None:
            poly = tracker.impact_detector.table_polygon_cv.astype(np.int32)
            cv2.polylines(frame, [poly], True,
                         self.config.table_outline_color,
                         self.config.table_outline_thickness)
    
    def _draw_stats(self, frame: np.ndarray, tracker: BallTracker,
                    extra_info: Optional[Dict[str, Any]] = None) -> None:
        """Рисует статистику на кадре."""
        y_pos = self.config.text_start_y
        
        # Детекции
        cv2.putText(frame, f"Det: {tracker.total_detections}",
                   (10, y_pos), self.config.font, self.config.font_scale,
                   self.config.text_color, self.config.font_thickness)
        y_pos += self.config.text_line_height
        
        # Пропущено
        cv2.putText(frame, f"Miss: {tracker.missed_frames}",
                   (10, y_pos), self.config.font, self.config.font_scale,
                   self.config.text_color, self.config.font_thickness)
        y_pos += self.config.text_line_height
        
        # Траектория
        cv2.putText(frame, f"Traj: {len(tracker.trajectory)}",
                   (10, y_pos), self.config.font, self.config.font_scale,
                   self.config.text_color, self.config.font_thickness)
        y_pos += self.config.text_line_height
        
        # УДАРЫ (выделено красным)
        impact_count = len(tracker.impact_detector.live_impacts)
        cv2.putText(frame, f"IMPACTS: {impact_count}",
                   (10, y_pos), self.config.font, 0.8,
                   self.config.text_highlight_color, self.config.font_thickness)
        y_pos += self.config.text_line_height
        
        # Информация о последнем ударе
        if impact_count > 0:
            last = tracker.impact_detector.live_impacts[-1]
            cv2.putText(frame,
                       f"#{impact_count}: in {last.incoming_speed:.1f} out {last.outgoing_speed:.1f}",
                       (10, y_pos), self.config.font, 0.6,
                       self.config.text_info_color, self.config.font_thickness)
            y_pos += 25
            cv2.putText(frame,
                       f"Pos ({last.x:.0f}, {last.y:.0f}) Curv {last.curvature:.1f}",
                       (10, y_pos), self.config.font, 0.6,
                       self.config.text_info_color, self.config.font_thickness)
            y_pos += self.config.text_line_height
        
        # Скорость
        avg_speed = tracker.get_average_speed()
        if avg_speed > 0:
            cv2.putText(frame, f"Speed: {avg_speed:.1f} px/f",
                       (10, y_pos), self.config.font, self.config.font_scale,
                       self.config.text_color, self.config.font_thickness)
            y_pos += self.config.text_line_height
        
        # Калман
        if tracker.kalman_predictions_used > 0:
            cv2.putText(frame, f"Kalman: {tracker.kalman_predictions_used}",
                       (10, y_pos), self.config.font, self.config.font_scale,
                       self.config.text_color, self.config.font_thickness)
        
        # Дополнительная информация
        if extra_info:
            y_pos += self.config.text_line_height
            for key, value in extra_info.items():
                cv2.putText(frame, f"{key}: {value}",
                           (10, y_pos), self.config.font, 0.6,
                           self.config.text_color, self.config.font_thickness)
                y_pos += 25
    
    def _draw_time(self, frame: np.ndarray, time_str: str) -> None:
        """Рисует время на кадре."""
        y_pos = frame.shape[0] - self.config.time_position_offset
        cv2.putText(frame, f"Time: {time_str}",
                   (10, y_pos), self.config.font, self.config.time_font_scale,
                   self.config.text_color, self.config.font_thickness)

