"""
Детектор ударов мяча об стол.

Использует анализ вогнутости траектории (паттерн вниз-вверх)
для определения момента отскока от стола.

Все параметры настраиваются через конфиг - нет hardcoded значений.
"""

import cv2
import numpy as np
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ImpactEvent:
    """Событие удара об стол."""
    frame: int
    time: float
    x: float
    y: float
    curvature: float
    y_range: float
    incoming_speed: float
    outgoing_speed: float
    incoming_vector: Tuple[float, float] = (0.0, 0.0)
    outgoing_vector: Tuple[float, float] = (0.0, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame': self.frame,
            'time': self.time,
            'x': self.x,
            'y': self.y,
            'curvature': self.curvature,
            'y_range': self.y_range,
            'incoming_speed': self.incoming_speed,
            'outgoing_speed': self.outgoing_speed,
            'incoming_vector': list(self.incoming_vector),
            'outgoing_vector': list(self.outgoing_vector)
        }


@dataclass
class ImpactDetectorConfig:
    """
    Конфигурация детектора ударов.
    Все параметры имеют значения по умолчанию и документацию.
    """
    # Минимальное количество последовательных детекций для подтверждения
    min_consecutive_detections: int = 3
    
    # Минимальный разрыв между ударами (кадры)
    min_frame_gap: int = 6
    
    # Минимальная длительность ралли (секунды)
    min_rally_duration: float = 0.05
    
    # Минимальная скорость мяча (пиксели/кадр)
    min_velocity: float = 2.0
    
    # Максимальная скорость мяча (пиксели/кадр) - фильтр артефактов
    max_velocity: float = 100.0
    
    # Максимальное изменение направления (градусы)
    max_direction_change: float = 180.0
    
    # Порог кривизны траектории (пиксели отклонения)
    curvature_threshold: float = 10.0
    
    # Минимальный диапазон Y для детекции отскока (пиксели)
    y_range_threshold: float = 15.0
    
    # Длина истории траектории для анализа
    trajectory_history_len: int = 10
    
    # Минимум точек для анализа паттерна
    min_points_for_pattern: int = 5
    
    # Максимум пропущенных кадров до сброса истории
    max_missed_frames: int = 5
    
    # Fallback зона стола (нормализованная, 0-1) если полигон не обнаружен
    fallback_table_zone_top: float = 0.55    # Y начала зоны стола
    fallback_table_zone_bottom: float = 0.95  # Y конца зоны стола
    fallback_table_zone_left: float = 0.1     # X левой границы
    fallback_table_zone_right: float = 0.9    # X правой границы
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ImpactDetectorConfig':
        """Создаёт конфиг из словаря, игнорируя неизвестные ключи."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


class SmartImpactDetector:
    """
    Детектор ударов об стол с анализом вогнутости траектории.
    
    Принцип работы:
    1. Собирает историю позиций мяча
    2. Анализирует паттерн "вниз-вверх" (вогнутость)
    3. Проверяет что точка отскока на столе
    4. Фильтрует по минимальным порогам
    
    Все параметры настраиваются через ImpactDetectorConfig.
    """
    
    def __init__(self, frame_height: int, fps: float = 60.0, 
                 config: Optional[ImpactDetectorConfig] = None):
        """
        Args:
            frame_height: Высота кадра в пикселях
            fps: Частота кадров видео
            config: Конфигурация параметров (или используется дефолт)
        """
        self.frame_height = frame_height
        self.fps = fps
        self.config = config or ImpactDetectorConfig()
        
        # Полигон стола (устанавливается извне)
        self.table_polygon: Optional[np.ndarray] = None
        self.table_polygon_cv: Optional[np.ndarray] = None
        
        # Fallback зона стола (в пикселях)
        self._fallback_zone = self._calculate_fallback_zone()
        
        # История траектории
        self.trajectory_history: deque = deque(maxlen=self.config.trajectory_history_len)
        self.pixel_history: deque = deque(maxlen=self.config.trajectory_history_len)
        self.y_history: deque = deque(maxlen=self.config.trajectory_history_len)
        
        # Обнаруженные удары
        self.live_impacts: List[ImpactEvent] = []
        self.last_impact_frame: Optional[int] = None
        
        # Счётчики
        self.impact_count: int = 0
        self.consecutive_detections: int = 0
        self.missed_frames: int = 0
        
        # Состояние ралли
        self.rally_start: Optional[Tuple[int, float]] = None
        self.impacts: List[Dict] = []
        
        logger.debug(f"SmartImpactDetector initialized: {self.config}")
    
    def _calculate_fallback_zone(self) -> Dict[str, float]:
        """Вычисляет fallback зону стола в пикселях."""
        return {
            'top': self.frame_height * self.config.fallback_table_zone_top,
            'bottom': self.frame_height * self.config.fallback_table_zone_bottom,
            'left': 0,  # Будет установлено при известной ширине
            'right': 0
        }
    
    def set_frame_width(self, width: int) -> None:
        """Устанавливает ширину кадра для fallback зоны."""
        self._fallback_zone['left'] = width * self.config.fallback_table_zone_left
        self._fallback_zone['right'] = width * self.config.fallback_table_zone_right
    
    def set_table_polygon(self, polygon: np.ndarray) -> None:
        """
        Устанавливает полигон стола.
        
        Args:
            polygon: Массив точек [(x1,y1), (x2,y2), ...]
        """
        self.table_polygon = polygon
        self.table_polygon_cv = polygon.reshape(-1, 1, 2).astype(np.float32)
        logger.info(f"Table polygon set with {len(polygon)} points")
    
    def _is_point_on_table(self, x: float, y: float) -> bool:
        """
        Проверяет, находится ли точка на столе.
        Использует полигон если есть, иначе fallback зону.
        """
        if self.table_polygon_cv is not None:
            result = cv2.pointPolygonTest(self.table_polygon_cv, (x, y), False)
            return result >= 0
        
        # Fallback: прямоугольная зона
        return (self._fallback_zone['left'] <= x <= self._fallback_zone['right'] and
                self._fallback_zone['top'] <= y <= self._fallback_zone['bottom'])
    
    def _calculate_curvature(self, points: List[Tuple[float, float]]) -> float:
        """
        Вычисляет кривизну траектории (вогнутость).
        
        Измеряет отклонение средней точки от линии начало-конец.
        Положительное значение = вогнутость вверх (отскок).
        """
        if len(points) < 3:
            return 0.0
        
        p1 = points[0]
        p2 = points[len(points) // 2]
        p3 = points[-1]
        
        # Отклонение средней точки от линии p1-p3
        curvature = p2[1] - max(p1[1], p3[1])
        
        # Если средняя точка НИЖЕ обоих концов
        if p2[1] < min(p1[1], p3[1]):
            curvature = min(p1[1], p3[1]) - p2[1]
        
        return abs(curvature)
    
    def detect_impact(self, position: Tuple[float, float], frame_idx: int,
                      pixel_position: Optional[Tuple[float, float]] = None) -> Optional[Dict[str, Any]]:
        """
        Анализирует позицию мяча и определяет, был ли удар об стол.
        
        Args:
            position: Нормализованная или пиксельная позиция
            frame_idx: Номер кадра
            pixel_position: Пиксельная позиция (если position нормализована)
            
        Returns:
            Метаданные удара или None если удара не было
        """
        if position is None:
            return None
        
        # Проверяем наличие стола или fallback
        has_table = (self.table_polygon_cv is not None or 
                     self._fallback_zone['right'] > 0)
        if not has_table:
            return None
        
        # Добавляем в историю
        self.trajectory_history.append(position)
        self.pixel_history.append(pixel_position if pixel_position else position)
        self.y_history.append(position[1])
        
        # Проверяем минимум точек
        if len(self.trajectory_history) < self.config.min_points_for_pattern:
            return None
        
        points_list = list(self.trajectory_history)
        first_y = points_list[0][1]
        middle_y = points_list[len(points_list) // 2][1]
        last_y = points_list[-1][1]
        
        # ПРОВЕРКА 1: Паттерн вниз-вверх
        goes_down = (first_y < middle_y)  # Y увеличивается = вниз
        goes_up = (middle_y > last_y)     # Y уменьшается = вверх
        
        if not (goes_down and goes_up):
            return None
        
        # ПРОВЕРКА 2: Достаточное движение по Y
        y_range = max(self.y_history) - min(self.y_history)
        if y_range < self.config.y_range_threshold:
            return None
        
        # ПРОВЕРКА 3: Кривизна траектории
        curvature = self._calculate_curvature(points_list)
        if curvature < self.config.curvature_threshold:
            return None
        
        # ПРОВЕРКА 4: Точка отскока на столе
        mid_idx = len(points_list) // 2
        bounce_point = points_list[mid_idx]
        pixel_points = list(self.pixel_history)
        bounce_pixel = pixel_points[mid_idx] if len(pixel_points) > mid_idx else bounce_point
        
        if not self._is_point_on_table(bounce_pixel[0], bounce_pixel[1]):
            return None
        
        # Вычисляем векторы и скорости
        prev_point = points_list[max(0, mid_idx - 1)]
        next_point = points_list[min(len(points_list) - 1, mid_idx + 1)]
        
        incoming_vec = (bounce_point[0] - prev_point[0], bounce_point[1] - prev_point[1])
        outgoing_vec = (next_point[0] - bounce_point[0], next_point[1] - bounce_point[1])
        incoming_speed = math.sqrt(incoming_vec[0]**2 + incoming_vec[1]**2)
        outgoing_speed = math.sqrt(outgoing_vec[0]**2 + outgoing_vec[1]**2)
        
        # ПРОВЕРКА 5: Скорость в реалистичном диапазоне (фильтр артефактов)
        if incoming_speed > self.config.max_velocity or outgoing_speed > self.config.max_velocity:
            logger.debug(f"Rejected: speed too high ({incoming_speed:.1f}/{outgoing_speed:.1f} > {self.config.max_velocity})")
            return None
        
        logger.debug(f"Impact detected at frame {frame_idx}: curvature={curvature:.1f}, y_range={y_range:.1f}")
        
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
    
    def process_detection(self, position: Tuple[float, float], frame_idx: int,
                          fps_time: float, impact_meta: Dict[str, Any]) -> bool:
        """
        Обрабатывает подтверждённый удар.
        
        Args:
            position: Позиция мяча
            frame_idx: Номер кадра
            fps_time: Время в секундах
            impact_meta: Метаданные от detect_impact()
            
        Returns:
            True если удар добавлен, False если обновлён существующий
        """
        self.consecutive_detections += 1
        
        if self.consecutive_detections < self.config.min_consecutive_detections:
            return False
        
        if not self.rally_start:
            self.rally_start = (frame_idx, fps_time)
        
        # Создаём событие
        event = ImpactEvent(
            frame=frame_idx,
            time=fps_time,
            x=float(impact_meta['bounce_pixel'][0]),
            y=float(impact_meta['bounce_pixel'][1]),
            curvature=float(impact_meta['curvature']),
            y_range=float(impact_meta['y_range']),
            incoming_speed=float(impact_meta['incoming_speed']),
            outgoing_speed=float(impact_meta['outgoing_speed']),
            incoming_vector=tuple(float(v) for v in impact_meta['incoming_vector']),
            outgoing_vector=tuple(float(v) for v in impact_meta['outgoing_vector'])
        )
        
        # Проверяем минимальный разрыв между ударами
        should_add_new = (
            self.last_impact_frame is None or
            (frame_idx - self.last_impact_frame) >= self.config.min_frame_gap
        )
        
        if should_add_new:
            self.live_impacts.append(event)
            self.last_impact_frame = frame_idx
            logger.info(f"New impact #{len(self.live_impacts)} at frame {frame_idx}")
        elif self.live_impacts:
            # Обновляем последний удар
            self.live_impacts[-1] = event
        
        self.impact_count = len(self.live_impacts)
        self.consecutive_detections = 0
        
        return should_add_new
    
    def on_missed_frame(self, frame_idx: int, fps_time: float) -> None:
        """
        Вызывается когда мяч не обнаружен на кадре.
        
        Args:
            frame_idx: Номер кадра
            fps_time: Время в секундах
        """
        self.missed_frames += 1
        
        if self.missed_frames > self.config.max_missed_frames:
            self.finalize_rally(frame_idx, fps_time)
    
    def finalize_rally(self, frame_idx: int, fps_time: float) -> bool:
        """
        Завершает текущий ралли.
        
        Returns:
            True если ралли был валидным
        """
        if not self.rally_start:
            self.consecutive_detections = 0
            self._clear_history()
            return False
        
        start_frame, start_time = self.rally_start
        duration = fps_time - start_time
        
        if duration >= self.config.min_rally_duration:
            self.impacts.append({
                'frame': start_frame,
                'time': start_time,
                'duration': duration,
                'end_frame': frame_idx
            })
            self.rally_start = None
            self.consecutive_detections = 0
            logger.debug(f"Rally finalized: duration={duration:.3f}s")
            return True
        
        self.rally_start = None
        self.consecutive_detections = 0
        return False
    
    def _clear_history(self) -> None:
        """Очищает историю траектории."""
        self.trajectory_history.clear()
        self.pixel_history.clear()
        self.y_history.clear()
        self.missed_frames = 0
    
    def reset(self) -> None:
        """Полный сброс детектора."""
        self._clear_history()
        self.live_impacts.clear()
        self.impacts.clear()
        self.last_impact_frame = None
        self.impact_count = 0
        self.consecutive_detections = 0
        self.rally_start = None
        logger.debug("ImpactDetector reset")
    
    def get_impacts_json(self) -> List[Dict[str, Any]]:
        """Возвращает список ударов в формате JSON."""
        return [e.to_dict() for e in self.live_impacts]

