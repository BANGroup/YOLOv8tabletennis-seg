"""
Основной модуль трекинга мяча.

Объединяет:
- YOLO детекцию
- Фильтр Калмана
- Детектор ударов
- Извлечение полигона стола
"""

import cv2
import numpy as np
import math
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

from ultralytics import YOLO

from .kalman import KalmanFilterWithGravity
from .impact_detector import SmartImpactDetector, ImpactDetectorConfig

logger = logging.getLogger(__name__)


@dataclass
class BallTrackerConfig:
    """Конфигурация трекера мяча."""
    # YOLO
    model_path: str = "runs/segment/train10/weights/best.pt"
    confidence: float = 0.1
    ball_class_id: int = 0
    table_class_id: int = 4
    
    # Калман фильтр
    kalman_gravity: float = 0.5
    kalman_process_noise: float = 0.1
    kalman_measurement_noise: float = 10.0
    
    # Трекинг
    trajectory_max_len: int = 100
    velocity_history_len: int = 15
    missed_frames_to_finalize: int = 5
    
    # Размер мяча для фильтрации (площадь в пикселях)
    ball_min_area: float = 80
    ball_max_area: float = 3000
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BallTrackerConfig':
        """Создаёт конфиг из словаря."""
        # Объединяем параметры из разных секций
        merged = {}
        
        if 'yolo' in d:
            merged['model_path'] = d['yolo'].get('model_path', cls.model_path)
            merged['confidence'] = d['yolo'].get('confidence', cls.confidence)
            merged['ball_class_id'] = d['yolo'].get('ball_class_id', cls.ball_class_id)
            merged['table_class_id'] = d['yolo'].get('table_class_id', cls.table_class_id)
        
        if 'kalman' in d:
            merged['kalman_gravity'] = d['kalman'].get('gravity', cls.kalman_gravity)
            merged['kalman_process_noise'] = d['kalman'].get('process_noise', cls.kalman_process_noise)
            merged['kalman_measurement_noise'] = d['kalman'].get('measurement_noise', cls.kalman_measurement_noise)
        
        if 'tracking' in d:
            merged['trajectory_max_len'] = d['tracking'].get('trajectory_max_len', cls.trajectory_max_len)
            merged['velocity_history_len'] = d['tracking'].get('velocity_history_len', cls.velocity_history_len)
            merged['missed_frames_to_finalize'] = d['tracking'].get('missed_frames_to_finalize', cls.missed_frames_to_finalize)
            merged['ball_min_area'] = d['tracking'].get('ball_min_area', cls.ball_min_area)
            merged['ball_max_area'] = d['tracking'].get('ball_max_area', cls.ball_max_area)
        
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in merged.items() if k in valid_keys}
        return cls(**filtered)


class BallTracker:
    """
    Трекер мяча для настольного тенниса.
    
    Функции:
    1. Детекция мяча через YOLO
    2. Сглаживание траектории через Калман фильтр
    3. Обнаружение ударов об стол
    4. Извлечение и использование маски стола
    """
    
    def __init__(self, config: BallTrackerConfig,
                 impact_config: Optional[ImpactDetectorConfig] = None,
                 frame_width: int = 1920, frame_height: int = 1080,
                 fps: float = 60.0, output_dir: Optional[str] = None):
        """
        Args:
            config: Конфигурация трекера
            impact_config: Конфигурация детектора ударов
            frame_width: Ширина кадра
            frame_height: Высота кадра
            fps: Частота кадров
            output_dir: Папка для сохранения отладочных изображений
        """
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.output_dir = output_dir
        
        # Инициализация компонентов
        logger.info(f"Loading YOLO model: {config.model_path}")
        self.model = YOLO(config.model_path)
        
        self.kalman = KalmanFilterWithGravity(
            dt=1/fps,
            gravity_pixels=config.kalman_gravity,
            process_noise=config.kalman_process_noise,
            measurement_noise=config.kalman_measurement_noise
        )
        
        self.impact_detector = SmartImpactDetector(
            frame_height=frame_height,
            fps=fps,
            config=impact_config
        )
        self.impact_detector.set_frame_width(frame_width)
        
        # История
        self.trajectory: deque = deque(maxlen=config.trajectory_max_len)
        self.velocities: deque = deque(maxlen=config.velocity_history_len)
        
        # Счётчики
        self.total_detections: int = 0
        self.missed_frames: int = 0
        self.kalman_predictions_used: int = 0
        self.frame_count: int = 0
        
        # Полигон стола
        self.table_polygon: Optional[np.ndarray] = None
        self._table_extracted: bool = False
        
        logger.info(f"BallTracker initialized: {frame_width}x{frame_height} @ {fps}fps")
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Детектирует мяч на кадре через YOLO.
        
        Args:
            frame: BGR изображение
            
        Returns:
            (x, y) центр мяча или None
        """
        try:
            results = self.model.predict(frame, conf=self.config.confidence, verbose=False)
            result = results[0]
        except Exception as e:
            logger.warning(f"YOLO prediction failed: {e}")
            return None
        
        # Извлекаем полигон стола (один раз)
        if not self._table_extracted:
            try:
                self._extract_table_polygon(result, frame)
            except Exception as e:
                logger.warning(f"Table extraction failed: {e}")
                self._table_extracted = True  # Не пытаться снова
        
        if result.boxes is None or len(result.boxes) == 0:
            return None
        
        try:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            boxes = result.boxes.xyxy.cpu().numpy()
        except Exception as e:
            logger.warning(f"Failed to extract boxes: {e}")
            return None
        
        # Ищем мяч (class_id = 0)
        best_ball = None
        best_area = 0
        
        for box, class_id in zip(boxes, class_ids):
            if class_id != self.config.ball_class_id:
                continue
            
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            
            # Фильтрация по размеру
            if not (self.config.ball_min_area <= area <= self.config.ball_max_area):
                continue
            
            # Выбираем мяч с наибольшей площадью (ближе к камере)
            if area > best_area:
                best_area = area
                best_ball = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        return best_ball
    
    def _extract_table_polygon(self, result, frame: np.ndarray) -> None:
        """
        Извлекает полигон стола из результатов сегментации.
        
        Args:
            result: Результат YOLO
            frame: Кадр для отладочного изображения
        """
        if result.masks is None or result.boxes is None:
            logger.debug("No masks or boxes in result")
            return
        
        try:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
        except Exception as e:
            logger.warning(f"Failed to get class_ids for table: {e}")
            return
        
        # Проверяем есть ли маски
        if not hasattr(result.masks, 'xy') or result.masks.xy is None:
            logger.debug("No mask polygons available")
            return
        
        try:
            polygons = result.masks.xy
            candidates = []
            
            for idx, cls in enumerate(class_ids):
                if cls == self.config.table_class_id and idx < len(polygons):
                    poly = np.array(polygons[idx], dtype=np.float32)
                    if poly.shape[0] >= 4:
                        area = abs(cv2.contourArea(poly))
                        candidates.append((area, poly))
            
            if not candidates:
                logger.warning("No table polygon found, using fallback zone")
                self._table_extracted = True
                return
            
            # Выбираем наибольший полигон
            _, best_poly = max(candidates, key=lambda x: x[0])
            
            # Упрощаем полигон
            epsilon = 0.01 * cv2.arcLength(best_poly, True)
            approx = cv2.approxPolyDP(best_poly, epsilon, True)
            
            if approx.shape[0] < 4:
                approx = cv2.convexHull(best_poly)
            
            polygon = approx.reshape(-1, 2)
            self.table_polygon = polygon
            
            # Передаём в детектор ударов
            self.impact_detector.set_table_polygon(polygon)
            
            self._table_extracted = True
            logger.info(f"Table polygon extracted: {len(polygon)} points")
            
            # Сохраняем отладочное изображение
            if self.output_dir:
                self._save_table_debug_image(frame, polygon)
                
        except Exception as e:
            logger.warning(f"Error extracting table polygon: {e}")
            self._table_extracted = True  # Использовать fallback
    
    def _save_table_debug_image(self, frame: np.ndarray, polygon: np.ndarray) -> None:
        """Сохраняет отладочное изображение с полигоном стола."""
        import os
        
        debug_img = frame.copy()
        poly_cv = polygon.reshape(-1, 1, 2).astype(np.int32)
        
        # Контур
        cv2.polylines(debug_img, [poly_cv], True, (0, 0, 255), 2)
        
        # Полупрозрачная заливка
        overlay = debug_img.copy()
        cv2.fillPoly(overlay, [poly_cv], (0, 0, 255))
        debug_img = cv2.addWeighted(overlay, 0.2, debug_img, 0.8, 0)
        
        debug_path = os.path.join(self.output_dir, "table_polygon_debug.png")
        cv2.imwrite(debug_path, debug_img)
        logger.info(f"Table debug image saved: {debug_path}")
    
    def track(self, frame: np.ndarray, frame_idx: int, fps_time: float) -> Optional[Tuple[float, float]]:
        """
        Основной метод трекинга мяча на кадре.
        
        Args:
            frame: BGR изображение
            frame_idx: Номер кадра
            fps_time: Время в секундах от начала видео
            
        Returns:
            (x, y) позиция мяча или None
        """
        self.frame_count += 1
        detection = self.detect_ball(frame)
        
        if detection:
            self.total_detections += 1
            self.missed_frames = 0
            
            # Обновляем Калман
            self.kalman.update(detection)
            
            # Добавляем в траекторию
            self.trajectory.append(detection)
            
            # Вычисляем скорость
            if len(self.trajectory) > 1:
                last_pos = self.trajectory[-2]
                dx = detection[0] - last_pos[0]
                dy = detection[1] - last_pos[1]
                speed = math.sqrt(dx*dx + dy*dy)
                self.velocities.append(speed)
                
                # Проверяем удар
                impact_meta = self.impact_detector.detect_impact(
                    detection, frame_idx, pixel_position=detection
                )
                if impact_meta:
                    self.impact_detector.process_detection(
                        detection, frame_idx, fps_time, impact_meta
                    )
        else:
            self.missed_frames += 1
            
            # Уведомляем детектор ударов
            self.impact_detector.on_missed_frame(frame_idx, fps_time)
            
            # Используем предсказание Калмана
            if self.kalman.initialized:
                pred = self.kalman.predict()
                if pred:
                    self.kalman_predictions_used += 1
                    # Можно добавить предсказание в траекторию с пометкой
                
                # Проверяем не пора ли сбросить Калман
                self.kalman.mark_missed()
        
        return detection
    
    def get_current_velocity(self) -> Optional[Tuple[float, float]]:
        """Возвращает текущую скорость из Калмана."""
        return self.kalman.get_velocity()
    
    def get_average_speed(self, last_n: int = 5) -> float:
        """Возвращает среднюю скорость за последние N кадров."""
        if not self.velocities:
            return 0.0
        recent = list(self.velocities)[-last_n:]
        return float(np.mean(recent))
    
    def get_trajectory_points(self) -> List[Tuple[int, int]]:
        """Возвращает траекторию как список целочисленных точек."""
        return [(int(p[0]), int(p[1])) for p in self.trajectory]
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику трекинга."""
        return {
            'total_detections': self.total_detections,
            'missed_frames': self.missed_frames,
            'kalman_predictions': self.kalman_predictions_used,
            'trajectory_length': len(self.trajectory),
            'impact_count': self.impact_detector.impact_count,
            'frame_count': self.frame_count
        }
    
    def finalize(self, frame_idx: int, fps_time: float) -> None:
        """
        Завершает трекинг (вызывать в конце обработки видео).
        
        Args:
            frame_idx: Последний номер кадра
            fps_time: Время последнего кадра
        """
        self.impact_detector.finalize_rally(frame_idx, fps_time)
        logger.info(f"Tracking finalized: {self.get_stats()}")
    
    def reset(self) -> None:
        """Полный сброс трекера."""
        self.trajectory.clear()
        self.velocities.clear()
        self.kalman.reset()
        self.impact_detector.reset()
        self.total_detections = 0
        self.missed_frames = 0
        self.kalman_predictions_used = 0
        self.frame_count = 0
        self._table_extracted = False
        self.table_polygon = None
        logger.debug("BallTracker reset")

