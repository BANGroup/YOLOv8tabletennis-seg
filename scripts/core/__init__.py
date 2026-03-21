"""
Core модули для трекинга мяча в настольном теннисе.

Модули:
- kalman: Улучшенный фильтр Калмана с моделью гравитации
- impact_detector: Детектор ударов об стол
- ball_tracker: Основной трекер мяча
- visualizer: Визуализация результатов
"""

from .kalman import KalmanFilterWithGravity
from .impact_detector import SmartImpactDetector
from .ball_tracker import BallTracker
from .visualizer import TrackingVisualizer
from .visualizer_clean import CleanVisualizer

__all__ = [
    'KalmanFilterWithGravity',
    'SmartImpactDetector', 
    'BallTracker',
    'TrackingVisualizer',
    'CleanVisualizer'
]

__version__ = '3.0.1'

