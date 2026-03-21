"""
Улучшенный фильтр Калмана с моделью гравитации.

Состояние: [x, y, vx, vy, ax, ay]
- x, y: позиция
- vx, vy: скорость
- ax, ay: ускорение (ay включает гравитацию)

Учитывает физику полёта мяча для настольного тенниса.
"""

import numpy as np
from typing import Optional, Tuple


class KalmanFilterWithGravity:
    """
    Фильтр Калмана с моделью ускорения для трекинга мяча.
    
    Улучшения относительно базовой версии:
    1. Состояние включает ускорение [x, y, vx, vy, ax, ay]
    2. Модель гравитации (ay ≈ g в пикселях/кадр²)
    3. Адаптивный шум измерений
    4. Возможность получить предсказание на N кадров вперёд
    """
    
    def __init__(self, dt: float = 1/60, gravity_pixels: float = 0.5,
                 process_noise: float = 0.1, measurement_noise: float = 10.0):
        """
        Args:
            dt: Время между кадрами (1/fps)
            gravity_pixels: Ускорение свободного падения в пикселях/кадр²
                           Для 1080p видео ~0.3-0.8 в зависимости от расстояния до камеры
            process_noise: Шум процесса (Q)
            measurement_noise: Шум измерений (R)
        """
        self.dt = dt
        self.gravity = gravity_pixels
        
        # Состояние: [x, y, vx, vy, ax, ay]
        self.state = np.zeros(6, dtype=np.float32)
        
        # Ковариация состояния
        self.P = np.eye(6, dtype=np.float32) * 1000
        
        # Матрица перехода состояния (с ускорением)
        # x' = x + vx*dt + 0.5*ax*dt²
        # vx' = vx + ax*dt
        # ax' = ax (ускорение константа между измерениями)
        self.F = np.array([
            [1, 0, dt, 0,  0.5*dt**2, 0],
            [0, 1, 0,  dt, 0,         0.5*dt**2],
            [0, 0, 1,  0,  dt,        0],
            [0, 0, 0,  1,  0,         dt],
            [0, 0, 0,  0,  1,         0],
            [0, 0, 0,  0,  0,         1]
        ], dtype=np.float32)
        
        # Матрица наблюдения (измеряем только x, y)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Шум процесса (Q) - больше для ускорения
        self.Q = np.diag([
            process_noise * 0.1,   # x
            process_noise * 0.1,   # y
            process_noise * 0.5,   # vx
            process_noise * 0.5,   # vy
            process_noise * 2.0,   # ax (больше неопределённость)
            process_noise * 2.0    # ay
        ]).astype(np.float32)
        
        # Шум измерений (R)
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Базовые значения для адаптации
        self._base_process_noise = process_noise
        self._base_measurement_noise = measurement_noise
        
        self.initialized = False
        self._missed_count = 0
        self._max_miss_before_reset = 30  # Сброс после 30 пропущенных кадров
    
    def predict(self) -> Optional[Tuple[float, float]]:
        """
        Предсказывает следующее состояние.
        
        Returns:
            (x, y) предсказанная позиция или None если не инициализирован
        """
        if not self.initialized:
            return None
        
        # Предсказание состояния
        self.state = self.F @ self.state
        
        # Добавляем гравитацию к вертикальному ускорению
        # ay стремится к gravity (мяч падает)
        self.state[5] = self.state[5] * 0.9 + self.gravity * 0.1
        
        # Предсказание ковариации
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return (float(self.state[0]), float(self.state[1]))
    
    def update(self, measurement: Tuple[float, float]) -> None:
        """
        Обновляет состояние по измерению.
        
        Args:
            measurement: (x, y) измеренная позиция
        """
        x, y = measurement
        
        if not self.initialized:
            # Инициализация: позиция известна, скорость и ускорение = 0
            self.state = np.array([x, y, 0, 0, 0, self.gravity], dtype=np.float32)
            self.initialized = True
            self._missed_count = 0
            return
        
        self._missed_count = 0
        
        # Измерение
        z = np.array([x, y], dtype=np.float32)
        
        # Инновация (разница между измерением и предсказанием)
        y_innov = z - self.H @ self.state
        
        # Ковариация инновации
        S = self.H @ self.P @ self.H.T + self.R
        
        # Коэффициент Калмана
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Обновление состояния
        self.state = self.state + K @ y_innov
        
        # Обновление ковариации
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        # Адаптация шума на основе инновации
        self._adapt_noise(y_innov)
    
    def _adapt_noise(self, innovation: np.ndarray) -> None:
        """
        Адаптивная настройка шума на основе величины инновации.
        Если инновация большая - увеличиваем шум измерений.
        """
        innov_magnitude = np.sqrt(innovation[0]**2 + innovation[1]**2)
        
        # Если инновация > 50 пикселей - увеличиваем R
        if innov_magnitude > 50:
            scale = min(innov_magnitude / 50, 5.0)
            self.R = np.eye(2, dtype=np.float32) * self._base_measurement_noise * scale
        else:
            # Возвращаем к базовому значению
            self.R = np.eye(2, dtype=np.float32) * self._base_measurement_noise
    
    def predict_n_frames(self, n: int) -> Optional[Tuple[float, float]]:
        """
        Предсказывает позицию через N кадров без изменения состояния.
        
        Args:
            n: Количество кадров вперёд
            
        Returns:
            (x, y) предсказанная позиция или None
        """
        if not self.initialized:
            return None
        
        # Копируем состояние
        state = self.state.copy()
        
        # Применяем F n раз
        for _ in range(n):
            state = self.F @ state
            state[5] = state[5] * 0.9 + self.gravity * 0.1
        
        return (float(state[0]), float(state[1]))
    
    def mark_missed(self) -> bool:
        """
        Отмечает пропущенный кадр.
        
        Returns:
            True если нужен сброс (слишком много пропусков)
        """
        self._missed_count += 1
        
        # Увеличиваем неопределённость при пропуске
        self.P *= 1.1
        
        if self._missed_count >= self._max_miss_before_reset:
            self.reset()
            return True
        return False
    
    def reset(self) -> None:
        """Сбрасывает фильтр в начальное состояние."""
        self.state = np.zeros(6, dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 1000
        self.initialized = False
        self._missed_count = 0
    
    def get_position(self) -> Optional[Tuple[float, float]]:
        """Возвращает текущую позицию."""
        if not self.initialized:
            return None
        return (float(self.state[0]), float(self.state[1]))
    
    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """Возвращает текущую скорость."""
        if not self.initialized:
            return None
        return (float(self.state[2]), float(self.state[3]))
    
    def get_acceleration(self) -> Optional[Tuple[float, float]]:
        """Возвращает текущее ускорение."""
        if not self.initialized:
            return None
        return (float(self.state[4]), float(self.state[5]))
    
    def get_speed(self) -> float:
        """Возвращает скаляр скорости."""
        vel = self.get_velocity()
        if vel is None:
            return 0.0
        return float(np.sqrt(vel[0]**2 + vel[1]**2))

