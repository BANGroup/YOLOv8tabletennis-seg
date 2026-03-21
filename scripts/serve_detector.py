#!/usr/bin/env python3
"""
Трекинг мяча v5 - ТОЧНАЯ КОПИЯ main_processor.py
Изменены только 2 порога:
- y_range: 15 → 12 (ловит более плоские отскоки)
- curvature: 10 → 8 (ловит более слабую вогнутость)

Оригинал: main_processor.py (НЕ ИЗМЕНЯТЬ!)
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

class SmartImpactDetector:
    """Умный детектор ударов + ДЕТЕКЦИЯ ПОДАЧИ (v4 - КОМБИНИРОВАННЫЙ)"""
    
    def __init__(self, frame_height, fps=59.94):
        self.frame_height = frame_height
        self.fps = fps
        self.table_zone_start = int(frame_height * 0.65)  # Нижние 35% = стол
        
        # ПАРАМЕТРЫ ИЗ SMART_TRACKING АНАЛИЗА
        self.max_direction_change = 180  # Градусы (разрешить обратные удары)
        self.min_rally_duration = 0.05    # Секунды (50 миллисекунд - минимум для отскока)
        self.min_velocity = 2.0          # Пиксели/кадр (фильтр шума)
        self.min_consecutive_detections = 3  # Детекции подряд
        self.min_frame_gap = 12          # Минимальный разрыв между ударами
        self.table_polygon = None        # Полигон стола (в пикселях)
        self.table_polygon_cv = None
        self.table_left_cv = None        # Левая половина стола
        self.table_right_cv = None       # Правая половина стола
        
        # === ДЕТЕКЦИЯ ПОДАЧИ v4 КОМБИНИРОВАННЫЙ ===
        # Метод 1: Пауза
        self.serve_pause_threshold = 2.5  # Пауза для нового розыгрыша
        self.last_impact_time = None
        self.last_impact_side = None
        
        # Метод 2: Позиция (первый удар розыгрыша)
        self.current_rally_first_side = None
        
        # Метод 3: Подброс
        self.toss_min_upward_frames = 4   # Мин. кадров движения вверх
        self.toss_min_height = 25         # Мин. пикселей подъёма
        self.toss_detection_window = 90   # Кадров до удара искать подброс
        self.ball_positions_raw = []
        self.last_toss_frame = -100
        
        # Результаты
        self.serve_count = 0
        self.serves = []
        self.serve_confidence = []  # Уверенность в подаче (0-3)
        
        # === ИГРОКИ И СЧЁТ ===
        # Player 1 = left side, Player 2 = right side
        self.serving_player = 1  # Кто сейчас подаёт (1 или 2)
        self.score_player1 = 0
        self.score_player2 = 0
        self.points_since_serve_change = 0  # Очков с последней смены подачи
        self.serve_change_interval = 2  # Смена подачи каждые 2 очка
        
        # === ДВОЙНОЙ ОТСКОК ===
        self.consecutive_same_side = 0  # Подряд ударов на одной стороне
        self.last_same_side_time = None  # Время последнего удара на той же стороне
        self.min_double_bounce_interval = 0.6  # Минимум 0.6 сек между ударами для двойного отскока
        self.last_rally_impacts = []  # Удары текущего розыгрыша
        self.double_bounce_events = []  # Зафиксированные двойные отскоки
        
        # === ИСТОРИЯ ОЧКОВ ===
        self.point_history = []  # История всех набранных очков
        
        # === СЕТКА И АУТ (REAL-TIME) ===
        self.net_polygon = None  # Полигон сетки
        self.net_zone_x = None   # X-координата центра сетки
        self.net_events = []     # События попадания в сетку
        self.out_events = []     # События аута (мяч за столом)
        
        # === REAL-TIME ДЕТЕКЦИЯ СОБЫТИЙ ===
        self.last_ball_positions = []  # Последние N позиций мяча для анализа траектории
        self.ball_lost_frame = None    # Кадр когда мяч пропал
        self.ball_lost_position = None # Позиция где мяч пропал
        self.ball_lost_velocity = None # Скорость когда пропал
        self.ball_lost_time = None     # Время когда пропал
        self.rally_in_progress = False # Идёт ли розыгрыш
        self.point_awarded_this_rally = False  # Очко уже присуждено в этом розыгрыше
        
        # === ИГРОВАЯ ЗОНА ===
        self.play_zone = None  # Будет установлена из BallTracker
        self.table_bottom_y = None  # Нижняя граница стола
        self.table_center_x = None  # Центр стола (для определения стороны удара)
        self.out_confirmation_time = 0.4  # Сек - мяч должен не вернуться чтобы засчитать аут
        
        # ИСТОРИЯ ТРАЕКТОРИИ ДЛЯ ОБНАРУЖЕНИЯ ВОГНУТОСТИ
        self.trajectory_history = deque(maxlen=10)  # Последние 10 нормализованных точек
        self.pixel_history = deque(maxlen=10)       # Оригинальные пиксельные координаты
        self.y_history = deque(maxlen=10)  # История нормализованных Y (высота над столом)
        self.live_impacts = []  # Живые события для HUD/логов
        self.last_impact_frame = None  # Последний подтверждённый кадр удара
        
        self.impacts = []
        self.impact_count = 0
        self.rally_start = None
        self.consecutive_detections = 0
    
    def calculate_curvature(self, points):
        """
        Вычисляет кривизну траектории (вогнутость)
        Возвращает ВЕЛИЧИНУ изменения направления (всегда положительная)
        """
        if len(points) < 3:
            return 0.0
        
        # Берем начало, середину, конец
        p1 = points[0]
        p2 = points[len(points) // 2]
        p3 = points[-1]
        
        # Вычисляем максимальное отклонение в середине от линии начало-конец
        # Если max(y) в середине (выпуклая вверх) - это вогнутость!
        y_min = min(p1[1], p3[1])
        y_max = max(p1[1], p3[1])
        
        # Если p2 находится ВЫШЕ обоих концов = вогнутость вверх
        curvature = p2[1] - max(p1[1], p3[1])
        
        # Если p2 находится НИЖЕ обоих концов (т.е. меньше min) = вогнутость вниз
        if p2[1] < min(p1[1], p3[1]):
            curvature = min(p1[1], p3[1]) - p2[1]
        
        return abs(curvature)
    
    def detect_impact(self, position, frame_idx, pixel_position=None):
        """
        Обнаруживает удар об стол по ВОГНУТОЙ траектории
        КЛЮЧЕВОЙ ПРИЗНАК: траектория вогнута вверх (вниз-вверх паттерн)
        """
        if not position:
            return False
        
        if self.table_polygon_cv is None:
            return False
        
        # Добавляем в историю
        self.trajectory_history.append(position)
        self.pixel_history.append(pixel_position if pixel_position is not None else position)
        self.y_history.append(position[1])
        
        # Нужно минимум 5 точек для анализа
        if len(self.trajectory_history) < 5:
            return False
        
        points_list = list(self.trajectory_history)
        first_y = points_list[0][1]
        middle_y = points_list[len(points_list)//2][1]
        last_y = points_list[-1][1]
        
        # ПРОВЕРКА 1: История показывает ЯСНЫЙ ПАТТЕРН вниз потом вверх
        goes_down = (first_y < middle_y)
        goes_up = (middle_y > last_y)
        
        if not (goes_down and goes_up):
            return False
        
        # ПРОВЕРКА 2: Достаточное движение по Y
        y_min = min(self.y_history)
        y_max = max(self.y_history)
        y_range = y_max - y_min
        
        if y_range < 8:
            return False
        
        # ПРОВЕРКА 3: Вогнутость траектории
        curvature = self.calculate_curvature(points_list)
        
        if curvature < 5:
            return False
        
        mid_idx = len(points_list) // 2
        bounce_point = points_list[mid_idx]
        pixel_points = list(self.pixel_history)
        bounce_pixel = pixel_points[mid_idx] if len(pixel_points) > mid_idx else bounce_point
        
        # Проверяем что удар на столе
        if self.table_polygon_cv is None:
            return False
        if cv2.pointPolygonTest(self.table_polygon_cv, (bounce_pixel[0], bounce_pixel[1]), False) < 0:
            return False
        
        # Определяем сторону ТОЛЬКО по X относительно центра стола (надёжный метод)
        center_x = self.table_center_x or 670
        table_side = 'left' if bounce_pixel[0] < center_x else 'right'
        
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
            'outgoing_speed': outgoing_speed,
            'table_side': table_side
        }
    
    def process_detection(self, position, frame_idx, fps_frame_time, impact_meta):
        """Обработка детекции с фильтром ложных срабатываний"""
        # Подсчитываем последовательные детекции
        self.consecutive_detections += 1
        
        if self.consecutive_detections < self.min_consecutive_detections:
            return False

        if not self.rally_start:
            self.rally_start = (frame_idx, fps_frame_time)

        # === ДЕТЕКЦИЯ ПОДАЧИ v4: КОМБИНИРОВАННЫЙ ===
        current_side = impact_meta.get('table_side', 'unknown')
        serve_signals = 0  # Счётчик сигналов подачи (0-3)
        toss_detected = self.detect_toss(frame_idx)
        is_new_rally = False
        
        # Сигнал 1: Пауза > threshold
        if self.last_impact_time is None:
            serve_signals += 1  # Первый удар = подача
            is_new_rally = True
        elif (fps_frame_time - self.last_impact_time) >= self.serve_pause_threshold:
            serve_signals += 1
            is_new_rally = True
            
            # === FALLBACK: Если очко не было присуждено в real-time ===
            if not self.point_awarded_this_rally and len(self.last_rally_impacts) >= 2:
                last_side = self.last_impact_side
                if last_side and last_side != 'unknown':
                    winner = 2 if last_side == 'left' else 1
                    reason = f"P{1 if last_side == 'left' else 2} missed"
                    self.award_point(winner, reason=reason)
            
            # Сбрасываем для нового розыгрыша
            self.point_awarded_this_rally = False
        
        # Сигнал 2: Подброс обнаружен
        if toss_detected and (frame_idx - self.last_toss_frame) > 30:
            serve_signals += 1
            self.last_toss_frame = frame_idx
        
        # Сигнал 3: Первый удар на той же стороне (свой стол при подаче)
        if is_new_rally:
            self.current_rally_first_side = current_side
            # При подаче первый отскок на своей стороне
            serve_signals += 1
        
        # Подача если >= 2 сигнала ИЛИ первый удар
        is_serve = (serve_signals >= 2) or (self.last_impact_time is None)
        
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
            'outgoing_speed': float(impact_meta['outgoing_speed']),
            'table_side': current_side,
            'is_serve': is_serve,
            'serve_confidence': serve_signals,
            'toss_detected': toss_detected,
            'is_new_rally': is_new_rally
        }
        
        should_add_new = (
            self.last_impact_frame is None or
            (frame_idx - self.last_impact_frame) >= self.min_frame_gap
        )
        
        if should_add_new:
            self.live_impacts.append(live_event)
            self.last_impact_frame = frame_idx
            self.last_impact_time = fps_frame_time
            
            # === ДВОЙНОЙ ОТСКОК ===
            # НЕ считаем двойным отскоком первый удар после подачи
            # И проверяем минимальный интервал времени между ударами
            if current_side != 'unknown' and not is_serve:
                if self.last_impact_side == current_side:
                    # Проверяем интервал времени
                    time_since_last = fps_frame_time - (self.last_same_side_time or 0)
                    
                    if time_since_last >= self.min_double_bounce_interval:
                        self.consecutive_same_side += 1
                        if self.consecutive_same_side >= 2:
                            # Реальный двойной отскок (достаточный интервал)
                            double_bounce = {
                                'frame': frame_idx,
                                'time': fps_frame_time,
                                'side': current_side,
                                'player_fault': 1 if current_side == 'left' else 2,
                                'interval': time_since_last
                            }
                            self.double_bounce_events.append(double_bounce)
                            live_event['is_double_bounce'] = True
                            
                            # Сразу присуждаем очко за двойной отскок!
                            if not self.point_awarded_this_rally:
                                winner = 2 if current_side == 'left' else 1
                                self.award_point(winner, reason=f"Double bounce P{1 if current_side == 'left' else 2}")
                                self.point_awarded_this_rally = True
                    
                    self.last_same_side_time = fps_frame_time
                else:
                    self.consecutive_same_side = 1
                    self.last_same_side_time = fps_frame_time
            
            self.last_impact_side = current_side
            self.last_rally_impacts.append(live_event)
            
            if is_serve:
                self.serve_count += 1
                self.serves.append(live_event)
                self.serve_confidence.append(serve_signals)
                # Определяем подающего по стороне первого удара
                # left = Player 1, right = Player 2
                if current_side == 'left':
                    self.serving_player = 1
                elif current_side == 'right':
                    self.serving_player = 2
                # Новый розыгрыш - сбрасываем
                self.last_rally_impacts = [live_event]
                self.consecutive_same_side = 1
        # НЕ обновляем старые удары - они фиксируются навсегда
        
        self.impact_count = len(self.live_impacts)
        self.consecutive_detections = 0  # сбрасываем, чтобы следующий удар требовал новый паттерн
        return should_add_new
    
    def detect_toss(self, current_frame):
        """Поиск подброса мяча перед ударом (вертикальное движение вверх)"""
        if len(self.ball_positions_raw) < self.toss_min_upward_frames + 2:
            return False
        
        # Ищем в последних N кадрах
        start_idx = max(0, len(self.ball_positions_raw) - self.toss_detection_window)
        positions = self.ball_positions_raw[start_idx:]
        
        # Ищем последовательность кадров с движением вверх (y уменьшается)
        upward_streak = 0
        max_upward_streak = 0
        streak_start_y = None
        streak_end_y = None
        best_start_y = None
        best_end_y = None
        
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]
            
            if curr_pos[1] < prev_pos[1]:  # Движение вверх (y уменьшается)
                if upward_streak == 0:
                    streak_start_y = prev_pos[1]
                upward_streak += 1
                streak_end_y = curr_pos[1]
            else:
                if upward_streak > max_upward_streak:
                    max_upward_streak = upward_streak
                    best_start_y = streak_start_y
                    best_end_y = streak_end_y
                upward_streak = 0
        
        # Проверка последней серии
        if upward_streak > max_upward_streak:
            max_upward_streak = upward_streak
            best_start_y = streak_start_y
            best_end_y = streak_end_y
        
        # Проверка: достаточно ли длинный подброс и высокий
        if max_upward_streak >= self.toss_min_upward_frames:
            if best_start_y and best_end_y:
                height = best_start_y - best_end_y  # Положительно = вверх
                if height >= self.toss_min_height:
                    return True
        
        return False
    
    def add_ball_position(self, pos, frame_idx):
        """Сохранение позиции мяча для детекции подброса"""
        self.ball_positions_raw.append((pos[0], pos[1], frame_idx))
    
    def award_point(self, winner_player, reason=""):
        """Начислить очко игроку и проверить смену подачи"""
        old_server = self.serving_player
        
        if winner_player == 1:
            self.score_player1 += 1
        else:
            self.score_player2 += 1
        
        self.points_since_serve_change += 1
        
        # Смена подачи каждые 2 очка
        serve_changed = False
        if self.points_since_serve_change >= self.serve_change_interval:
            self.serving_player = 2 if self.serving_player == 1 else 1
            self.points_since_serve_change = 0
            serve_changed = True
        
        point_event = {
            'winner': winner_player,
            'reason': reason,
            'score': (self.score_player1, self.score_player2),
            'server_before': old_server,
            'server_after': self.serving_player,
            'serve_changed': serve_changed
        }
        
        # Сохраняем в историю
        if not hasattr(self, 'point_history'):
            self.point_history = []
        self.point_history.append(point_event)
        
        return point_event
    
    def finalize_rally(self, frame_idx, fps_frame_time):
        """Завершить ралли и проверить минимальную длительность"""
        if not self.rally_start:
            self.consecutive_detections = 0
            # Только тут очищаем историю - когда мяч надолго потерян
            self.trajectory_history.clear()
            self.y_history.clear()
            self.pixel_history.clear()
            return False
        
        start_frame, start_time = self.rally_start
        duration = fps_frame_time - start_time
        
        # Проверяем минимальную длительность
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
            # НЕ очищаем историю - нужна для следующих ударов!
            return True
        self.rally_start = None
        self.consecutive_detections = 0
        # НЕ очищаем историю - нужна для следующих ударов!
        return False

class BallTracker:
    def __init__(self, model_path, frame_width, frame_height, fps, output_dir, impact_params=None):
        self.model = YOLO(model_path)
        self.kalman = KalmanFilter()
        self.impact_detector = SmartImpactDetector(frame_height, fps)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.output_dir = output_dir
        
        # Если передали параметры из конфига - используем их
        if impact_params:
            if 'min_rally_duration' in impact_params:
                self.impact_detector.min_rally_duration = impact_params['min_rally_duration']
            if 'min_velocity' in impact_params:
                self.impact_detector.min_velocity = impact_params['min_velocity']
            if 'min_consecutive_detections' in impact_params:
                self.impact_detector.min_consecutive_detections = impact_params['min_consecutive_detections']
            if 'max_direction_change' in impact_params:
                self.impact_detector.max_direction_change = impact_params['max_direction_change']
            if 'min_frame_gap' in impact_params:
                self.impact_detector.min_frame_gap = impact_params['min_frame_gap']
        
        self.trajectory = deque(maxlen=100)
        self.velocities = deque(maxlen=15)
        self.total_detections = 0
        self.missed_frames = 0
        self.kalman_predictions_used = 0
        self.fps = fps
        self.table_polygon = None
        self.table_class_id = impact_params.get('table_class_id', 4) if impact_params else 4
        self.net_class_id = 1  # Net class
        
        # === ИГРОВАЯ ЗОНА ===
        # По X - широко (игроки отходят далеко)
        # По Y - ограничено столом (сверху и снизу)
        self.play_zone = None
        self.play_zone_margin_x = 300  # Большой запас по X
        self.play_zone_margin_y = 100  # Небольшой запас по Y (для подбросов)
        self.table_bottom_y = None
        self.table_top_y = None
        self.net_bbox = None   # Bounding box сетки [x1,y1,x2,y2]
        
    def detect_ball(self, frame):
        results = self.model.predict(frame, conf=0.1, verbose=False)
        result = results[0]
        self._maybe_update_table_polygon(result, frame)
        self._maybe_update_net(result)
        
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
        
        # Собираем все полигоны стола
        table_polys = []
        for idx, cls in enumerate(class_ids):
            if cls == self.table_class_id and idx < len(polygons):
                poly = np.array(polygons[idx], dtype=np.float32)
                if poly.shape[0] >= 4:
                    center_x = np.mean(poly[:, 0])
                    table_polys.append((center_x, poly))
        
        if len(table_polys) < 1:
            return
        
        # Сортируем по X (левый первый)
        table_polys.sort(key=lambda x: x[0])
        
        # Если два полигона - левый и правый
        if len(table_polys) >= 2:
            left_poly = table_polys[0][1]
            right_poly = table_polys[-1][1]
            self.impact_detector.table_left_cv = left_poly.reshape(-1, 1, 2).astype(np.float32)
            self.impact_detector.table_right_cv = right_poly.reshape(-1, 1, 2).astype(np.float32)
            # Объединяем в общий полигон
            all_points = np.vstack([left_poly, right_poly])
            hull = cv2.convexHull(all_points)
            polygon = hull.reshape(-1, 2)
        else:
            # Один полигон - используем как есть
            polygon = table_polys[0][1]
            self.impact_detector.table_left_cv = None
            self.impact_detector.table_right_cv = None
        self.table_polygon = polygon
        self.impact_detector.table_polygon = polygon
        self.impact_detector.table_polygon_cv = polygon.reshape(-1, 1, 2).astype(np.float32)
        self.impact_detector.table_zone_start = int(np.mean(polygon[:, 1]))
        debug_img = frame.copy()
        overlay = debug_img.copy()
        
        # Рисуем левый полигон (синий)
        if self.impact_detector.table_left_cv is not None:
            left_poly = self.impact_detector.table_left_cv.astype(np.int32)
            cv2.polylines(debug_img, [left_poly], True, (255, 0, 0), 2)
            cv2.fillPoly(overlay, [left_poly], (255, 0, 0))
            print(f"[+] Left table polygon: {len(left_poly)} points")
        
        # Рисуем правый полигон (зеленый)
        if self.impact_detector.table_right_cv is not None:
            right_poly = self.impact_detector.table_right_cv.astype(np.int32)
            cv2.polylines(debug_img, [right_poly], True, (0, 255, 0), 2)
            cv2.fillPoly(overlay, [right_poly], (0, 255, 0))
            print(f"[+] Right table polygon: {len(right_poly)} points")
        
        # Общий полигон (красный контур)
        cv2.polylines(debug_img, [self.impact_detector.table_polygon_cv.astype(np.int32)], True, (0, 0, 255), 2)
        
        debug_img = cv2.addWeighted(overlay, 0.2, debug_img, 0.8, 0)
        if self.output_dir:
            debug_path = os.path.join(self.output_dir, "table_polygon_debug.png")
            cv2.imwrite(debug_path, debug_img)
            print(f"[+] Table polygon saved → {debug_path}")
        
        # === ОПРЕДЕЛЯЕМ ИГРОВУЮ ЗОНУ И ЦЕНТР СТОЛА ===
        table_x_min = np.min(polygon[:, 0])
        table_x_max = np.max(polygon[:, 0])
        table_y_min = np.min(polygon[:, 1])
        table_y_max = np.max(polygon[:, 1])
        table_center_x = (table_x_min + table_x_max) / 2
        
        self.table_top_y = table_y_min
        self.table_bottom_y = table_y_max
        self.table_center_x = table_center_x
        self.play_zone = {
            'x_min': max(0, table_x_min - self.play_zone_margin_x),
            'x_max': min(self.frame_width, table_x_max + self.play_zone_margin_x),
            'y_min': max(0, table_y_min - self.play_zone_margin_y),
            'y_max': table_y_max + 50  # Небольшой запас снизу
        }
        self.impact_detector.play_zone = self.play_zone
        self.impact_detector.table_bottom_y = self.table_bottom_y
        self.impact_detector.table_top_y = table_y_min
        self.impact_detector.table_center_x = table_center_x
        print(f"[+] Play zone: X[{self.play_zone['x_min']:.0f}-{self.play_zone['x_max']:.0f}], Y[{self.play_zone['y_min']:.0f}-{self.play_zone['y_max']:.0f}]")
        print(f"[+] Table: X[{table_x_min:.0f}-{table_x_max:.0f}], center={table_center_x:.0f}")
    
    def _maybe_update_net(self, result):
        """Детекция сетки (один раз)"""
        if self.net_bbox is not None:
            return
        if result.boxes is None:
            return
        
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()
        
        for box, cls in zip(boxes, class_ids):
            if cls == self.net_class_id:
                self.net_bbox = box  # [x1, y1, x2, y2]
                net_center_x = (box[0] + box[2]) / 2
                self.impact_detector.net_zone_x = net_center_x
                print(f"[+] Net detected: x={net_center_x:.0f}, bbox={box}")
                break
    
    def is_ball_in_net_zone(self, ball_pos):
        """Проверка: мяч в зоне сетки?"""
        if self.net_bbox is None or ball_pos is None:
            return False
        x, y = ball_pos
        x1, y1, x2, y2 = self.net_bbox
        # Расширяем зону сетки по X на 20 пикселей
        margin = 20
        return (x1 - margin) <= x <= (x2 + margin) and y1 <= y <= y2
    
    def is_ball_out_of_table(self, ball_pos):
        """Проверка: мяч за пределами стола?"""
        if self.table_polygon is None or ball_pos is None:
            return False
        point = np.array([[ball_pos[0], ball_pos[1]]], dtype=np.float32)
        # cv2.pointPolygonTest возвращает > 0 если внутри, < 0 если снаружи
        dist = cv2.pointPolygonTest(
            self.impact_detector.table_polygon_cv, 
            (float(ball_pos[0]), float(ball_pos[1])), 
            False
        )
        return dist < 0
    
    def track_ball(self, frame, frame_idx, fps_time):
        self._current_time = fps_time
        detection = self.detect_ball(frame)
        
        if detection:
            self.total_detections += 1
            self.missed_frames = 0
            self.kalman.update(detection)
            self.trajectory.append(detection)
            
            # Сохраняем позицию для детекции подброса
            self.impact_detector.add_ball_position(detection, frame_idx)
            
            # Вычисляем скорость
            if len(self.trajectory) > 1:
                last_pos = self.trajectory[-2]
                dx = detection[0] - last_pos[0]
                dy = detection[1] - last_pos[1]
                speed = math.sqrt(dx*dx + dy*dy)
                self.velocities.append(speed)
                
                # Анализируем удар об стол
                impact_meta = self.impact_detector.detect_impact(detection, frame_idx, pixel_position=detection)
                if impact_meta:
                    self.impact_detector.process_detection(detection, frame_idx, fps_time, impact_meta)
        else:
            self.missed_frames += 1
            
            # Если пропустили > 5 кадров - завершаем ралли
            if self.missed_frames > 5:
                self.impact_detector.finalize_rally(frame_idx, fps_time)
            
            if self.kalman.initialized:
                pred = self.kalman.predict()
                if pred:
                    self.kalman_predictions_used += 1
        
        return detection
    
    def _detect_rally_end_event(self, frame_idx, fps_time):
        """Определяет причину окончания розыгрыша по траектории мяча"""
        if self.impact_detector.point_awarded_this_rally:
            return  # Очко уже присуждено
        
        # Минимум 1 удар должен быть зафиксирован для розыгрыша
        if len(self.impact_detector.last_rally_impacts) < 1:
            return
        
        last_pos = self.impact_detector.ball_lost_position
        last_vel = self.impact_detector.ball_lost_velocity
        
        if last_pos is None or last_vel is None:
            return
        
        # Используем центр стола для определения стороны (более надёжно чем детекция сетки)
        center_x = self.impact_detector.table_center_x or self.impact_detector.net_zone_x or (self.frame_width / 2)
        net_x = self.impact_detector.net_zone_x or center_x  # Для детекции удара в сетку
        
        # Определяем направление и скорость
        vx, vy = last_vel
        speed = math.sqrt(vx*vx + vy*vy)
        
        # Горизонтальная скорость важнее для определения направления
        horizontal_speed = abs(vx)
        
        # === ДЕТЕКЦИЯ СЕТКИ ===
        # Мяч пропал близко к центру стола (где сетка)
        # Используем center_x т.к. net_x может быть неточным
        near_center = abs(last_pos[0] - center_x) < 80
        
        # Определяем с какой стороны мяч пришёл
        ball_from_left = last_pos[0] < center_x
        ball_from_right = last_pos[0] >= center_x
        
        # Проверяем что мяч двигался к центру
        moving_towards_center = (ball_from_left and vx >= 0) or (ball_from_right and vx <= 0)
        
        if near_center and moving_towards_center and speed < 15:  # Низкая скорость = остановился у сетки
            if ball_from_left:
                self.impact_detector.award_point(2, reason="P1 net")
                self.impact_detector.net_events.append({'time': fps_time, 'player': 1})
            else:
                self.impact_detector.award_point(1, reason="P2 net")
                self.impact_detector.net_events.append({'time': fps_time, 'player': 2})
            self.impact_detector.point_awarded_this_rally = True
            return
        
        # === ДЕТЕКЦИЯ АУТА ===
        # Мяч пропал далеко от центра + высокая скорость
        far_from_center = abs(last_pos[0] - center_x) > 150
        is_horizontal_motion = horizontal_speed > abs(vy)
        
        if far_from_center and horizontal_speed > 10 and is_horizontal_motion:
            # Определяем кто выбил в аут по позиции относительно центра
            if last_pos[0] > center_x:  # Мяч справа от центра - P1 выбил вправо
                self.impact_detector.award_point(2, reason="P1 out")
                self.impact_detector.out_events.append({
                    'time': fps_time, 'player_fault': 1, 
                    'frame': frame_idx, 'position': last_pos, 'type': 'long'
                })
            else:  # Мяч слева от центра - P2 выбил влево
                self.impact_detector.award_point(1, reason="P2 out")
                self.impact_detector.out_events.append({
                    'time': fps_time, 'player_fault': 2,
                    'frame': frame_idx, 'position': last_pos, 'type': 'long'
                })
            self.impact_detector.point_awarded_this_rally = True
            return
    
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
        if self.impact_detector.table_polygon_cv is not None:
            poly = self.impact_detector.table_polygon_cv.astype(np.int32)
            cv2.polylines(annotated, [poly], True, (120, 120, 255), 2)
        
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
        impacts_logged = len(self.impact_detector.live_impacts)
        cv2.putText(annotated, f"[RED] IMPACTS (SMART): {impacts_logged}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_pos += 30

        last_event = None
        if impacts_logged > 0:
            last_event = self.impact_detector.live_impacts[-1]
            for i, event in enumerate(self.impact_detector.live_impacts):
                hit_x = int(event['x'])
                hit_y = int(event['y'])
                # Красные круги как было раньше
                cv2.circle(annotated, (hit_x, hit_y), 8, (0, 0, 255), -1)
                cv2.circle(annotated, (hit_x, hit_y), 10, (0, 0, 255), 2)

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
        
        # === СЧЁТ (верхний центр) ===
        frame_w = annotated.shape[1]
        frame_h = annotated.shape[0]
        score1 = self.impact_detector.score_player1
        score2 = self.impact_detector.score_player2
        
        # Большой счёт по центру сверху
        score_text = f"{score1} : {score2}"
        score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        score_x = (frame_w - score_size[0]) // 2
        
        # Фон для счёта
        cv2.rectangle(annotated, (score_x - 15, 5), (score_x + score_size[0] + 15, 50), (0, 0, 0), -1)
        cv2.rectangle(annotated, (score_x - 15, 5), (score_x + score_size[0] + 15, 50), (255, 255, 255), 2)
        cv2.putText(annotated, score_text, (score_x, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Имена игроков под счётом
        p1_text = "P1"
        p2_text = "P2"
        cv2.putText(annotated, p1_text, (score_x - 5, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(annotated, p2_text, (score_x + score_size[0] - 15, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # === ПОДАЧА ИГРОКА (правый верхний угол) ===
        serving = self.impact_detector.serving_player
        serve_text = f"Serve: P{serving}"
        text_size = cv2.getTextSize(serve_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = frame_w - text_size[0] - 15
        
        # Фон для текста
        cv2.rectangle(annotated, (text_x - 5, 10), (frame_w - 10, 40), (0, 0, 0), -1)
        # Цвет: жёлтый для Player 1 (left), голубой для Player 2 (right)
        color = (0, 255, 255) if serving == 1 else (255, 255, 0)
        cv2.putText(annotated, serve_text, (text_x, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # === ДВОЙНОЙ ОТСКОК (красное предупреждение) ===
        if self.impact_detector.double_bounce_events:
            last_db = self.impact_detector.double_bounce_events[-1]
            # Показываем 2 секунды после события
            if hasattr(self, '_current_time') and (self._current_time - last_db['time']) < 2.0:
                fault_player = last_db['player_fault']
                db_text = f"DOUBLE BOUNCE! P{fault_player} fault"
                db_size = cv2.getTextSize(db_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                db_x = (frame_w - db_size[0]) // 2
                cv2.rectangle(annotated, (db_x - 10, 80), (db_x + db_size[0] + 10, 110), (0, 0, 150), -1)
                cv2.putText(annotated, db_text, (db_x, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # === ПОПАДАНИЕ В СЕТКУ (оранжевое предупреждение) ===
        if self.impact_detector.net_events:
            last_net = self.impact_detector.net_events[-1]
            if hasattr(self, '_current_time') and (self._current_time - last_net['time']) < 2.0:
                net_text = f"NET FAULT! P{last_net['player']}"
                net_size = cv2.getTextSize(net_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                net_x = (frame_w - net_size[0]) // 2
                cv2.rectangle(annotated, (net_x - 10, 80), (net_x + net_size[0] + 10, 110), (0, 100, 255), -1)
                cv2.putText(annotated, net_text, (net_x, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # === АУТ (синее предупреждение) ===
        if self.impact_detector.out_events:
            last_out = self.impact_detector.out_events[-1]
            if hasattr(self, '_current_time') and (self._current_time - last_out['time']) < 2.0:
                out_text = f"OUT! P{last_out['player_fault']} fault"
                out_size = cv2.getTextSize(out_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                out_x = (frame_w - out_size[0]) // 2
                cv2.rectangle(annotated, (out_x - 10, 80), (out_x + out_size[0] + 10, 110), (255, 100, 0), -1)
                cv2.putText(annotated, out_text, (out_x, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # === Рисуем зону сетки (если есть) ===
        if self.net_bbox is not None:
            x1, y1, x2, y2 = map(int, self.net_bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 1)  # Оранжевая рамка сетки
        
        return annotated

def load_config(config_path):
    """Загружает конфиг из JSON файла"""
    if not os.path.exists(config_path):
        print(f"[!] Ошибка: конфиг не найден: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # CLI параметры
    parser = argparse.ArgumentParser(description='Трекинг мяча в настольном теннисе')
    parser.add_argument('--video', required=True, help='Путь к видео файлу')
    parser.add_argument('--start', type=int, required=True, help='Начало сегмента в секундах')
    parser.add_argument('--end', type=int, required=True, help='Конец сегмента в секундах')
    parser.add_argument('--config', default='scripts/config/detection_params.json', 
                       help='Путь к конфиг файлу (default: scripts/config/detection_params.json)')
    
    args = parser.parse_args()
    
    # Загружаем конфиг
    config = load_config(args.config)
    if not config:
        return
    
    # Переопределяем из CLI
    video_path = args.video
    start_time = args.start
    end_time = args.end
    
    # Используем конфиг
    model_path = config['yolo']['model_path']
    output_dir = config['output']['output_dir']
    
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
    print(f"[*] Сегмент: {start_time}-{end_time} сек")
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Передаем параметры impact_detection из конфига
    impact_params = config.get('impact_detection', {})
    tracker = BallTracker(model_path, frame_width, frame_height, fps, output_dir, impact_params)
    
    print("[*] Создаю видео со SMART обнаружением ударов...")
    output_video = os.path.join(output_dir, "tracked_v5.mp4")
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
        frame_duration = time.perf_counter() - frame_loop_start
        frame_durations.append(frame_duration)
        
        if processed_count % 30 == 0 or processed_count == 1:
            elapsed = time.perf_counter() - processing_start
            effective_fps = processed_count / elapsed if elapsed > 0 else 0
            progress_pct = (processed_count / total_frames) * 100
            print(f"[PROGRESS] {processed_count}/{total_frames} "
                  f"({progress_pct:.1f}%) | SMART: {tracker.impact_detector.impact_count} | "
                  f"FPS: {effective_fps:.2f}")
    
    cap.release()
    out.release()
    processing_total = time.perf_counter() - processing_start
    
    # Завершаем ПОСЛЕДНИЙ ралли если он не завершен
    # NOTE: fps_frame_time это глобальное время от начала видео, а не от начала сегмента!
    final_frame_in_video = start_frame + processed_count
    final_fps_time = (final_frame_in_video / tracker.fps) if tracker.fps > 0 else 0
    tracker.impact_detector.finalize_rally(final_frame_in_video, final_fps_time)
    
    print(f"\n[+] ВИДЕО: {output_video}")
    
    # Статистика по сторонам стола
    left_impacts = sum(1 for imp in tracker.impact_detector.live_impacts if imp.get('table_side') == 'left')
    right_impacts = sum(1 for imp in tracker.impact_detector.live_impacts if imp.get('table_side') == 'right')
    
    print(f"\n[=] РЕЗУЛЬТАТЫ (SMART ЛОГИКА из smart_tracking):")
    print(f"    Кадров: {processed_count}")
    print(f"    Детекций мяча: {tracker.total_detections}")
    print(f"    Пропущено: {tracker.missed_frames}")
    print(f"    SMART УДАРОВ об СТОЛ: {tracker.impact_detector.impact_count}")
    print(f"      └─ Left:  {left_impacts}")
    print(f"      └─ Right: {right_impacts}")
    print(f"    ПОДАЧ (комбинированный): {tracker.impact_detector.serve_count}")
    for i, serve in enumerate(tracker.impact_detector.serves):
        conf = tracker.impact_detector.serve_confidence[i] if i < len(tracker.impact_detector.serve_confidence) else 0
        toss = "✓" if serve.get('toss_detected') else "✗"
        player = 1 if serve['table_side'] == 'left' else 2
        print(f"      └─ Подача {i+1}: Player {player}, t={serve['time']:.2f}s, уверенность={conf}/3, подброс={toss}")
    
    # Двойные отскоки
    double_bounces = tracker.impact_detector.double_bounce_events
    print(f"    ДВОЙНЫХ ОТСКОКОВ: {len(double_bounces)}")
    for db in double_bounces:
        print(f"      └─ t={db['time']:.2f}s, Player {db['player_fault']} fault")
    
    # Счёт
    s1 = tracker.impact_detector.score_player1
    s2 = tracker.impact_detector.score_player2
    print(f"    СЧЁТ: Player 1 [{s1}] : [{s2}] Player 2")
    print(f"    Текущий подающий: Player {tracker.impact_detector.serving_player}")
    
    # История очков
    point_history = tracker.impact_detector.point_history
    if point_history:
        print(f"    ИСТОРИЯ ОЧКОВ ({len(point_history)} очков):")
        for i, pt in enumerate(point_history):
            serve_info = " [СМЕНА ПОДАЧИ]" if pt['serve_changed'] else ""
            print(f"      └─ Очко {i+1}: Player {pt['winner']} выиграл ({pt['reason']}) → {pt['score'][0]}:{pt['score'][1]}{serve_info}")
    
    # Сетка и ауты
    net_events = tracker.impact_detector.net_events
    out_events = tracker.impact_detector.out_events
    print(f"    ПОПАДАНИЙ В СЕТКУ: {len(net_events)}")
    for ne in net_events:
        print(f"      └─ t={ne['time']:.2f}s, Player {ne['player']} fault")
    print(f"    АУТОВ (мяч за столом): {len(out_events)}")
    for oe in out_events:
        out_type = oe.get('type', 'unknown')
        type_str = "(перелёт)" if out_type == 'long' else "(недолёт)" if out_type == 'short' else ""
        print(f"      └─ t={oe['time']:.2f}s, Player {oe['player_fault']} fault {type_str}")
    
    print(f"    Kalman предсказаний: {tracker.kalman_predictions_used}")
    
    print(f"\n[*] ИСПОЛЬЗУЕМЫЕ ПАРАМЕТРЫ из smart_tracking анализа:")
    print(f"    max_direction_change: {tracker.impact_detector.max_direction_change}°")
    print(f"    min_rally_duration: {tracker.impact_detector.min_rally_duration} сек (из конфига: {impact_params.get('min_rally_duration', 'НЕ ЗАДАН')})")
    print(f"    min_velocity: {tracker.impact_detector.min_velocity} пикс/кадр")
    print(f"    min_consecutive_detections: {tracker.impact_detector.min_consecutive_detections}")
    
    avg_frame_time = (processing_total / processed_count) if processed_count > 0 else 0
    min_frame_time = min(frame_durations) if frame_durations else 0
    max_frame_time = max(frame_durations) if frame_durations else 0
    effective_fps_total = processed_count / processing_total if processing_total > 0 else 0
    
    print(f"\n[TIMING] Total: {processing_total:.2f}s | "
          f"Avg/frame: {avg_frame_time*1000:.2f} ms | "
          f"Min: {min_frame_time*1000:.2f} ms | "
          f"Max: {max_frame_time*1000:.2f} ms | "
          f"Effective FPS: {effective_fps_total:.2f}")
    
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
        },
        'timing': {
            'total_seconds': processing_total,
            'avg_ms_per_frame': avg_frame_time * 1000,
            'min_ms_per_frame': min_frame_time * 1000,
            'max_ms_per_frame': max_frame_time * 1000,
            'effective_fps': effective_fps_total,
            'processed_frames': processed_count
        },
        'segment': {
            'video': video_path,
            'start': start_time,
            'end': end_time
        }
    }
    
    json_path = os.path.join(output_dir, "smart_impacts_analysis.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    live_impacts_path = os.path.join(output_dir, "live_impacts.json")
    with open(live_impacts_path, 'w', encoding='utf-8') as f:
        json.dump(tracker.impact_detector.live_impacts, f, ensure_ascii=False, indent=2)
    
    print(f"    Анализ: {json_path}")
    print(f"    Live impacts log: {live_impacts_path}")

if __name__ == "__main__":
    main()

