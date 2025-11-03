#!/usr/bin/env python3
"""
Улучшенный трекинг мяча с умными алгоритмами детекции и предсказания
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import time
from collections import deque
import math
import json
from scipy import ndimage
from scipy.optimize import minimize

class EnhancedBallTracker:
    """Улучшенный трекер мяча с адаптивными алгоритмами"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Адаптивные пороги детекции
        self.base_confidence = 0.3
        self.min_confidence = 0.15
        self.max_confidence = 0.8
        self.current_confidence = self.base_confidence
        
        # Фильтры размера мяча
        self.min_ball_area = 50      # Минимальная площадь мяча
        self.max_ball_area = 2000    # Максимальная площадь мяча
        self.expected_ball_size = 400  # Ожидаемый размер мяча
        
        # Физические параметры мяча
        self.max_speed = 200         # Максимальная скорость (пикселей/кадр)
        self.max_acceleration = 50   # Максимальное ускорение
        self.gravity = 0.5           # Гравитация (пикселей/кадр²)
        
        # История для анализа
        self.trajectory = deque(maxlen=50)
        self.velocities = deque(maxlen=10)
        self.confidences = deque(maxlen=20)
        
        # Адаптивные параметры
        self.detection_history = deque(maxlen=10)
        self.missed_frames = 0
        self.consecutive_detections = 0
        
        # Цвета для визуализации
        self.colors = {
            'ball': (0, 255, 0),      # Зеленый
            'trajectory': (255, 0, 255),  # Пурпурный
            'prediction': (0, 255, 255),  # Желтый
            'search_area': (255, 255, 0), # Голубой
            'velocity': (255, 0, 0)   # Красный
        }
        
        # Статистика
        self.total_detections = 0
        self.false_positives = 0
        self.missed_detections = 0
        
    def adaptive_confidence(self):
        """Адаптивная настройка порога уверенности"""
        if len(self.confidences) < 5:
            return self.current_confidence
        
        # Анализируем последние детекции
        recent_conf = list(self.confidences)[-5:]
        avg_conf = np.mean(recent_conf)
        
        # Если детекции стабильны, можем повысить порог
        if self.consecutive_detections > 5 and avg_conf > 0.5:
            self.current_confidence = min(self.max_confidence, self.current_confidence + 0.05)
        # Если много пропусков, снижаем порог
        elif self.missed_frames > 3:
            self.current_confidence = max(self.min_confidence, self.current_confidence - 0.1)
        
        return self.current_confidence
    
    def validate_ball_detection(self, detection, frame_shape):
        """Валидация детекции мяча по физическим параметрам"""
        if not detection:
            return False
        
        box = detection['box']
        center = detection['center']
        confidence = detection['confidence']
        size = detection['size']
        
        # Проверка размера
        if size < self.min_ball_area or size > self.max_ball_area:
            return False
        
        # Проверка формы (соотношение сторон)
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        
        # Мяч должен быть примерно круглым
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:
            return False
        
        # Проверка позиции (мяч не должен быть в углах)
        h, w = frame_shape[:2]
        margin = min(w, h) * 0.1
        if (center[0] < margin or center[0] > w - margin or 
            center[1] < margin or center[1] > h - margin):
            return False
        
        # Проверка скорости (если есть предыдущие позиции)
        if len(self.trajectory) > 0:
            last_pos = self.trajectory[-1]['position']
            dx = center[0] - last_pos[0]
            dy = center[1] - last_pos[1]
            speed = math.sqrt(dx*dx + dy*dy)
            
            if speed > self.max_speed:
                return False
        
        return True
    
    def detect_ball_enhanced(self, frame):
        """Улучшенная детекция мяча с адаптивными параметрами"""
        h, w = frame.shape[:2]
        
        # Адаптивный порог уверенности
        conf_threshold = self.adaptive_confidence()
        
        # Детекция с текущими параметрами
        results = self.model.predict(
            frame,
            conf=conf_threshold,
            iou=0.3,
            classes=[0],  # Только мяч
            verbose=False
        )
        
        detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                if class_id == 0:  # Ball
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    size = (x2 - x1) * (y2 - y1)
                    
                    detection = {
                        'box': box,
                        'center': (center_x, center_y),
                        'confidence': conf,
                        'size': size,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                    
                    # Валидация детекции
                    if self.validate_ball_detection(detection, frame.shape):
                        detections.append(detection)
        
        # Выбираем лучшую детекцию
        if detections:
            # Сортируем по комбинации уверенности и соответствия ожидаемому размеру
            def score_detection(det):
                size_score = 1.0 - abs(det['size'] - self.expected_ball_size) / self.expected_ball_size
                return det['confidence'] * 0.7 + size_score * 0.3
            
            best_detection = max(detections, key=score_detection)
            return best_detection
        
        return None
    
    def predict_ball_position_physics(self, frame_shape):
        """Физическое предсказание позиции мяча с учетом гравитации"""
        if len(self.trajectory) < 2:
            return None
        
        # Берем последние позиции
        recent_points = list(self.trajectory)[-3:]
        
        if len(recent_points) < 2:
            return None
        
        # Вычисляем скорость и ускорение
        positions = [p['position'] for p in recent_points]
        
        # Скорость
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        
        # Ускорение (если есть 3 точки)
        if len(positions) >= 3:
            dx2 = positions[-2][0] - positions[-3][0]
            dy2 = positions[-2][1] - positions[-3][1]
            ax = dx - dx2
            ay = dy - dy2 + self.gravity  # Добавляем гравитацию
        else:
            ax = 0
            ay = self.gravity
        
        # Предсказываем следующую позицию
        predicted_x = positions[-1][0] + dx + ax * 0.5
        predicted_y = positions[-1][1] + dy + ay * 0.5
        
        # Проверяем границы
        h, w = frame_shape[:2]
        if 0 <= predicted_x < w and 0 <= predicted_y < h:
            return (predicted_x, predicted_y)
        
        return None
    
    def search_ball_in_predicted_area(self, frame, predicted_pos, search_radius=80):
        """Поиск мяча в области вокруг предсказанной позиции"""
        if predicted_pos is None:
            return None
        
        x, y = predicted_pos
        h, w = frame.shape[:2]
        
        # Определяем область поиска
        x1 = max(0, int(x - search_radius))
        y1 = max(0, int(y - search_radius))
        x2 = min(w, int(x + search_radius))
        y2 = min(h, int(y + search_radius))
        
        # Вырезаем область
        search_region = frame[y1:y2, x1:x2]
        
        if search_region.size == 0:
            return None
        
        # Ищем мяч в области с очень низким порогом
        results = self.model.predict(
            search_region,
            conf=0.1,  # Низкий порог для поиска в области
            iou=0.3,
            classes=[0],
            verbose=False
        )
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Берем лучшую детекцию
            best_idx = np.argmax(confidences)
            box = boxes[best_idx]
            conf = confidences[best_idx]
            
            # Переводим координаты обратно в исходное изображение
            x1_region, y1_region, x2_region, y2_region = box
            center_x = x1 + (x1_region + x2_region) / 2
            center_y = y1 + (y1_region + y2_region) / 2
            size = (x2_region - x1_region) * (y2_region - y1_region)
            
            detection = {
                'box': (x1 + x1_region, y1 + y1_region, x1 + x2_region, y1 + y2_region),
                'center': (center_x, center_y),
                'confidence': conf,
                'size': size,
                'width': x2_region - x1_region,
                'height': y2_region - y1_region
            }
            
            # Валидация найденной детекции
            if self.validate_ball_detection(detection, frame.shape):
                return detection
        
        return None
    
    def track_ball_enhanced(self, frame):
        """Основной метод улучшенного трекинга мяча"""
        h, w = frame.shape[:2]
        
        # 1. Прямая детекция мяча
        detection = self.detect_ball_enhanced(frame)
        
        # 2. Если мяч не найден, пытаемся предсказать позицию
        if detection is None and len(self.trajectory) > 0:
            predicted_pos = self.predict_ball_position_physics(frame.shape)
            if predicted_pos:
                detection = self.search_ball_in_predicted_area(frame, predicted_pos)
        
        # 3. Обновляем статистику и историю
        if detection:
            self.total_detections += 1
            self.missed_frames = 0
            self.consecutive_detections += 1
            
            # Добавляем в историю
            center = detection['center']
            self.trajectory.append({
                'position': center,
                'confidence': detection['confidence'],
                'timestamp': time.time(),
                'size': detection['size']
            })
            
            # Обновляем историю уверенности
            self.confidences.append(detection['confidence'])
            
            # Вычисляем скорость
            if len(self.trajectory) > 1:
                last_pos = self.trajectory[-2]['position']
                dx = center[0] - last_pos[0]
                dy = center[1] - last_pos[1]
                speed = math.sqrt(dx*dx + dy*dy)
                self.velocities.append(speed)
            
            # Обновляем историю детекций
            self.detection_history.append(True)
            
        else:
            self.missed_frames += 1
            self.consecutive_detections = 0
            self.detection_history.append(False)
        
        return detection
    
    def draw_enhanced_tracking(self, frame, detection):
        """Улучшенная отрисовка результатов трекинга"""
        annotated_frame = frame.copy()
        
        # Отрисовываем текущую детекцию
        if detection:
            box = detection['box']
            center = detection['center']
            confidence = detection['confidence']
            
            # Bounding box с градиентом
            x1, y1, x2, y2 = map(int, box)
            
            # Градиентный прямоугольник
            for i in range(3):
                color_intensity = int(255 * (1 - i * 0.3))
                color = (0, color_intensity, 0)
                cv2.rectangle(annotated_frame, 
                            (x1-i, y1-i), (x2+i, y2+i), color, 2)
            
            # Центр мяча с пульсацией
            center_x, center_y = map(int, center)
            pulse = int(8 + 4 * math.sin(time.time() * 10))
            cv2.circle(annotated_frame, (center_x, center_y), pulse, self.colors['ball'], -1)
            cv2.circle(annotated_frame, (center_x, center_y), pulse+3, self.colors['ball'], 2)
            
            # Информация о мяче
            label = f"Ball {confidence:.3f}"
            cv2.putText(annotated_frame, label, (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['ball'], 2)
            
            # Координаты и размер
            cv2.putText(annotated_frame, f"({center_x}, {center_y})", 
                       (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['ball'], 2)
            cv2.putText(annotated_frame, f"Size: {detection['size']:.0f}", 
                       (x1, y2+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['ball'], 2)
        
        # Отрисовываем траекторию с физическими параметрами
        if len(self.trajectory) > 1:
            trajectory_points = []
            for i, point in enumerate(self.trajectory):
                pos = point['position']
                trajectory_points.append((int(pos[0]), int(pos[1])))
            
            # Рисуем траекторию с градиентом и толщиной по скорости
            for i in range(1, len(trajectory_points)):
                # Градиент от старых к новым точкам
                alpha = i / len(trajectory_points)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                
                # Толщина линии зависит от скорости
                thickness = 2
                if i < len(self.velocities):
                    speed_factor = min(self.velocities[i-1] / 50, 2)
                    thickness = int(2 + speed_factor)
                
                cv2.line(annotated_frame, trajectory_points[i-1], trajectory_points[i], 
                        color, thickness)
            
            # Рисуем точки траектории с размером по уверенности
            for i, point in enumerate(trajectory_points):
                alpha = i / len(trajectory_points)
                radius = int(2 + 3 * alpha)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                
                cv2.circle(annotated_frame, point, radius, color, -1)
        
        # Предсказание следующей позиции
        if len(self.trajectory) > 0:
            predicted_pos = self.predict_ball_position_physics(frame.shape)
            if predicted_pos:
                pred_x, pred_y = map(int, predicted_pos)
                cv2.circle(annotated_frame, (pred_x, pred_y), 8, self.colors['prediction'], 2)
                cv2.putText(annotated_frame, "PRED", (pred_x+10, pred_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['prediction'], 2)
                
                # Область поиска
                cv2.circle(annotated_frame, (pred_x, pred_y), 80, self.colors['search_area'], 1)
        
        # Вектор скорости
        if len(self.velocities) > 0 and detection:
            center_x, center_y = map(int, detection['center'])
            speed = self.velocities[-1]
            
            # Направление движения
            if len(self.trajectory) > 1:
                last_pos = self.trajectory[-2]['position']
                dx = center_x - last_pos[0]
                dy = center_y - last_pos[1]
                
                # Нормализуем вектор
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx_norm = dx / length
                    dy_norm = dy / length
                    
                    # Рисуем вектор скорости
                    end_x = int(center_x + dx_norm * speed)
                    end_y = int(center_y + dy_norm * speed)
                    
                    cv2.arrowedLine(annotated_frame, (center_x, center_y), 
                                  (end_x, end_y), self.colors['velocity'], 3)
        
        # Статистика
        cv2.putText(annotated_frame, f"Detections: {self.total_detections}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Missed: {self.missed_frames}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Conf: {self.current_confidence:.2f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Скорость
        if len(self.velocities) > 0:
            avg_speed = np.mean(list(self.velocities)[-5:])
            cv2.putText(annotated_frame, f"Speed: {avg_speed:.1f} px/frame", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated_frame

def test_enhanced_tracking():
    """Тестирование улучшенного трекинга"""
    
    # Путь к модели
    model_path = "runs/segment/train10/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Ошибка: модель не найдена по пути {model_path}")
        return
    
    # Инициализация трекера
    print("🎯 Инициализация улучшенного трекера мяча...")
    tracker = EnhancedBallTracker(model_path)
    
    # Путь к тестовым изображениям
    test_images_path = "dataset/test/images"
    
    if not os.path.exists(test_images_path):
        print(f"❌ Ошибка: папка с тестовыми изображениями не найдена: {test_images_path}")
        return
    
    # Получаем список изображений
    image_files = list(Path(test_images_path).glob("*.jpg"))
    print(f"📸 Найдено {len(image_files)} тестовых изображений")
    
    # Создаем папку для результатов
    output_dir = "enhanced_tracking_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🚀 Начинаем улучшенный трекинг мяча...")
    print("=" * 60)
    
    # Обрабатываем каждое изображение
    for i, image_path in enumerate(image_files):
        print(f"\n📷 Обрабатываем кадр {i+1}/{len(image_files)}: {image_path.name}")
        
        # Загружаем изображение
        frame = cv2.imread(str(image_path))
        if frame is None:
            print("  ❌ Ошибка загрузки изображения")
            continue
        
        # Трекинг мяча
        detection = tracker.track_ball_enhanced(frame)
        
        # Отрисовка результатов
        annotated_frame = tracker.draw_enhanced_tracking(frame, detection)
        
        # Сохраняем результат
        output_path = Path(output_dir) / f"enhanced_{image_path.name}"
        cv2.imwrite(str(output_path), annotated_frame)
        
        # Выводим информацию
        if detection:
            center = detection['center']
            confidence = detection['confidence']
            print(f"  ✅ Мяч обнаружен!")
            print(f"     📍 Позиция: ({center[0]:.1f}, {center[1]:.1f})")
            print(f"     🎯 Уверенность: {confidence:.3f}")
            print(f"     📏 Размер: {detection['size']:.1f}")
            print(f"     🔧 Порог: {tracker.current_confidence:.2f}")
        else:
            print(f"  ❌ Мяч не обнаружен")
            print(f"     🔧 Порог: {tracker.current_confidence:.2f}")
        
        print(f"  📊 Статистика:")
        print(f"     🔍 Всего детекций: {tracker.total_detections}")
        print(f"     ❌ Пропущено кадров: {tracker.missed_frames}")
        print(f"     📈 Траектория: {len(tracker.trajectory)} точек")
        
        if len(tracker.velocities) > 0:
            avg_speed = np.mean(list(tracker.velocities)[-3:])
            print(f"     🏃 Средняя скорость: {avg_speed:.1f} px/frame")
    
    print("\n" + "=" * 60)
    print("🎉 Улучшенный трекинг завершен!")
    print("=" * 60)
    
    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"   🔍 Всего детекций: {tracker.total_detections}")
    print(f"   ❌ Пропущено кадров: {tracker.missed_frames}")
    print(f"   📈 Точки траектории: {len(tracker.trajectory)}")
    print(f"   📁 Результаты сохранены в: {output_dir}")
    
    if tracker.total_detections > 0:
        detection_rate = tracker.total_detections / len(image_files) * 100
        print(f"   🎯 Процент обнаружения: {detection_rate:.1f}%")
    
    # Сохраняем детальную статистику
    stats = {
        'total_detections': tracker.total_detections,
        'missed_frames': tracker.missed_frames,
        'detection_rate': tracker.total_detections / len(image_files) * 100,
        'final_confidence': tracker.current_confidence,
        'trajectory_points': len(tracker.trajectory),
        'avg_speed': np.mean(list(tracker.velocities)) if tracker.velocities else 0
    }
    
    with open('enhanced_tracking_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"   📄 Статистика сохранена в: enhanced_tracking_stats.json")

if __name__ == "__main__":
    test_enhanced_tracking()

