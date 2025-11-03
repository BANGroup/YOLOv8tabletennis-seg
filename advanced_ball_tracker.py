#!/usr/bin/env python3
"""
Продвинутый трекинг мяча с улучшенной детекцией и траекторией
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import time
from collections import deque
import math

class AdvancedBallTracker:
    """Продвинутый трекер мяча с улучшенной детекцией"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Настройки детекции мяча
        self.ball_confidence_threshold = 0.1  # Очень низкий порог
        self.ball_iou_threshold = 0.3
        
        # Траектория мяча
        self.trajectory = deque(maxlen=100)
        self.ball_positions = deque(maxlen=50)
        
        # Цвета
        self.ball_color = (0, 255, 0)  # Зеленый
        self.trajectory_color = (255, 0, 255)  # Пурпурный
        self.prediction_color = (0, 255, 255)  # Желтый
        
        # Статистика
        self.total_detections = 0
        self.missed_frames = 0
        
    def detect_ball_advanced(self, frame):
        """Улучшенная детекция мяча"""
        # Выполняем детекцию с очень низким порогом
        results = self.model.predict(
            frame,
            conf=self.ball_confidence_threshold,
            iou=self.ball_iou_threshold,
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
                    
                    detections.append({
                        'box': box,
                        'center': (center_x, center_y),
                        'confidence': conf,
                        'size': (x2 - x1) * (y2 - y1)
                    })
        
        # Если найдено несколько детекций, берем лучшую
        if detections:
            # Сортируем по уверенности и размеру
            best_detection = max(detections, key=lambda x: x['confidence'] * x['size'])
            return best_detection
        
        return None
    
    def predict_ball_position(self, frame_shape):
        """Предсказание позиции мяча на основе траектории"""
        if len(self.ball_positions) < 2:
            return None
        
        # Берем последние 3 позиции
        recent_positions = list(self.ball_positions)[-3:]
        
        if len(recent_positions) < 2:
            return None
        
        # Вычисляем скорость
        dx = recent_positions[-1][0] - recent_positions[-2][0]
        dy = recent_positions[-1][1] - recent_positions[-2][1]
        
        # Предсказываем следующую позицию
        predicted_x = recent_positions[-1][0] + dx
        predicted_y = recent_positions[-1][1] + dy
        
        # Проверяем границы
        if 0 <= predicted_x < frame_shape[1] and 0 <= predicted_y < frame_shape[0]:
            return (predicted_x, predicted_y)
        
        return None
    
    def search_ball_in_region(self, frame, center, search_radius=50):
        """Поиск мяча в области вокруг предсказанной позиции"""
        if center is None:
            return None
        
        x, y = center
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
        
        # Ищем мяч в области
        results = self.model.predict(
            search_region,
            conf=0.05,  # Очень низкий порог
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
            
            return {
                'box': (x1 + x1_region, y1 + y1_region, x1 + x2_region, y1 + y2_region),
                'center': (center_x, center_y),
                'confidence': conf,
                'size': (x2_region - x1_region) * (y2_region - y1_region)
            }
        
        return None
    
    def track_ball(self, frame):
        """Основной метод трекинга мяча"""
        h, w = frame.shape[:2]
        
        # 1. Прямая детекция мяча
        detection = self.detect_ball_advanced(frame)
        
        # 2. Если мяч не найден, пытаемся предсказать позицию
        if detection is None and len(self.ball_positions) > 0:
            predicted_pos = self.predict_ball_position(frame.shape)
            if predicted_pos:
                detection = self.search_ball_in_region(frame, predicted_pos)
        
        # 3. Если все еще не найден, ищем по всему изображению с очень низким порогом
        if detection is None:
            # Разбиваем изображение на области и ищем в каждой
            regions = [
                (0, 0, w//2, h//2),           # Верхний левый
                (w//2, 0, w, h//2),           # Верхний правый
                (0, h//2, w//2, h),           # Нижний левый
                (w//2, h//2, w, h),           # Нижний правый
            ]
            
            for x1, y1, x2, y2 in regions:
                region = frame[y1:y2, x1:x2]
                if region.size == 0:
                    continue
                
                results = self.model.predict(
                    region,
                    conf=0.01,  # Экстремально низкий порог
                    iou=0.1,
                    classes=[0],
                    verbose=False
                )
                
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    best_idx = np.argmax(confidences)
                    box = boxes[best_idx]
                    conf = confidences[best_idx]
                    
                    # Переводим координаты
                    x1_region, y1_region, x2_region, y2_region = box
                    center_x = x1 + (x1_region + x2_region) / 2
                    center_y = y1 + (y1_region + y2_region) / 2
                    
                    detection = {
                        'box': (x1 + x1_region, y1 + y1_region, x1 + x2_region, y1 + y2_region),
                        'center': (center_x, center_y),
                        'confidence': conf,
                        'size': (x2_region - x1_region) * (y2_region - y1_region)
                    }
                    break
        
        # Обновляем статистику
        if detection:
            self.total_detections += 1
            self.missed_frames = 0
            
            # Добавляем в траекторию
            center = detection['center']
            self.ball_positions.append(center)
            self.trajectory.append({
                'position': center,
                'confidence': detection['confidence'],
                'timestamp': time.time()
            })
        else:
            self.missed_frames += 1
        
        return detection
    
    def draw_tracking(self, frame, detection):
        """Отрисовка результатов трекинга"""
        annotated_frame = frame.copy()
        
        # Отрисовываем текущую детекцию
        if detection:
            box = detection['box']
            center = detection['center']
            confidence = detection['confidence']
            
            # Bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.ball_color, 3)
            
            # Центр мяча
            center_x, center_y = map(int, center)
            cv2.circle(annotated_frame, (center_x, center_y), 8, self.ball_color, -1)
            cv2.circle(annotated_frame, (center_x, center_y), 12, self.ball_color, 2)
            
            # Информация о мяче
            label = f"Ball {confidence:.3f}"
            cv2.putText(annotated_frame, label, (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ball_color, 2)
            
            # Координаты центра
            cv2.putText(annotated_frame, f"({center_x}, {center_y})", 
                       (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ball_color, 2)
        
        # Отрисовываем траекторию
        if len(self.trajectory) > 1:
            trajectory_points = []
            for i, point in enumerate(self.trajectory):
                pos = point['position']
                trajectory_points.append((int(pos[0]), int(pos[1])))
            
            # Рисуем траекторию с градиентом
            for i in range(1, len(trajectory_points)):
                # Градиент от старых к новым точкам
                alpha = i / len(trajectory_points)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                
                cv2.line(annotated_frame, trajectory_points[i-1], trajectory_points[i], color, 3)
            
            # Рисуем точки траектории
            for i, point in enumerate(trajectory_points):
                alpha = i / len(trajectory_points)
                radius = int(3 + 2 * alpha)
                color_intensity = int(255 * alpha)
                color = (color_intensity, 0, 255 - color_intensity)
                
                cv2.circle(annotated_frame, point, radius, color, -1)
        
        # Предсказание следующей позиции
        if len(self.ball_positions) > 0:
            predicted_pos = self.predict_ball_position(frame.shape)
            if predicted_pos:
                pred_x, pred_y = map(int, predicted_pos)
                cv2.circle(annotated_frame, (pred_x, pred_y), 6, self.prediction_color, 2)
                cv2.putText(annotated_frame, "PRED", (pred_x+10, pred_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.prediction_color, 1)
        
        # Статистика
        cv2.putText(annotated_frame, f"Detections: {self.total_detections}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Missed: {self.missed_frames}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Trajectory: {len(self.trajectory)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Информация о скорости
        if len(self.ball_positions) > 1:
            recent_positions = list(self.ball_positions)[-2:]
            dx = recent_positions[-1][0] - recent_positions[-2][0]
            dy = recent_positions[-1][1] - recent_positions[-2][1]
            speed = math.sqrt(dx*dx + dy*dy)
            cv2.putText(annotated_frame, f"Speed: {speed:.1f} px/frame", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated_frame

def test_advanced_tracking():
    """Тестирование продвинутого трекинга"""
    
    # Путь к модели
    model_path = "runs/segment/train10/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Ошибка: модель не найдена по пути {model_path}")
        return
    
    # Инициализация трекера
    print("🎯 Инициализация продвинутого трекера мяча...")
    tracker = AdvancedBallTracker(model_path)
    
    # Путь к тестовым изображениям
    test_images_path = "dataset/test/images"
    
    if not os.path.exists(test_images_path):
        print(f"Ошибка: папка с тестовыми изображениями не найдена: {test_images_path}")
        return
    
    # Получаем список изображений
    image_files = list(Path(test_images_path).glob("*.jpg"))
    print(f"📸 Найдено {len(image_files)} тестовых изображений")
    
    # Создаем папку для результатов
    output_dir = "advanced_tracking_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🚀 Начинаем продвинутый трекинг мяча...")
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
        detection = tracker.track_ball(frame)
        
        # Отрисовка результатов
        annotated_frame = tracker.draw_tracking(frame, detection)
        
        # Сохраняем результат
        output_path = Path(output_dir) / f"advanced_{image_path.name}"
        cv2.imwrite(str(output_path), annotated_frame)
        
        # Выводим информацию
        if detection:
            center = detection['center']
            confidence = detection['confidence']
            print(f"  ✅ Мяч обнаружен!")
            print(f"     📍 Позиция: ({center[0]:.1f}, {center[1]:.1f})")
            print(f"     🎯 Уверенность: {confidence:.3f}")
            print(f"     📏 Размер: {detection['size']:.1f}")
        else:
            print(f"  ❌ Мяч не обнаружен")
        
        print(f"  📊 Статистика:")
        print(f"     🔍 Всего детекций: {tracker.total_detections}")
        print(f"     ❌ Пропущено кадров: {tracker.missed_frames}")
        print(f"     📈 Траектория: {len(tracker.trajectory)} точек")
        
        # Показываем результат
        cv2.imshow('Advanced Ball Tracking', annotated_frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'):  # Показываем каждое изображение 1 секунду
            break
    
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("🎉 Продвинутый трекинг завершен!")
    print("=" * 60)
    
    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"   🔍 Всего детекций: {tracker.total_detections}")
    print(f"   ❌ Пропущено кадров: {tracker.missed_frames}")
    print(f"   📈 Точки траектории: {len(tracker.trajectory)}")
    print(f"   📁 Результаты сохранены в: {output_dir}")
    
    if tracker.total_detections > 0:
        detection_rate = tracker.total_detections / len(image_files) * 100
        print(f"   🎯 Процент обнаружения: {detection_rate:.1f}%")
    
    # Сохраняем траекторию
    if tracker.trajectory:
        trajectory_data = []
        for point in tracker.trajectory:
            trajectory_data.append({
                'x': point['position'][0],
                'y': point['position'][1],
                'confidence': point['confidence'],
                'timestamp': point['timestamp']
            })
        
        import json
        with open('ball_trajectory.json', 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
        
        print(f"   📄 Траектория сохранена в: ball_trajectory.json")

if __name__ == "__main__":
    test_advanced_tracking()




