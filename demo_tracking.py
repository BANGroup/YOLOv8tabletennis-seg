#!/usr/bin/env python3
"""
Демонстрация профессионального трекинга мяча
"""

import cv2
import numpy as np
from professional_ball_tracker import TableTennisTracker
import os
from pathlib import Path
import time

def demo_tracking():
    """Демонстрация трекинга с подробным выводом"""
    
    # Путь к обученной модели
    model_path = "runs/segment/train10/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Ошибка: модель не найдена по пути {model_path}")
        return
    
    # Конфигурация стола
    table_config = {
        'bounds': {
            'x_min': 100, 'x_max': 500,
            'y_min': 100, 'y_max': 400
        },
        'net_position': {
            'x': 300,
            'y_min': 100,
            'y_max': 400
        }
    }
    
    # Инициализация трекера
    print("🏓 Инициализация профессионального трекера настольного тенниса...")
    tracker = TableTennisTracker(model_path, table_config)
    
    # Путь к тестовым изображениям
    test_images_path = "dataset/test/images"
    
    if not os.path.exists(test_images_path):
        print(f"Ошибка: папка с тестовыми изображениями не найдена: {test_images_path}")
        return
    
    # Получаем список изображений
    image_files = list(Path(test_images_path).glob("*.jpg"))
    print(f"📸 Найдено {len(image_files)} тестовых изображений")
    
    # Создаем папку для результатов
    output_dir = "demo_tracking_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🎯 Начинаем профессиональный трекинг...")
    print("=" * 60)
    
    # Обрабатываем каждое изображение
    for i, image_path in enumerate(image_files):
        print(f"\n📷 Обрабатываем кадр {i+1}/{len(image_files)}: {image_path.name}")
        
        # Загружаем изображение
        frame = cv2.imread(str(image_path))
        if frame is None:
            print("  ❌ Ошибка загрузки изображения")
            continue
        
        # Обрабатываем кадр
        annotated_frame, web_data = tracker.process_frame(frame, i)
        
        # Сохраняем результат
        output_path = Path(output_dir) / f"demo_{image_path.name}"
        cv2.imwrite(str(output_path), annotated_frame)
        
        # Выводим подробную информацию
        print(f"  📊 Статистика кадра:")
        print(f"     🏆 Счет: Игрок 1: {web_data['game_state']['player1_score']} | Игрок 2: {web_data['game_state']['player2_score']}")
        print(f"     🎾 Ударов: {web_data['statistics']['total_hits']}")
        print(f"     🔄 Ралли: {web_data['statistics']['rally_count']}")
        print(f"     📈 Макс. ралли: {web_data['statistics']['max_rally_length']}")
        print(f"     🎮 Статус игры: {web_data['game_state']['game_status']}")
        print(f"     ⚡ Мяч в игре: {'Да' if web_data['game_state']['ball_in_play'] else 'Нет'}")
        
        if web_data['ball_position']:
            ball_pos = web_data['ball_position']
            print(f"  🎯 Позиция мяча:")
            print(f"     📍 X: {ball_pos['x']:.1f}px")
            print(f"     📍 Y: {ball_pos['y']:.1f}px")
            print(f"     🎯 Уверенность: {ball_pos['confidence']:.2f}")
        else:
            print(f"  ❌ Мяч не обнаружен")
        
        if web_data['hit_event']:
            hit = web_data['hit_event']
            print(f"  💥 УДАР ОБНАРУЖЕН!")
            print(f"     👤 Игрок: {hit['player']}")
            print(f"     📍 Позиция: ({hit['position'][0]:.1f}, {hit['position'][1]:.1f})")
            print(f"     🚀 Скорость: {hit['speed']:.1f} px/s")
            print(f"     🧭 Направление: {hit['direction']:.1f}°")
        
        # Показываем траекторию
        if len(tracker.trajectory_analyzer.trajectory) > 0:
            print(f"  📈 Траектория: {len(tracker.trajectory_analyzer.trajectory)} точек")
            if len(tracker.trajectory_analyzer.velocity_history) > 0:
                avg_speed = np.mean(list(tracker.trajectory_analyzer.velocity_history))
                print(f"     🚀 Средняя скорость: {avg_speed:.1f} px/s")
        
        print(f"  💾 Результат сохранен: {output_path}")
        
        # Небольшая пауза для демонстрации
        time.sleep(0.2)
    
    print("\n" + "=" * 60)
    print("🎉 Профессиональный трекинг завершен!")
    print("=" * 60)
    
    # Итоговая статистика
    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"   🏆 Финальный счет: Игрок 1: {tracker.game_state.player1_score} | Игрок 2: {tracker.game_state.player2_score}")
    print(f"   🎾 Всего ударов: {tracker.total_hits}")
    print(f"   🔄 Текущее ралли: {tracker.rally_count}")
    print(f"   📈 Максимальное ралли: {tracker.max_rally_length}")
    print(f"   📝 События ударов: {len(tracker.hit_events)}")
    print(f"   📁 Результаты сохранены в: {output_dir}")
    
    # Анализ траектории
    if len(tracker.trajectory_analyzer.trajectory) > 0:
        print(f"\n📈 АНАЛИЗ ТРАЕКТОРИИ:")
        print(f"   📍 Всего точек траектории: {len(tracker.trajectory_analyzer.trajectory)}")
        
        if len(tracker.trajectory_analyzer.velocity_history) > 0:
            velocities = list(tracker.trajectory_analyzer.velocity_history)
            print(f"   🚀 Скорость мяча:")
            print(f"      📊 Средняя: {np.mean(velocities):.1f} px/s")
            print(f"      📊 Максимальная: {np.max(velocities):.1f} px/s")
            print(f"      📊 Минимальная: {np.min(velocities):.1f} px/s")
        
        if len(tracker.trajectory_analyzer.acceleration_history) > 0:
            accelerations = list(tracker.trajectory_analyzer.acceleration_history)
            print(f"   ⚡ Ускорение мяча:")
            print(f"      📊 Среднее: {np.mean(accelerations):.1f} px/s²")
            print(f"      📊 Максимальное: {np.max(accelerations):.1f} px/s²")
            print(f"      📊 Минимальное: {np.min(accelerations):.1f} px/s²")
    
    # Детали ударов
    if tracker.hit_events:
        print(f"\n💥 ДЕТАЛИ УДАРОВ:")
        for i, hit in enumerate(tracker.hit_events):
            print(f"   Удар {i+1}:")
            print(f"      👤 Игрок: {hit.player}")
            print(f"      📍 Позиция: ({hit.position[0]:.1f}, {hit.position[1]:.1f})")
            print(f"      🚀 Скорость: {hit.ball_speed:.1f} px/s")
            print(f"      🧭 Направление: {hit.ball_direction:.1f}°")
    
    print(f"\n🌐 Веб-интерфейс доступен по адресу: http://localhost:5000")
    print(f"📁 Детальная статистика сохранена в: professional_game_statistics.json")

if __name__ == "__main__":
    demo_tracking()




