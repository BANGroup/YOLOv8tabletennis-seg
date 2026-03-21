#!/usr/bin/env python3
"""
СТАБИЛЬНАЯ ВЕРСИЯ - НЕ ИЗМЕНЯТЬ!

Трекинг мяча v3 - модульная архитектура.
Эта версия достигла цели: 14 ударов на 20-секундном видео.

Оптимальный пресет: balanced
Дата создания: 2026-01-26
Результаты:
  - Frames: 1198
  - Detections: 843 (70% покрытие)  
  - IMPACTS: 14 (цель достигнута!)
  - Effective FPS: 77.71

Изменения относительно v1/v2:
1. Модульная структура (core/)
2. Калман фильтр с гравитацией
3. Все параметры в конфиге (без hardcode)
4. Fallback для полигона стола
5. Логирование вместо print
6. Поддержка пресетов (sensitive/strict/balanced)

Использование:
    python main_processor_v3_stable.py --video VIDEO --start SEC --end SEC --preset balanced
"""

import cv2
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from core import BallTracker
from core.ball_tracker import BallTrackerConfig
from core.impact_detector import ImpactDetectorConfig
from core.visualizer_clean import CleanVisualizer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('main_processor_v3_stable')


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Загружает конфигурацию из JSON файла."""
    if not os.path.exists(config_path):
        logger.error(f"Config not found: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    logger.info(f"Config loaded: {config_path}")
    return config


def apply_preset(config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """
    Применяет пресет к конфигурации.
    
    Args:
        config: Базовая конфигурация
        preset_name: Имя пресета ('sensitive', 'strict', 'ultra', 'yolo_boost', 'balanced')
        
    Returns:
        Обновлённая конфигурация
    """
    presets = config.get('presets', {})
    
    if preset_name not in presets:
        logger.warning(f"Preset '{preset_name}' not found, using defaults")
        return config
    
    preset = presets[preset_name]
    logger.info(f"Applying preset: {preset_name}")
    
    # Параметры и их целевые секции
    yolo_params = {'confidence', 'model_path', 'ball_class_id', 'table_class_id'}
    tracking_params = {'ball_min_area', 'ball_max_area', 'trajectory_max_len', 
                       'velocity_history_len', 'missed_frames_to_finalize'}
    
    for key, value in preset.items():
        if key.startswith('_'):
            continue
            
        # Определяем целевую секцию
        if key in yolo_params:
            if 'yolo' in config:
                config['yolo'][key] = value
                logger.info(f"  yolo.{key} = {value}")
        elif key in tracking_params:
            if 'tracking' in config:
                config['tracking'][key] = value
                logger.info(f"  tracking.{key} = {value}")
        else:
            # По умолчанию - impact_detection
            if 'impact_detection' in config:
                config['impact_detection'][key] = value
                logger.info(f"  impact.{key} = {value}")
    
    return config


def format_time(seconds: float) -> str:
    """Форматирует время в MM:SS.ms"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 100)
    return f"{minutes:02d}:{secs:02d}.{ms:02d}"


def process_video(video_path: str, start_time: int, end_time: int,
                  config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Обрабатывает видео сегмент.
    
    Args:
        video_path: Путь к видео
        start_time: Начало в секундах
        end_time: Конец в секундах
        config: Конфигурация
        output_dir: Папка для результатов
        
    Returns:
        Результаты обработки
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return {}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {frame_width}x{frame_height} @ {fps:.2f} FPS")
    logger.info(f"Segment: {start_time}s - {end_time}s")
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = end_frame - start_frame
    
    # Создаём конфиги компонентов
    tracker_config = BallTrackerConfig.from_dict(config)
    impact_config = ImpactDetectorConfig.from_dict(config.get('impact_detection', {}))
    
    # Создаём трекер
    tracker = BallTracker(
        config=tracker_config,
        impact_config=impact_config,
        frame_width=frame_width,
        frame_height=frame_height,
        fps=fps,
        output_dir=output_dir
    )
    
    # Создаём чистый визуализатор (как в оригинальном main_processor.py)
    visualizer = CleanVisualizer()
    
    # Выходное видео
    output_video = os.path.join(output_dir, "tracked_v3_stable.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    # Переходим к начальному кадру
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Обработка
    processed_count = 0
    processing_start = time.perf_counter()
    frame_times = []
    
    logger.info("Processing started...")
    
    for frame_idx in range(total_frames):
        frame_start = time.perf_counter()
        
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame {frame_idx}")
            break
        
        # Время в секундах от начала видео
        fps_time = start_time + (frame_idx / fps)
        
        # Трекинг
        detection = tracker.track(frame, frame_idx, fps_time)
        
        # Время для отображения
        time_str = format_time(fps_time)
        
        # Визуализация
        annotated = visualizer.draw(frame, tracker, detection, time_str)
        
        out.write(annotated)
        processed_count += 1
        
        frame_time = time.perf_counter() - frame_start
        frame_times.append(frame_time)
        
        # Прогресс каждые 30 кадров
        if processed_count % 30 == 0 or processed_count == 1:
            elapsed = time.perf_counter() - processing_start
            effective_fps = processed_count / elapsed if elapsed > 0 else 0
            progress = (processed_count / total_frames) * 100
            impacts = len(tracker.impact_detector.live_impacts)
            logger.info(f"[{progress:5.1f}%] Frame {processed_count}/{total_frames} | "
                       f"Impacts: {impacts} | FPS: {effective_fps:.1f}")
    
    # Завершаем
    final_fps_time = start_time + (processed_count / fps)
    tracker.finalize(processed_count, final_fps_time)
    
    cap.release()
    out.release()
    
    processing_total = time.perf_counter() - processing_start
    
    # Статистика
    stats = tracker.get_stats()
    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
    effective_fps = processed_count / processing_total if processing_total > 0 else 0
    
    logger.info("=" * 50)
    logger.info("RESULTS v3 STABLE:")
    logger.info(f"  Frames processed: {processed_count}")
    logger.info(f"  Detections: {stats['total_detections']}")
    logger.info(f"  Missed: {stats['missed_frames']}")
    logger.info(f"  IMPACTS: {stats['impact_count']}")
    logger.info(f"  Kalman predictions: {stats['kalman_predictions']}")
    logger.info(f"  Processing time: {processing_total:.2f}s")
    logger.info(f"  Effective FPS: {effective_fps:.2f}")
    logger.info(f"  Output: {output_video}")
    logger.info("=" * 50)
    
    # Формируем результат
    result = {
        'version': 'v3_stable',
        'video': video_path,
        'segment': {'start': start_time, 'end': end_time},
        'stats': stats,
        'impacts': tracker.impact_detector.get_impacts_json(),
        'parameters': {
            'tracker': {
                'model_path': tracker_config.model_path,
                'confidence': tracker_config.confidence,
                'kalman_gravity': tracker_config.kalman_gravity
            },
            'impact_detection': {
                'min_consecutive_detections': impact_config.min_consecutive_detections,
                'min_frame_gap': impact_config.min_frame_gap,
                'curvature_threshold': impact_config.curvature_threshold,
                'y_range_threshold': impact_config.y_range_threshold
            }
        },
        'timing': {
            'total_seconds': processing_total,
            'avg_ms_per_frame': avg_frame_time * 1000,
            'effective_fps': effective_fps,
            'processed_frames': processed_count
        }
    }
    
    # Сохраняем JSON
    json_path = os.path.join(output_dir, "analysis_v3_stable.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Analysis saved: {json_path}")
    
    # Сохраняем live impacts
    impacts_path = os.path.join(output_dir, "live_impacts_v3_stable.json")
    with open(impacts_path, 'w', encoding='utf-8') as f:
        json.dump(tracker.impact_detector.get_impacts_json(), f, ensure_ascii=False, indent=2)
    logger.info(f"Live impacts saved: {impacts_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Трекинг мяча v3 STABLE - эталонная версия с 14 ударами',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
СТАБИЛЬНАЯ ВЕРСИЯ - ЦЕЛЬ ДОСТИГНУТА: 14 ударов

Примеры:
  # Запуск с оптимальным пресетом balanced
  python main_processor_v3_stable.py --video GX010025_20sec_raw.mp4 --start 0 --end 20 --preset balanced

  # С чувствительным пресетом (больше ударов)
  python main_processor_v3_stable.py --video sources/match.mp4 --start 395 --end 415 --preset sensitive
        """
    )
    
    parser.add_argument('--video', required=True, help='Путь к видео файлу')
    parser.add_argument('--start', type=int, required=True, help='Начало сегмента (секунды)')
    parser.add_argument('--end', type=int, required=True, help='Конец сегмента (секунды)')
    parser.add_argument('--config', default='scripts/config/detection_params_v3.json',
                       help='Путь к конфигу (default: scripts/config/detection_params_v3.json)')
    parser.add_argument('--preset', choices=['sensitive', 'strict', 'ultra', 'yolo_boost', 'yolo_extreme', 'balanced'],
                       default='balanced',
                       help='Применить пресет настроек (default: balanced)')
    parser.add_argument('--output', help='Папка для результатов (default: из конфига)')
    parser.add_argument('--debug', action='store_true', help='Включить отладочный вывод')
    
    args = parser.parse_args()
    
    # Уровень логирования
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Загружаем конфиг
    config = load_config(args.config)
    if not config:
        logger.error("Failed to load config, exiting")
        return 1
    
    # Применяем пресет если указан
    if args.preset:
        config = apply_preset(config, args.preset)
    
    # Проверяем файлы
    if not os.path.exists(args.video):
        logger.error(f"Video not found: {args.video}")
        return 1
    
    model_path = config.get('yolo', {}).get('model_path', '')
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return 1
    
    # Папка для результатов
    output_dir = args.output or config.get('output', {}).get('output_dir', 'results/v3')
    
    # Запускаем обработку
    try:
        result = process_video(
            video_path=args.video,
            start_time=args.start,
            end_time=args.end,
            config=config,
            output_dir=output_dir
        )
        
        if result:
            logger.info(f"Processing completed. Impacts: {result['stats']['impact_count']}")
            return 0
        else:
            logger.error("Processing failed")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
