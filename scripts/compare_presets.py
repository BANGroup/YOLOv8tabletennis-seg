#!/usr/bin/env python3
"""
Скрипт для сравнительного тестирования разных пресетов детекции ударов.
Запускает обработку видео с каждым пресетом и сравнивает результаты.

Использование:
    python scripts/compare_presets.py --video GX010025_20sec_raw.mp4 --start 0 --end 20
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List

# Добавляем путь к модулям
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import cv2
import numpy as np

from core import BallTracker, TrackingVisualizer
from core.ball_tracker import BallTrackerConfig
from core.impact_detector import ImpactDetectorConfig
from core.visualizer import VisualizerConfig

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('compare_presets')


def load_config(config_path: str) -> Dict[str, Any]:
    """Загружает конфигурацию из JSON файла."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_preset(config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """Применяет пресет к конфигурации."""
    import copy
    config = copy.deepcopy(config)
    
    presets = config.get('presets', {})
    if preset_name not in presets:
        logger.warning(f"Preset '{preset_name}' not found")
        return config
    
    preset = presets[preset_name]
    
    # Параметры и их секции
    yolo_params = {'confidence', 'model_path', 'ball_class_id', 'table_class_id'}
    tracking_params = {'ball_min_area', 'ball_max_area', 'trajectory_max_len', 
                       'velocity_history_len', 'missed_frames_to_finalize'}
    
    for key, value in preset.items():
        if key.startswith('_'):
            continue
        if key in yolo_params:
            config.setdefault('yolo', {})[key] = value
        elif key in tracking_params:
            config.setdefault('tracking', {})[key] = value
        else:
            config.setdefault('impact_detection', {})[key] = value
    
    return config


def run_with_preset(video_path: str, start_time: int, end_time: int,
                    config: Dict[str, Any], preset_name: str,
                    output_dir: str) -> Dict[str, Any]:
    """
    Запускает обработку видео с указанным пресетом.
    
    Returns:
        Результаты: impacts, detections, timing и т.д.
    """
    preset_config = apply_preset(config, preset_name)
    
    # Создаём папку для пресета
    preset_dir = os.path.join(output_dir, preset_name)
    os.makedirs(preset_dir, exist_ok=True)
    
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return {}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = end_frame - start_frame
    
    # Создаём компоненты
    tracker_config = BallTrackerConfig.from_dict(preset_config)
    impact_config = ImpactDetectorConfig.from_dict(preset_config.get('impact_detection', {}))
    
    tracker = BallTracker(
        config=tracker_config,
        impact_config=impact_config,
        frame_width=frame_width,
        frame_height=frame_height,
        fps=fps,
        output_dir=preset_dir
    )
    
    vis_config = VisualizerConfig()
    visualizer = TrackingVisualizer(vis_config)
    
    # Выходное видео
    output_video = os.path.join(preset_dir, f"tracked_{preset_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    # Переходим к начальному кадру
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Обработка
    processing_start = time.perf_counter()
    detection_frames = []  # Кадры где был мяч
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        fps_time = start_time + (frame_idx / fps)
        detection = tracker.track(frame, frame_idx, fps_time)
        
        if detection:
            detection_frames.append(frame_idx)
        
        # Визуализация
        time_str = f"{int(fps_time // 60):02d}:{int(fps_time % 60):02d}.{int((fps_time % 1) * 100):02d}"
        annotated = visualizer.draw(frame, tracker, detection, time_str)
        out.write(annotated)
    
    # Финализация
    tracker.finalize(total_frames - 1, start_time + (total_frames - 1) / fps)
    
    processing_time = time.perf_counter() - processing_start
    
    cap.release()
    out.release()
    
    # Собираем результаты
    stats = tracker.get_stats()
    impacts = tracker.impact_detector.live_impacts
    
    # Анализ промежутков детекции
    detection_gaps = []
    if len(detection_frames) > 1:
        for i in range(1, len(detection_frames)):
            gap = detection_frames[i] - detection_frames[i-1]
            if gap > 5:  # Промежуток больше 5 кадров
                detection_gaps.append({
                    'from_frame': detection_frames[i-1],
                    'to_frame': detection_frames[i],
                    'gap_frames': gap,
                    'gap_seconds': gap / fps
                })
    
    result = {
        'preset': preset_name,
        'impacts_count': len(impacts),
        'impacts': [
            {
                'frame': imp.frame,
                'time': imp.time,
                'x': imp.x,
                'y': imp.y,
                'curvature': imp.curvature,
                'y_range': imp.y_range
            }
            for imp in impacts
        ],
        'stats': stats,
        'detection_coverage': len(detection_frames) / total_frames * 100,
        'detection_gaps': detection_gaps,
        'processing_time': processing_time,
        'output_video': output_video,
        'parameters': {
            'confidence': preset_config.get('yolo', {}).get('confidence'),
            'ball_min_area': preset_config.get('tracking', {}).get('ball_min_area'),
            'ball_max_area': preset_config.get('tracking', {}).get('ball_max_area'),
            'curvature_threshold': preset_config.get('impact_detection', {}).get('curvature_threshold'),
            'y_range_threshold': preset_config.get('impact_detection', {}).get('y_range_threshold'),
            'min_consecutive_detections': preset_config.get('impact_detection', {}).get('min_consecutive_detections'),
            'min_frame_gap': preset_config.get('impact_detection', {}).get('min_frame_gap'),
            'min_points_for_pattern': preset_config.get('impact_detection', {}).get('min_points_for_pattern')
        }
    }
    
    # Сохраняем результаты
    result_file = os.path.join(preset_dir, f"results_{preset_name}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    return result


def print_comparison(results: List[Dict[str, Any]]) -> None:
    """Выводит сравнительную таблицу результатов."""
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ ПРЕСЕТОВ")
    print("=" * 80)
    
    # Заголовок
    print(f"{'Preset':<15} {'Impacts':<10} {'Detect%':<10} {'Gaps>5f':<10} {'Time(s)':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['preset']:<15} "
              f"{r['impacts_count']:<10} "
              f"{r['detection_coverage']:.1f}%{'':<5} "
              f"{len(r['detection_gaps']):<10} "
              f"{r['processing_time']:.2f}")
    
    print("=" * 80)
    
    # Детали по ударам
    print("\nУДАРЫ ПО ПРЕСЕТАМ:")
    for r in results:
        print(f"\n--- {r['preset']} ({r['impacts_count']} ударов) ---")
        for imp in r['impacts']:
            print(f"  Frame {imp['frame']:4d} ({imp['time']:.2f}s): "
                  f"curv={imp['curvature']:.1f}, y_range={imp['y_range']:.1f}")
    
    # Промежутки в детекции
    print("\n" + "=" * 80)
    print("БОЛЬШИЕ ПРОМЕЖУТКИ В ДЕТЕКЦИИ (>5 кадров):")
    for r in results:
        if r['detection_gaps']:
            print(f"\n--- {r['preset']} ({len(r['detection_gaps'])} промежутков) ---")
            for gap in r['detection_gaps'][:10]:  # Показываем только первые 10
                print(f"  Frames {gap['from_frame']}-{gap['to_frame']}: "
                      f"{gap['gap_frames']} кадров ({gap['gap_seconds']:.2f}s)")
    
    # Рекомендация
    print("\n" + "=" * 80)
    best = max(results, key=lambda x: x['impacts_count'])
    print(f"РЕКОМЕНДАЦИЯ: Пресет '{best['preset']}' показал лучший результат: {best['impacts_count']} ударов")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Сравнение пресетов детекции ударов')
    parser.add_argument('--video', type=str, required=True, help='Путь к видео')
    parser.add_argument('--start', type=int, default=0, help='Начало сегмента (сек)')
    parser.add_argument('--end', type=int, default=20, help='Конец сегмента (сек)')
    parser.add_argument('--config', type=str, default='scripts/config/detection_params_v3.json',
                        help='Путь к конфигу')
    parser.add_argument('--output', type=str, default='results/comparison',
                        help='Папка для результатов')
    parser.add_argument('--presets', type=str, nargs='+', 
                        default=['sensitive', 'ultra', 'yolo_boost'],
                        help='Список пресетов для тестирования')
    
    args = parser.parse_args()
    
    # Проверяем видео
    if not os.path.exists(args.video):
        logger.error(f"Video not found: {args.video}")
        sys.exit(1)
    
    # Загружаем конфиг
    config = load_config(args.config)
    
    # Создаём папку результатов с timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Comparing presets: {args.presets}")
    logger.info(f"Video: {args.video}")
    logger.info(f"Segment: {args.start}s - {args.end}s")
    logger.info(f"Output: {output_dir}")
    
    results = []
    
    for preset in args.presets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing preset: {preset}")
        logger.info(f"{'='*60}")
        
        result = run_with_preset(
            video_path=args.video,
            start_time=args.start,
            end_time=args.end,
            config=config,
            preset_name=preset,
            output_dir=output_dir
        )
        
        if result:
            results.append(result)
            logger.info(f"Preset {preset}: {result['impacts_count']} impacts, "
                       f"{result['detection_coverage']:.1f}% detection coverage")
    
    # Сравнение результатов
    print_comparison(results)
    
    # Сохраняем общий отчёт
    report_file = os.path.join(output_dir, 'comparison_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'video': args.video,
            'segment': {'start': args.start, 'end': args.end},
            'results': results
        }, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\nReport saved: {report_file}")


if __name__ == '__main__':
    main()
