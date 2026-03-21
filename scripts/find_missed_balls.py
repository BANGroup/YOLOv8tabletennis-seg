#!/usr/bin/env python3
"""
Агент: coder-tracking
Задача: найти кадры где модель теряет мяч для дообучения

Принцип: если Kalman предсказывает позицию, но YOLO не видит мяч —
это кандидат для доразметки.
"""

import cv2
import json
import os
import argparse
from pathlib import Path
from ultralytics import YOLO


def find_missed_detections(video_path: str, model_path: str, output_dir: str,
                           start_sec: float = 0, end_sec: float = None,
                           confidence: float = 0.03):
    """
    Находит кадры где модель НЕ видит мяч, но он вероятно есть.
    Сохраняет кадры для последующей разметки.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames_to_annotate')
    os.makedirs(frames_dir, exist_ok=True)
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec else total_frames
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Статистика
    stats = {
        'total_frames': 0,
        'detected': 0,
        'missed': 0,
        'missed_frames': [],
        'detection_by_zone': {'left': 0, 'center': 0, 'right': 0}
    }
    
    # Трекинг для определения "пропусков"
    last_detection = None
    consecutive_misses = 0
    frame_idx = start_frame
    
    print(f"Анализирую видео: {video_path}")
    print(f"Кадры: {start_frame} - {end_frame}")
    print(f"Модель: {model_path}")
    print()
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        stats['total_frames'] += 1
        
        # Детекция
        results = model(frame, conf=confidence, verbose=False)[0]
        
        ball_detected = False
        ball_x = None
        
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # Ball
                ball_detected = True
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                ball_x = (x1 + x2) / 2 / frame.shape[1]  # normalized
                
                # Зона
                if ball_x < 0.4:
                    stats['detection_by_zone']['left'] += 1
                elif ball_x < 0.6:
                    stats['detection_by_zone']['center'] += 1
                else:
                    stats['detection_by_zone']['right'] += 1
                break
        
        if ball_detected:
            stats['detected'] += 1
            last_detection = frame_idx
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            
            # Если пропуск внутри "серии детекций" — вероятно мяч есть
            # но модель его не видит
            if last_detection and consecutive_misses <= 5:
                stats['missed'] += 1
                stats['missed_frames'].append({
                    'frame': frame_idx,
                    'time': frame_idx / fps,
                    'consecutive_miss': consecutive_misses
                })
                
                # Сохраняем кадр для разметки
                frame_path = os.path.join(frames_dir, f'frame_{frame_idx:06d}.jpg')
                cv2.imwrite(frame_path, frame)
        
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"  Обработано: {frame_idx - start_frame}/{end_frame - start_frame}")
    
    cap.release()
    
    # Сохраняем отчёт
    report = {
        'video': video_path,
        'model': model_path,
        'stats': stats,
        'recommendation': ''
    }
    
    detection_rate = stats['detected'] / stats['total_frames'] * 100 if stats['total_frames'] else 0
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("="*60)
    print(f"Всего кадров: {stats['total_frames']}")
    print(f"Мяч обнаружен: {stats['detected']} ({detection_rate:.1f}%)")
    print(f"Пропущено (вероятно есть мяч): {stats['missed']}")
    print()
    print("Распределение детекций по зонам:")
    total_det = sum(stats['detection_by_zone'].values())
    for zone, count in stats['detection_by_zone'].items():
        pct = count / total_det * 100 if total_det else 0
        print(f"  {zone}: {count} ({pct:.1f}%)")
    
    print()
    print(f"Кадры для доразметки сохранены в: {frames_dir}")
    print(f"Всего кадров для разметки: {len(os.listdir(frames_dir))}")
    
    # Рекомендация
    zones = stats['detection_by_zone']
    if zones['right'] < zones['left'] * 0.5:
        report['recommendation'] = "РЕКОМЕНДАЦИЯ: Добавить больше примеров мяча в ПРАВОЙ зоне"
    elif zones['left'] < zones['right'] * 0.5:
        report['recommendation'] = "РЕКОМЕНДАЦИЯ: Добавить больше примеров мяча в ЛЕВОЙ зоне"
    else:
        report['recommendation'] = "Распределение сбалансировано"
    
    print(f"\n{report['recommendation']}")
    
    # Сохраняем JSON
    report_path = os.path.join(output_dir, 'missed_detection_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nОтчёт сохранён: {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Найти кадры где модель пропускает мяч')
    parser.add_argument('--video', required=True, help='Путь к видео')
    parser.add_argument('--model', default='runs/segment/train10/weights/best.pt')
    parser.add_argument('--output', default='results/missed_detections')
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float, default=None)
    parser.add_argument('--conf', type=float, default=0.03)
    
    args = parser.parse_args()
    
    find_missed_detections(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output,
        start_sec=args.start,
        end_sec=args.end,
        confidence=args.conf
    )


if __name__ == '__main__':
    main()
