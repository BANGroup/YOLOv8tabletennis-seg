#!/usr/bin/env python3
"""
Сравнение вариантов детекции подачи
v1 - по паузе (>3 сек)
v2 - по позиции первого удара
v3 - по подбросу мяча  
v4 - комбинированный (все три метода)
"""

import subprocess
import sys
import os
import json
from datetime import datetime

# Варианты скриптов
VARIANTS = [
    ("v1_pause", "serve_detector_v1_pause.py", "По паузе >3с"),
    ("v2_position", "serve_detector_v2_position.py", "По позиции первого удара"),
    ("v3_toss", "serve_detector_v3_toss.py", "По подбросу мяча"),
    ("v4_combined", "serve_detector_v4_combined.py", "Комбинированный"),
]

def run_variant(script_name, video_path, start, end):
    """Запуск одного варианта и получение результатов"""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    cmd = [
        sys.executable, script_path,
        "--video", video_path,
        "--start", str(start),
        "--end", str(end),
        "--output", f"results/serve_test/{script_name.replace('.py', '')}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

def parse_results(output):
    """Парсинг результатов из вывода скрипта"""
    results = {
        "impacts": 0,
        "serves": 0,
        "serve_times": []
    }
    
    for line in output.split('\n'):
        if "SMART УДАРОВ об СТОЛ:" in line:
            try:
                results["impacts"] = int(line.split(':')[1].strip())
            except:
                pass
        elif "ПОДАЧ" in line and ":" in line:
            try:
                results["serves"] = int(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif "Подача" in line and "t=" in line:
            try:
                time_str = line.split('t=')[1].split('s')[0]
                results["serve_times"].append(float(time_str))
            except:
                pass
    
    return results

def main():
    if len(sys.argv) < 4:
        print("Использование: python compare_serve_detectors.py <video> <start> <end>")
        print("Пример: python compare_serve_detectors.py GX010025_20sec_raw.mp4 0 20")
        sys.exit(1)
    
    video_path = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    
    print("=" * 70)
    print("СРАВНЕНИЕ МЕТОДОВ ДЕТЕКЦИИ ПОДАЧИ")
    print("=" * 70)
    print(f"Видео: {video_path}")
    print(f"Сегмент: {start}s - {end}s")
    print("=" * 70)
    
    all_results = {}
    
    for variant_id, script_name, description in VARIANTS:
        print(f"\n[*] Запуск: {description} ({script_name})...")
        output = run_variant(script_name, video_path, start, end)
        results = parse_results(output)
        all_results[variant_id] = {
            "description": description,
            "script": script_name,
            **results
        }
        print(f"    Ударов: {results['impacts']}")
        print(f"    Подач:  {results['serves']}")
        if results['serve_times']:
            print(f"    Время подач: {', '.join([f'{t:.1f}s' for t in results['serve_times']])}")
    
    # Итоговая таблица
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 70)
    print(f"{'Метод':<30} {'Ударов':<10} {'Подач':<10} {'Время подач'}")
    print("-" * 70)
    
    for variant_id, data in all_results.items():
        times = ', '.join([f'{t:.1f}s' for t in data['serve_times']]) if data['serve_times'] else '-'
        print(f"{data['description']:<30} {data['impacts']:<10} {data['serves']:<10} {times}")
    
    print("=" * 70)
    
    # Сохранение результатов
    output_dir = "results/serve_comparison"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"comparison_{timestamp}.json")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "video": video_path,
            "segment": {"start": start, "end": end},
            "timestamp": timestamp,
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nОтчёт сохранён: {report_path}")

if __name__ == "__main__":
    main()
