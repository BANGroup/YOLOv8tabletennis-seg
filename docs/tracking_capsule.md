# Tracking Capsule — Детекция ударов в настольном теннисе

## Scope
**Трогаем:** детекция мяча, траектория, определение ударов об стол, конфиг prod-модели
**НЕ трогаем:** подсчёт очков (scoring_capsule), переобучение YOLO

## Invariants (≤12)

1. **Prod YOLO** — `runs/segment/train14_ttseg_v3/weights/best.pt` (SSoT: `scripts/config/detection_params.json`)
2. **Классы train14** — Ball(0), Net(1), Player1(2), Player2(3), TableLeft(4), TableRight(5) (SSoT: `tt_seg_v3_export/dataset/data.yaml`)
3. **Удар = вогнутая траектория** — мяч шёл вниз, отскочил вверх (`serve_detector.py:213-221`)
4. **Y-range >= 8** — минимальное движение по Y для удара (`serve_detector.py:215`)
5. **Curvature >= 5** — минимальная кривизна траектории (`serve_detector.py:221`)
6. **min_frame_gap = 12** — минимум 12 кадров между ударами (~0.2 сек) (`detection_params.json`, `serve_detector.py:73`)
7. **Удар внутри стола** — точка должна быть в `table_polygon` (объединённый hull)
8. **Сторона по X** — left если X < `table_center_x`, иначе right (`serve_detector.py:236-237`); scoring **не** использует TableLeft/TableRight полигоны для стороны
9. **`bounce_frame`/`bounce_time`** — середина `frame_history`/`time_history`; вся time-логика impact использует их, не текущий кадр
10. **`min_consecutive_detections=3`** — минимум 3 детекции мяча в окне ±2 кадра вокруг `bounce_frame` (`detected_frames`, `detection_window_radius=2`); имя в конфиге без изменений, изменён смысл
11. **Kalman фильтр** — сглаживание траектории при пропусках
12. **Rollback трекинга на train12** — `detection_params_train12.json`, single Table class 4

## Decisions (≤7)

| Решение | Обоснование |
|---------|-------------|
| Вогнутость вместо скорости | Скорость нестабильна, вогнутость надёжнее |
| Окно ±2 кадра вместо счётчика кандидатов | Старый счётчик терял 21/22 отскоков на IMG_0291 — SSoT: `process_detection()` |
| Отклонена строгая серия подряд кадров | Теряла 7 реальных отскоков из-за одиночных пропусков детекции |
| bounce timing из history midpoint | Устранено отставание 4–9 кадров между координатой и timestamp |
| serve_detector: два class-specific полигона | train14 даёт TableLeft/TableRight |
| min_frame_gap = 12 | Меньше — дубликаты; side-dependent gap — следующий шаг |
| `--debug-trace` optional | `debug_trace.json` с `last_reject_reason`/`last_process_reject`; default off |

## Impact Paths

```
scripts/config/detection_params.json
├── serve_detector.py (prod path, table_class_ids [4,5])
│   ├── SmartImpactDetector.detect_impact() — y_range, curvature, frame/time_history → bounce_*
│   ├── process_detection() — detected_frames window, min_frame_gap, prev_impact_time
│   ├── BallTracker._maybe_update_table_polygon() — dual TableLeft/TableRight
│   ├── BallTracker.track_ball() — is_impact_on_net → scoring pending_net
│   └── side assignment — table_center_x (не полигоны 4/5)
└── main_processor_stable.py — эталон, single table_class_id=4
```

## Validation Proof

```bash
# Prod serve path с train14 (IMG_0291, 45–180 s)
python scripts/serve_detector.py --config scripts/config/detection_params.json
# Ожидается на consecfix: 87/94 физ. отскоков принято, 85 impacts, счёт 8:6
# Артефакт: results/serve_detector_train14_img0291_consecfix/tracked_v5.mp4

# Опционально: покадровая отладка
python scripts/serve_detector.py --config scripts/config/detection_params.json --debug-trace
# → output_dir/debug_trace.json
```

Метрики IMG_0291 45–180 s (`docs/CHANGELOG.md` [06-09-2026]): принято отскоков 71→**87**, потеряно 22→**7**, impacts 66→**85**.

## Risks (≤7)

| Риск | Митигация |
|------|-----------|
| `min_frame_gap` склеивает соседние отскоки | 6 кластеров потерь (было 1); план side-dependent gap |
| Ложные удары при шуме | y_range >= 8 + окно detected_frames |
| Пропуск быстрых ударов | Kalman предсказывает позицию |
| Неправильная сторона | `table_center_x` / net_zone_x, не полигоны 4/5 |
| stable vs serve_detector расхождение | Явно различать baseline и prod dual-table path |
| train14 FP multi-ball | Мониторинг; откат detection_params_train12.json |
| Одиночные пропуски детекции | Окно ±2 кадра; не strict consecutive |

## Open Questions (≤7)

1. Side-dependent `min_frame_gap` — пороги по left/right?
2. Оптимальные `detection_window_radius` и `min_consecutive_detections` для train14?
3. Как улучшить детекцию на правой стороне с train14?
4. Нужен ли перенос dual-table в stable baseline?
5. Влияет ли hull двух половин на false in-table/out-table?
6. Калибровка table_center_x от net vs bbox стола на train14?

## Pointers (≤15)

- `scripts/config/detection_params.json` — prod model_path, table_class_ids
- `scripts/config/detection_params_train12.json` — rollback
- `scripts/serve_detector.py:143-157` — frame/time_history, detected_frames, pending state
- `scripts/serve_detector.py:206-282` — detect_impact(), bounce_frame/bounce_time
- `scripts/serve_detector.py:294-312` — process_detection() window filter
- `scripts/serve_detector.py:236-237` — side by center_x
- `scripts/serve_detector.py:539-543` — table_class_ids init
- `scripts/serve_detector.py:597-658` — dual table polygon
- `scripts/serve_detector.py:73` — min_frame_gap default
- `scripts/serve_detector.py:948` — is_impact_on_net() (net during rally)
- `scripts/serve_detector.py:1453` — `--debug-trace` CLI
- `scripts/serve_detector.py:1681` — debug_trace.json export
- `scripts/main_processor_stable.py` — эталон (single table_class_id=4)
- `runs/segment/train14_ttseg_v3/weights/best.pt` — prod YOLO
- `results/serve_detector_train14_img0291_consecfix/` — consecfix validation output
