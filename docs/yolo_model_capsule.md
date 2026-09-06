# YOLO Model Capsule — Сегментационная модель

## Scope
**Трогаем:** использование модели, интерпретация результатов, prod/rollback конфиги
**НЕ трогаем:** переобучение модели (требует governance approval)

## Invariants (≤12)

1. **Prod-модель** — `runs/segment/train14_ttseg_v3/weights/best.pt` (SSoT: `scripts/config/detection_params.json`)
2. **Тип** — YOLOv8 Instance Segmentation
3. **Классы train14 (nc=6):** 0 Ball, 1 Net, 2 Player1, 3 Player2, 4 TableLeft, 5 TableRight (SSoT: `results/roboflow_pipeline/runs/tt_seg_v3_export/dataset/data.yaml`)
4. **Предыдущие чекпоинты:** `train12_ball68` (5 классов, Table=4), `train13_ttseg_v2` (база для train14, `runs/segment/train14_ttseg_v3/args.yaml`)
5. **Конфиг SSoT** — `scripts/config/detection_params.json`
6. **Rollback конфиг** — `scripts/config/detection_params_train12.json` → `train12_ball68`
7. **serve_detector** читает `table_class_ids: [4, 5]` из секции `yolo` конфига (`scripts/config/detection_params.json`)
8. **main_processor_stable.py** — только `table_class_id=4`, dual-table не поддерживает (`scripts/main_processor_stable.py:302`, `:331`)
9. **Confidence по умолчанию** — 0.25 (конвенция проекта); `serve_detector.py` вызывает `predict(..., conf=0.1)` (`scripts/serve_detector.py:560`)
10. **Выход boxes** — bounding boxes [x1, y1, x2, y2]
11. **Выход masks** — полигоны сегментации (`result.masks.xy`)
12. **Запрет сторонних весов** как prod-base — `.cursorrules` (только локальные `runs/segment/**`)

## Decisions (≤7)

| Решение | Обоснование |
|---------|-------------|
| train14_ttseg_v3 в prod | Human-gate 2026-09-01: проверка распознавания стола на img0291 |
| Сегментация вместо detection | Нужны полигоны стола/сетки |
| TableLeft + TableRight (4, 5) | Раздельные маски половин стола в train14 |
| Rollback через detection_params_train12.json | Быстрый откат на train12 без правки кода |
| Ball как class 0 | Первый в приоритете детекции |
| train13 → train14 lineage | train14 fine-tune от train13 (`runs/segment/train14_ttseg_v3/args.yaml`) |

## Impact Paths

```
detection_params.json (model_path, table_class_ids)
├── serve_detector.py — BallTracker, dual table polygons (~605)
├── main_processor_stable.py — single table_class_id=4 (эталон, не train14-path)
└── main_processor_*.py — legacy single-table по table_class_id
```

## Validation Proof

```python
from ultralytics import YOLO
model = YOLO('runs/segment/train14_ttseg_v3/weights/best.pt')
result = model(frame)[0]
# result.boxes — bounding boxes, cls 0–5
# result.masks.xy — полигоны TableLeft/TableRight при наличии
```

Rollback: подставить `--config scripts/config/detection_params_train12.json`.

## Risks (≤7)

| Риск | Митигация |
|------|-----------|
| Больше FP «2+ мяча» vs train13 | Мониторинг на img0291; откат на train12 при деградации |
| Пропуск мяча при размытии | Kalman-предсказание в трекере |
| Неточный полигон стола | Fallback на `table_center_x` |
| Несколько детекций мяча | Берётся лучшая по conf среди class=0 |
| Dual-table только в serve_detector | stable/main_processor* не используют [4,5] без доработки |

## Open Questions (≤7)

1. Стабильность TableLeft/TableRight на других ракурсах кроме img0291?
2. Нужен ли отдельный preset `detection_params_train13.json` для A/B?
3. Оптимальный conf для train14: 0.25 vs текущий 0.1 в serve_detector?
4. Перенос dual-table в main_processor_stable или отдельный процессор?
5. Метрики FP multi-ball на полном img0291 vs train13?

## Pointers (≤15)

- `runs/segment/train14_ttseg_v3/weights/best.pt` — prod-веса
- `runs/segment/train12_ball68/weights/best.pt` — rollback-веса
- `runs/segment/train13_ttseg_v2/` — предыдущий чекпоинт
- `scripts/config/detection_params.json` — prod SSoT
- `scripts/config/detection_params_train12.json` — rollback SSoT
- `scripts/config/detection_params_train14.json` — дубликат train14 preset
- `results/roboflow_pipeline/runs/tt_seg_v3_export/dataset/data.yaml` — классы train14
- `data.yaml` — legacy 5 классов (train10/train12)
- `scripts/serve_detector.py:539-543` — table_class_id(s) init
- `scripts/serve_detector.py:597-658` — `_maybe_update_table_polygon()` dual-table
- `scripts/serve_detector.py:560` — YOLO predict conf
- `scripts/main_processor_stable.py:302` — single table_class_id
- `runs/segment/train14_ttseg_v3/args.yaml` — lineage train13→train14
- `.cursorrules` — запрет сторонних весов, актуальный prod path
