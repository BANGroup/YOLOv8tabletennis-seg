# YOLO Model Capsule — Сегментационная модель

## Scope
**Трогаем:** использование модели, интерпретация результатов
**НЕ трогаем:** переобучение модели (требует governance approval)

## Invariants (≤12)

1. **Модель** — `runs/segment/train10/weights/best.pt`
2. **Тип** — YOLOv8 Instance Segmentation
3. **Классы:**
   - 0: Ball (мяч)
   - 1: Net (сетка)
   - 2: Player1 (игрок слева)
   - 3: Player2 (игрок справа)
   - 4: Table (стол)
4. **Выход boxes** — bounding boxes [x1, y1, x2, y2]
5. **Выход masks** — полигоны сегментации (masks.xy)
6. **Конфиденс** — по умолчанию 0.25

## Decisions (≤7)

| Решение | Обоснование |
|---------|-------------|
| Сегментация вместо detection | Нужны полигоны стола |
| train10 как финальная | Лучшие метрики |
| Ball как class 0 | Первый в приоритете |

## Impact Paths

```
Модель используется в:
├── BallTracker.detect_ball() — детекция мяча
├── BallTracker._maybe_update_table_polygon() — полигон стола
└── BallTracker._maybe_update_net() — bbox сетки
```

## Validation Proof

```python
from ultralytics import YOLO
model = YOLO('runs/segment/train10/weights/best.pt')
result = model(frame)[0]
# result.boxes — bounding boxes
# result.masks.xy — полигоны
```

## Risks (≤7)

| Риск | Митигация |
|------|-----------|
| Пропуск мяча при размытии | Kalman предсказание |
| Неточный полигон стола | Fallback на center_x |
| Несколько детекций мяча | Берём первую с class=0 |

## Open Questions (≤7)

1. Нужно ли переобучить на большем датасете?
2. Добавить класс для ракетки?
3. Оптимальный confidence threshold?

## Pointers (≤15)

- `runs/segment/train10/weights/best.pt` — веса модели
- `data.yaml` — конфигурация классов
- `scripts/serve_detector.py:519` — загрузка модели
