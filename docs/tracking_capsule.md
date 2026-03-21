# Tracking Capsule — Детекция ударов в настольном теннисе

## Scope
**Трогаем:** детекция мяча, траектория, определение ударов об стол
**НЕ трогаем:** подсчёт очков, визуализация, YOLO модель

## Invariants (≤12)

1. **YOLO модель** — `runs/segment/train10/weights/best.pt`
2. **Классы** — Ball(0), Net(1), Player1(2), Player2(3), Table(4)
3. **Удар = вогнутая траектория** — мяч шёл вниз, отскочил вверх
4. **Y-range >= 8** — минимальное движение по Y для удара
5. **Curvature >= 5** — минимальная кривизна траектории
6. **min_frame_gap = 12** — минимум 12 кадров между ударами (~0.2 сек)
7. **Удар внутри стола** — точка должна быть в table_polygon
8. **Сторона по X** — left если X < table_center_x, иначе right
9. **Kalman фильтр** — сглаживание траектории при пропусках
10. **Stable baseline** — `main_processor_stable.py` = эталон (15 ударов на 20 сек)

## Decisions (≤7)

| Решение | Обоснование |
|---------|-------------|
| Вогнутость вместо скорости | Скорость нестабильна, вогнутость надёжнее |
| Один полигон стола | YOLO часто даёт 1 полигон, не 2 половины |
| table_center_x для стороны | Полигоны ненадёжны, X-координата проще |
| min_frame_gap = 12 | Меньше — дубликаты, больше — пропуски |
| y_range = 8 | Компромисс между чувствительностью и шумом |

## Impact Paths

```
serve_detector.py
├── SmartImpactDetector.detect_impact() → определяет удар
├── SmartImpactDetector.process_detection() → фильтрует и сохраняет
├── BallTracker.track_ball() → основной цикл трекинга
└── BallTracker._maybe_update_table_polygon() → определяет границы стола
```

## Validation Proof

```bash
# Эталонный тест
python scripts/main_processor_stable.py --video GX010025_20sec_raw.mp4 --start 0 --end 20
# Ожидается: 15 ударов, Left: 7, Right: 7
```

## Risks (≤7)

| Риск | Митигация |
|------|-----------|
| Ложные удары при шуме | y_range >= 8 фильтрует мелкие колебания |
| Пропуск быстрых ударов | Kalman предсказывает позицию |
| Неправильная сторона | Используем center_x вместо полигонов |
| Один полигон вместо двух | Fallback на X-координату |

## Open Questions (≤7)

1. Как улучшить детекцию на правой стороне?
2. Оптимальные параметры для разных ракурсов?
3. Нужна ли калибровка под каждое видео?

## Pointers (≤15)

- `scripts/serve_detector.py:178` — detect_impact()
- `scripts/serve_detector.py:270` — process_detection()
- `scripts/serve_detector.py:568` — _maybe_update_table_polygon()
- `scripts/main_processor_stable.py` — эталонная реализация
- `scripts/config/detection_params.json` — параметры
- `runs/segment/train10/weights/best.pt` — YOLO модель
