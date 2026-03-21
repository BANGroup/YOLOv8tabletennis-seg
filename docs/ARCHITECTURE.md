# ARCHITECTURE - Архитектура системы Table Tennis CV Score

**Дата:** 23-01-2026  
**Версия:** 3.0  
**Статус:** Рефакторинг завершён, модульная архитектура

---

## 🏗️ ОБЩАЯ АРХИТЕКТУРА v3

```
┌─────────────────────────────────────────────────────────┐
│                  VIDEO INPUT (60 FPS)                   │
│              GX010138.MP4 (сегмент 5 мин)              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│             YOLO v8 SEGMENTATION MODEL                  │
│  (runs/segment/train10 - Ball, Net, Players, Table)    │
│                                                         │
│  ➜ Детектирует: Мяч, Сетку, Игроков, Стол             │
│  ➜ Вывод: Bounding boxes + Маски сегментации           │
│  ➜ Confidence: 0.1 (настраивается)                     │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
    ┌─────────────┐      ┌──────────────┐
    │  Ball Det.  │      │ Table Mask   │
    │  (x, y, a)  │      │  (polygon)   │
    └─────┬───────┘      └────────┬─────┘
          │                       │
          └───────────┬───────────┘
                      │
                      ▼
        ┌────────────────────────────┐
        │   KALMAN FILTER v3         │
        │   (x, y, vx, vy, ax, ay)   │
        │                            │
        │ ➜ Модель с ГРАВИТАЦИЕЙ     │
        │ ➜ Адаптивный шум           │
        │ ➜ Предсказание N кадров    │
        │ ➜ Автосброс при потере     │
        └─────────┬──────────────────┘
                  │
                  ▼
        ┌─────────────────────────┐
        │  BALL TRACKER v3        │
        │  (core/ball_tracker.py) │
        │                         │
        │ ✓ Фильтрация по размеру │
        │ ✓ Проверка скорости     │
        │ ✓ Fallback зона стола   │
        │ ✓ Модульная структура   │
        └────────┬────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │  IMPACT DETECTOR v3            │
    │  (core/impact_detector.py)     │
    │                                │
    │ ✓ Все параметры из конфига     │
    │ ✓ Fallback если нет полигона   │
    │ ✓ Dataclass для событий        │
    │ ✓ Логирование                  │
    │                                │
    │ Паттерн: вниз → вверх          │
    │ (вогнутая траектория)          │
    └────────┬─────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────┐
  │  VISUALIZER v3                   │
  │  (core/visualizer.py)            │
  │                                  │
  │ ➜ Отделён от логики трекинга     │
  │ ➜ Конфигурируемые цвета/размеры  │
  │ ➜ Траектория с градиентом        │
  │ ➜ Точки ударов                   │
  │ ➜ Статистика HUD                 │
  └────────┬─────────────────────────┘
           │
           ▼
    ┌─────────────────────────┐
    │  VIDEO OUTPUT (60 FPS)  │
    │                         │
    │  ➜ MP4 (mp4v кодек)     │
    │  ➜ results/v3/          │
    └─────────────────────────┘
           │
           ▼
    ┌─────────────────────────┐
    │   ANALYSIS OUTPUT       │
    │                         │
    │  ➜ analysis_v3.json     │
    │  ➜ live_impacts_v3.json │
    │  ➜ results/v3/          │
    └─────────────────────────┘
```

---

## 📦 МОДУЛЬНАЯ СТРУКТУРА v3

```
scripts/
├── main_processor.py          # v1 - оригинал (не трогать!)
├── main_processor_v2.py       # v2 - оптимизация параметров
├── main_processor_v3.py       # v3 - модульная архитектура ⭐
│
├── core/                      # Модули v3
│   ├── __init__.py           # Экспорт классов
│   ├── kalman.py             # KalmanFilterWithGravity
│   ├── impact_detector.py    # SmartImpactDetector + Config
│   ├── ball_tracker.py       # BallTracker + Config
│   └── visualizer.py         # TrackingVisualizer + Config
│
└── config/
    ├── detection_params.json      # v1/v2 конфиг
    └── detection_params_v3.json   # v3 конфиг с пресетами ⭐
```

---

## 🔧 КОМПОНЕНТЫ v3

### 1. KalmanFilterWithGravity

**Файл:** `scripts/core/kalman.py`

**Состояние:** `[x, y, vx, vy, ax, ay]` — позиция + скорость + ускорение

**Улучшения v3:**
- Модель гравитации (ay ≈ 0.5 пикс/кадр²)
- Адаптивный шум измерений
- Предсказание на N кадров вперёд
- Автосброс после 30 пропущенных кадров

```python
# Матрица перехода с ускорением
F = [
    [1, 0, dt, 0,  0.5*dt², 0      ],
    [0, 1, 0,  dt, 0,       0.5*dt²],
    [0, 0, 1,  0,  dt,      0      ],
    [0, 0, 0,  1,  0,       dt     ],
    [0, 0, 0,  0,  1,       0      ],
    [0, 0, 0,  0,  0,       1      ]
]
```

---

### 2. SmartImpactDetector

**Файл:** `scripts/core/impact_detector.py`

**Конфигурация через dataclass:**
```python
@dataclass
class ImpactDetectorConfig:
    min_consecutive_detections: int = 3
    min_frame_gap: int = 6
    curvature_threshold: float = 10.0
    y_range_threshold: float = 15.0
    # ... все параметры документированы
```

**Fallback зона стола:**
- Если YOLO не обнаружил стол → используется прямоугольная зона
- Настраивается через `fallback_table_zone_*` параметры

---

### 3. BallTracker

**Файл:** `scripts/core/ball_tracker.py`

**Объединяет:**
- YOLO детекцию
- Калман фильтр
- Детектор ударов
- Извлечение полигона стола

**Конфигурация:**
```python
@dataclass
class BallTrackerConfig:
    model_path: str = "runs/segment/train10/weights/best.pt"
    confidence: float = 0.1
    kalman_gravity: float = 0.5
    ball_min_area: float = 80
    ball_max_area: float = 3000
```

---

### 4. TrackingVisualizer

**Файл:** `scripts/core/visualizer.py`

**Отделён от логики трекинга** (принцип единственной ответственности)

**Рисует:**
- Траекторию с градиентом цвета
- Точки ударов (красные)
- Стрелку скорости
- Полигон стола
- HUD со статистикой
- Время

---

## 📋 КОНФИГУРАЦИЯ v3

**Файл:** `scripts/config/detection_params_v3.json`

### Секции:

| Секция | Описание |
|--------|----------|
| `yolo` | Модель, confidence, class IDs |
| `kalman` | Гравитация, шумы |
| `tracking` | Размеры мяча, история |
| `impact_detection` | Все пороги детекции ударов |
| `visualization` | Включение/выключение элементов |
| `output` | Папка результатов |
| `presets` | Готовые пресеты |

### Пресеты:

| Пресет | Описание | Когда использовать |
|--------|----------|-------------------|
| `sensitive` | Больше ударов, возможны ложные | Когда теряются удары |
| `strict` | Меньше ложных, могут пропускаться | Когда много ложных срабатываний |

**Использование:**
```bash
python main_processor_v3.py --video VIDEO --start 395 --end 415 --preset sensitive
```

---

## 🔄 ПОТОК ДАННЫХ v3

```
Frame (60 FPS)
    │
    ├─► BallTracker.track(frame, idx, time)
    │   │
    │   ├─► detect_ball(frame)
    │   │   ├─► YOLO.predict()
    │   │   ├─► _extract_table_polygon() [один раз]
    │   │   └─► Фильтрация по размеру
    │   │   ──► (x, y) или None
    │   │
    │   ├─► KalmanFilterWithGravity
    │   │   ├─► update() при детекции
    │   │   ├─► predict() при пропуске
    │   │   └─► Адаптация шума
    │   │
    │   ├─► SmartImpactDetector
    │   │   ├─► detect_impact() - анализ паттерна
    │   │   ├─► process_detection() - подтверждение
    │   │   └─► on_missed_frame() - при пропуске
    │   │
    │   └─► Обновление статистики
    │
    ├─► TrackingVisualizer.draw()
    │   ├─► _draw_table()
    │   ├─► _draw_trajectory()
    │   ├─► _draw_impacts()
    │   ├─► _draw_velocity_arrow()
    │   ├─► _draw_stats()
    │   └─► _draw_time()
    │
    └─► VideoWriter.write()
```

---

## 📊 СРАВНЕНИЕ ВЕРСИЙ

| Аспект | v1 | v2 | v3 |
|--------|----|----|-----|
| Структура | Монолит | Монолит | **Модульная** |
| Калман | 4 состояния | 4 состояния | **6 состояний + гравитация** |
| Параметры | Hardcoded | Частично в конфиге | **Все в конфиге** |
| Fallback стола | Нет | Нет | **Есть** |
| Логирование | print() | print() | **logging** |
| Пресеты | Нет | Нет | **sensitive/strict** |
| Дублирование | Нет | v1 копия | **Нет** |
| Визуализация | В трекере | В трекере | **Отдельный модуль** |

---

## 🚀 ИСПОЛЬЗОВАНИЕ v3

### Базовый запуск:
```bash
cd "/home/misha/Рабочий стол/YOLOv8tabletennis-seg/YOLOv8tabletennis-seg"
source venv/bin/activate

python scripts/main_processor_v3.py \
  --video sources/GX010138.MP4 \
  --start 395 \
  --end 415
```

### С пресетом:
```bash
# Чувствительный (больше ударов)
python scripts/main_processor_v3.py --video VIDEO --start 395 --end 415 --preset sensitive

# Строгий (меньше ложных)
python scripts/main_processor_v3.py --video VIDEO --start 395 --end 415 --preset strict
```

### Docker:
```bash
docker run --rm \
  -v $(pwd)/sources:/app/sources \
  -v $(pwd)/results:/app/results \
  tt-tracker:v1 \
  scripts/main_processor_v3.py --video sources/VIDEO.MP4 --start 395 --end 415
```

---

## 📂 ФАЙЛЫ ПРОЕКТА

```
YOLOv8tabletennis-seg/
├── scripts/
│   ├── main_processor.py      # v1 (оригинал, НЕ ТРОГАТЬ)
│   ├── main_processor_v2.py   # v2 (оптимизация)
│   ├── main_processor_v3.py   # v3 (модульный) ⭐
│   ├── core/                  # Модули v3
│   │   ├── __init__.py
│   │   ├── kalman.py
│   │   ├── impact_detector.py
│   │   ├── ball_tracker.py
│   │   └── visualizer.py
│   └── config/
│       ├── detection_params.json
│       └── detection_params_v3.json
│
├── runs/segment/train10/      # Модель YOLO
│   └── weights/best.pt
│
├── results/
│   ├── v3/                    # Результаты v3
│   └── data/                  # Старые результаты
│
├── docs/
│   ├── ARCHITECTURE.md        # Этот файл
│   ├── ANALYSIS.md
│   └── CHANGELOG.md
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## ✅ СТАТУС v3

**Исправлены проблемы:**
- ✅ Монолитная структура → Модульная
- ✅ Дублирование v1/v2 → Один параметризуемый класс
- ✅ Калман без гравитации → 6-состояний с гравитацией
- ✅ Hardcoded пороги → Все в конфиге
- ✅ Нет fallback стола → Есть fallback зона
- ✅ Устаревшая документация → Обновлена

**Готово к использованию!** 🎉

---

## 🔮 БУДУЩИЕ УЛУЧШЕНИЯ

1. **Unit тесты** для модулей core/
2. **Классификация типов ударов** (топспин, подрезка)
3. **FSM для подсчёта очков**
4. **GPU оптимизация** (batch inference)
5. **Real-time режим** (веб-камера/стрим)
