# Results - Структура папки

**Дата организации:** 14-11-2025

---

## 📊 СТРУКТУРА

### `data/` - Полезные данные (JSON файлы анализа)

```
data/
├── impacts_analysis.json          - Анализ ударов о стол
├── smart_impacts_analysis.json    - Анализ smart_tracking логики
└── rallies.json                   - Данные о обнаруженных rally
```

**Назначение:** Хранит итоговые данные анализа для обработки и визуализации.

**Использование:**
- Импортировать в Python: `import json; data = json.load(open('data/impacts_analysis.json'))`
- Использовать в аналитике и отчетах
- Переносить в базу данных при необходимости

---

### `videos/` - Видео результаты

```
videos/
├── tracked_2sec_smart_impacts.mp4     - Видео трекинга со smart логикой (2 сек)
└── tracked_2sec_table_impacts.mp4     - Видео трекинга с detection ударов (2 сек)
```

**Назначение:** Финальные видео с отслеживанием мяча и аналитикой.

**Содержимое:**
- Траектория мяча (визуализация)
- Обнаруженные удары/события
- Статистика в углу экрана
- Время и координаты

---

### `archive/` - Архив (старые данные, промежуточные результаты)

```
archive/
├── rally_detection_results_full/      (899 JPG кадров)
│   └── rally_frame_*.jpg
│
├── segmentation_results/              (34 JPG кадра + 4 скрипта)
│   ├── seg_frame_*.jpg
│   ├── extract_frames.py
│   ├── extract_end_frames.py
│   └── ...
│
├── smart_tracking/                    (промежуточные результаты)
│   ├── smart_tracking_results/
│   │   └── smart_frame_*.jpg (многократное количество)
│   └── smart_tracking_results_combined/
│       ├── ANALYSIS_COMPLETE.txt
│       └── rallies.json
│
├── templates/                         (старые/тестовые скрипты)
│   ├── test_ball_tracking.py
│   ├── test_tracking.py
│   └── tracking_segmentation.py
│
├── track_2sec_smart_impacts/          (дубликат данных из videos/)
│   ├── smart_impacts_analysis.json
│   └── tracked_2sec_smart_impacts.mp4
│
└── track_2sec_table_impacts/          (дубликат данных из videos/)
    ├── impacts_analysis.json
    └── tracked_2sec_table_impacts.mp4
```

**Назначение:** Хранение старых экспериментов и промежуточных результатов.

**Состояние:** АРХИВ - не используется в основном пайплайне.

---

### `sources/` - Исходные видео

```
sources/
├── GX010138.MP4               - Основное видео (60 минут)
└── (другие исходные видео)
```

**Назначение:** Входные видео для обработки.

**Использование:**
- Не модифицировать!
- Использовать как source в скриптах обработки
- Хранить в безопасности (backup)

---

## 🔄 РАБОЧИЙ ПРОЦЕСС

### Обработка видео:

1. **Источник:** `sources/GX010138.MP4`
2. **Обработка:** `scripts/main_processor.py` (основной скрипт)
3. **Результаты:**
   - Видео → `videos/`
   - Анализ → `data/`
   - Старые версии → `archive/`

### Использование данных:

```python
# Загрузить анализ ударов
import json
with open('data/impacts_analysis.json') as f:
    impacts = json.load(f)

# Загрузить rally данные
with open('data/rallies.json') as f:
    rallies = json.load(f)
```

---

## 📝 РАЗМЕРЫ

| Папка | Размер | Примечание |
|-------|--------|-----------|
| `data/` | ~500 KB | JSON файлы |
| `videos/` | ~100-200 MB | MP4 видео |
| `archive/` | ~2-3 GB | Кадры, старые данные |
| `sources/` | ~2-3 GB | Исходные видео |

---

## 🧹 ПОДДЕРЖКА

### Чистка архива (если нужна):

```bash
# Удалить старые JPG кадры
Remove-Item archive/rally_detection_results_full -Recurse
Remove-Item archive/segmentation_results -Recurse

# Оставить только JSON и MP4
```

### Добавление новых результатов:

1. Сохранить полезные JSON в `data/`
2. Сохранить видео в `videos/`
3. Старое переместить в `archive/`

---

## ✅ ИТОГО

✓ **data/** - Конечные данные анализа  
✓ **videos/** - Финальные видео результаты  
✓ **archive/** - Всё старое и промежуточное  
✓ **sources/** - Исходные видео (не трогать)  

**Структура ГОТОВА к использованию!**

