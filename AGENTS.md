# YOLOv8tabletennis-seg — Карта проекта для агента и разработчика

Система автоматического трекинга мяча, детекции отскоков, подач и подсчёта очков в настольном теннисе на базе YOLOv8 Instance Segmentation и фильтра Калмана.

## 1. Единственный источник истины (SSoT) по модели и параметрам

- **Актуальный конфиг прод-модели и порогов:** `scripts/config/detection_params.json`
  - Модель: `runs/segment/train14_ttseg_v3/weights/best.pt` (YOLOv8n-seg, 6 классов)
  - Классы (`nc=6`): `0: Ball`, `1: Net`, `2: Player1`, `3: Player2`, `4: TableLeft`, `5: TableRight`
  - Конфиг отката на предыдущую модель (5 классов): `scripts/config/detection_params_train12.json`
- **Реестр видео и разделение выборок (`train` / `tune` / `test`):** `config/data_split.json`
- **Эталоны приёмки и регрессии:** `tests/expected/*.yaml` + скрипт сверки `scripts/verify_expected.py`
- **Сторонние веса** (`yolov8n.pt`, `yolov8n-seg.pt` и др. из интернета) как база для инференса или дообучения **запрещены**.

## 2. Основные скрипты и зоны

- `scripts/serve_detector.py` — **основной прод-пайплайн**: сегментация (две половины стола `TableLeft` + `TableRight`, сетка `Net`, ориентация игроков `Player1`/`Player2`), трекинг мяча (Kalman), детекция ударов (`SmartImpactDetector`) и подсчёт очков (подача, сетка, аут с отложенным окном 0.4 с, двойной отскок).
- `scripts/verify_expected.py` — **автоматическая сверка результатов** (`smart_impacts_analysis.json` и `run_stdout.log`) с эталонами из `tests/expected/*.yaml`.
- `scripts/sync_nas_videos.py` — **синхронизация и статус видео между локальным `sources/` и `Home_NAS`** (`/mnt/nas-drive/CV/tabletennis/{raw,sdr,exports}`).
- `scripts/main_processor_stable.py` — **неприкосновенный baseline** (работает с одним классом стола `table_class_id=4`). Не изменять.
- `scripts/find_missed_balls.py` — поиск пропусков мяча для доразметки.
- `results/roboflow_pipeline/scripts/` — конвейер отбора кадров (`sample_frames.py`), предразметки (`prelabel_yolo_seg.py`), упаковки (`build_upload_zip.py`), загрузки в Roboflow, проверки сплитов (`repack_splits_by_video.py`) и дообучения (`train_ttseg_v2.py`).

## 3. Роли агентов (3 контура)

- **`orchestrator`** (главный агент сессии): архитектура, план, настройка `scripts/config/*.json` и `config/data_split.json`, подбор кадров/параметров, ведение `config/knowledge_graph.json`, `docs/DECISIONS.md`, `docs/CHANGELOG.md` и капсул. Код в `scripts/*.py` напрямую не пишет — ставит задачу `coder`.
- **`coder`** (`.cursor/agents/coder.md`): реализует изменения в Python-коде (`scripts/**`) по ТЗ оркестратора и проверяет синтаксис (`python3 -m py_compile`).
- **`validator`** (`.cursor/agents/validator.md`): запускает прогоны на видео (`serve_detector.py`, `ffmpeg`) и сверку эталонов (`scripts/verify_expected.py`), собирает метрики «до → после» (длинные прогоны — в фоне).

Модели субагентов заданы в `config/agent_models.json` (`grok-4.7-high-fast`).

## 4. Частые команды и проверка

```bash
# Автоматическая сверка со всеми эталонами из tests/expected/*.yaml
python3 scripts/verify_expected.py --all

# Сверка конкретного нового прогона с эталоном
python3 scripts/verify_expected.py \
  --spec tests/expected/IMG_0291_45_180.yaml \
  --actual results/<run_name>/smart_impacts_analysis.json

# Прод-прогон serve_detector с актуальным конфигом
YOLO_OFFLINE=1 python3 scripts/serve_detector.py \
  --video sources/IMG_0291.MOV --start 45 --end 180 \
  --config scripts/config/detection_params.json --debug-trace

# Baseline (только single-table, train12)
YOLO_OFFLINE=1 python3 scripts/main_processor_stable.py \
  --video GX010025_20sec_raw.mp4 --start 0 --end 20

# Проверка и синхронизация видео с NAS (/mnt/nas-drive/CV/tabletennis)
python3 scripts/sync_nas_videos.py --status
python3 scripts/sync_nas_videos.py --push
python3 scripts/sync_nas_videos.py --pull --symlink
```

## 5. Бэклог развития (зафиксирован, не в текущем спринте)

- **B-01 (`LET` / Переподача при касании сетки на подаче):** если после 1-го отскока подачи мяч задевает полигон `Net` и приземляется на половине принимающего — фиксировать `LET` (переподача, очко никому не начисляется, очередь подач не сдвигается).
- **B-02 (Универсальность под любое видео из сети / движущуюся камеру):** периодическое обновление (`EMA`) полигонов `TableLeft`, `TableRight` и `Net` вместо однократной фиксации на старте, определение стороны по полигонам половин стола/оси сетки и скользящее окно ориентации `Player1`/`Player2`.

## 6. Карта знаний и документации

- `config/knowledge_graph.json` — компактный граф знаний проекта в JSON (узлы, связи, инварианты, ключевые решения).
- `config/data_split.json` — реестр видео, ракурсов, цветовых профилей и ролей (`train_candidates`, `tune`, `test`).
- `config/capsules_manifest.json` — индекс контекстных капсул:
  - `docs/tracking_capsule.md` — детекция мяча, геометрия стола и ударов;
  - `docs/scoring_capsule.md` — правила начисления очков, подача, сетка, аут, двойной отскок, ориентация P1/P2;
  - `docs/yolo_model_capsule.md` — веса, классы, конфиги и пайплайн Roboflow.
- `docs/DECISIONS.md` — журнал архитектурных и параметрических решений (одна строка на решение).
- `docs/CHANGELOG.md` — хронология изменений с замерами «до → после».
