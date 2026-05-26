# Старт работы — YOLOv8tabletennis-seg

**Это корень git-репозитория.** Открывай в Cursor именно эту папку, не родительскую `YOLOv8tabletennis/`.

```
/home/misha/Рабочий стол/YOLOv8tabletennis/YOLOv8tabletennis-seg-main
```

GitHub: [BANGroup/YOLOv8tabletennis-seg](https://github.com/BANGroup/YOLOv8tabletennis-seg)

---

## 1. Открыть проект в Cursor

**File → Open Folder** → выбери **`YOLOv8tabletennis-seg-main`**.

Или дважды кликни файл **`YOLOv8tabletennis-seg.code-workspace`** в этой папке.

После смены папки: **Ctrl+Shift+P → Developer: Reload Window**.

---

## 2. Roboflow MCP

| Файл | Назначение |
|------|------------|
| `.cursor/mcp.json` | конфиг MCP (Roboflow HTTP) |
| `.env` | **Private API Key** — не коммитится |
| `.env.example` | шаблон |
| `scripts/config/roboflow.json` | workspace/project в Roboflow |

Проверка: **Cursor Settings → MCP** → сервер `roboflow` = **Connected**.

Ключ: [app.roboflow.com/settings/api](https://app.roboflow.com/settings/api)

---

## 3. Git (push / pull)

Команды **только из этой папки**:

```bash
cd "/home/misha/Рабочий стол/YOLOv8tabletennis/YOLOv8tabletennis-seg-main"
git status
git pull origin main
git push origin main
```

Или кнопка **Sync / Push** в Source Control (когда открыта эта папка).

---

## 4. Актуальная модель и конфиг

| Что | Путь |
|-----|------|
| Модель (seg) | `runs/segment/train12_ball68/weights/best.pt` |
| Параметры детекции | `scripts/config/detection_params.json` |
| Правило «только своя -seg модель» | `.cursorrules` |

---

## 5. Основные команды

```bash
# Трекинг + подача + счёт (development)
python scripts/serve_detector.py \
  --video /path/to/video.MOV \
  --start 0 --end 357 \
  --config scripts/config/detection_params.json

# Эталон трекинга ударов
python scripts/main_processor_stable.py --video VIDEO.mp4 --start 0 --end 20
```

Результаты: `results/<имя_прогона>/`

---

## 6. Roboflow проект (датасет)

- Workspace: `yolo-table-tennis`
- Project: `auto-label-lmrwb`
- Universe: https://universe.roboflow.com/yolo-table-tennis/auto-label-lmrwb

---

## 7. Что игнорировать в Source Control

Не коммитить без необходимости:

- `.venv_train/` — локальное venv
- `dataset/**/labels.cache` — кэш YOLO
- `results/**/*.mp4` — видео (в `.gitignore`)
- `runs/segment/runs/` — артеfact nested от ultralytics

---

## 8. Родительская папка `YOLOv8tabletennis/`

Старая обёртка Cursor (без `.git`). **Не работай оттуда** — там нет репозитория и push падает с `fatal: не найден git репозиторий`.

Все наработки (коммит `train12`, `serve_detector`, MCP, `.env`) — **здесь**, в `seg-main`.

---

*Обновлено: май 2026*
