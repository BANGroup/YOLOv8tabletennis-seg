#!/usr/bin/env python3
"""Синхронизация видео sources/, Windows-архива и NAS.

Читает config/data_split.json. Раскладка на NAS:
  /mnt/nas-drive/CV/tabletennis/videos/<VIDEO_ID>/{raw,sdr,clips,frames,passport.json}

--status  — LOCAL / WIN_DISK / NAS и паспорт.
--push    — копирует файл из sources/ или win_archive_path на NAS и обновляет паспорта.
--pull    — забирает отсутствующие локально файлы с NAS (копия или --symlink).
--ingest-inbox — переносит новые ролики из inbox/ в videos/<id>/raw/.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_SPLIT_PATH = REPO_ROOT / "config" / "data_split.json"
PASSPORTS_PATH = REPO_ROOT / "config" / "video_passports.json"

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}
TREE_DIRS = ("raw", "sdr", "clips", "frames")
COPY_CHUNK = 8 * 1024 * 1024
COPY_REPORT = 256 * 1024 * 1024

NAS_HINT = (
    "NAS не смонтирован в /mnt/nas-drive. Выполните: "
    "cp /home/misha/Infra/hosts/ubuntu-misha/smb/nas-smb-credentials.example "
    "~/.config/nas-smb-credentials && chmod 600 ~/.config/nas-smb-credentials && "
    "sudo /home/misha/Infra/hosts/ubuntu-misha/smb/install-smb-drive.sh"
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Статус и синхронизация видео sources/ и Windows-архива с NAS "
            "(config/data_split.json → /mnt/nas-drive/CV/tabletennis/videos)."
        )
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--status",
        action="store_true",
        help="Сводка LOCAL / WIN_DISK / NAS и passport.json (режим по умолчанию)",
    )
    action.add_argument(
        "--push",
        action="store_true",
        help="Скопировать файлы из sources/ или win_archive_path на NAS и обновить паспорта",
    )
    action.add_argument(
        "--pull",
        action="store_true",
        help="Забрать с NAS файлы, которых нет в sources/",
    )
    action.add_argument(
        "--ingest-inbox",
        action="store_true",
        help="Принять новые видео из inbox/ в videos/<VIDEO_ID>/raw/ и записать паспорт",
    )
    parser.add_argument(
        "--video",
        default=None,
        metavar="VIDEO_ID",
        help="Ограничить одним video_id, например IMG_1043",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="При --pull создать символическую ссылку вместо копии",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать копирование и ссылки без записи на диск",
    )
    args = parser.parse_args(argv)
    if args.symlink and not args.pull:
        parser.error("--symlink используется только вместе с --pull")
    if not any((args.status, args.push, args.pull, args.ingest_inbox)):
        args.status = True
    return args


def iso_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_split():
    if not DATA_SPLIT_PATH.is_file():
        print(f"Не найден {DATA_SPLIT_PATH}", file=sys.stderr)
        return None
    with DATA_SPLIT_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if "nas_storage" not in data or "videos" not in data:
        print(
            "В config/data_split.json нет ключей nas_storage или videos",
            file=sys.stderr,
        )
        return None
    return data


def storage_paths(nas):
    mount_point = Path(nas.get("mount_point") or "/mnt/nas-drive")
    project_root = Path(
        nas.get("project_root") or (mount_point / "CV" / "tabletennis")
    )
    inbox_dir = Path(nas.get("inbox_dir") or (project_root / "inbox"))
    videos_dir = Path(nas.get("videos_dir") or (project_root / "videos"))
    datasets_dir = Path(nas.get("datasets_dir") or (project_root / "datasets"))
    return mount_point, project_root, inbox_dir, videos_dir, datasets_dir


def is_mounted(mount_point):
    if not mount_point.exists():
        return False
    try:
        resolved = mount_point.resolve()
    except OSError:
        resolved = mount_point
    if os.path.ismount(mount_point) or os.path.ismount(resolved):
        return True
    targets = {str(mount_point), str(resolved)}
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in targets:
            return True
    return False


def ancestor_writable(path):
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            return False
        current = parent
    return os.access(current, os.W_OK)


def inspect_mount(mount_point, project_root):
    exists = mount_point.exists()
    mounted = is_mounted(mount_point)
    project_exists = project_root.exists()
    project_accessible = bool(
        mounted and project_exists and os.access(project_root, os.R_OK | os.X_OK)
    )
    if mounted and project_exists:
        writable = os.access(project_root, os.W_OK)
    elif mounted:
        writable = ancestor_writable(project_root)
    else:
        writable = False
    return {
        "exists": exists,
        "mounted": mounted,
        "project_exists": project_exists,
        "project_accessible": project_accessible,
        "writable": writable,
        "reachable": mounted,
    }


def fail_nas_unavailable():
    print(NAS_HINT, file=sys.stderr)
    return 1


def select_videos(videos, video_id):
    if not video_id:
        return videos
    chosen = [item for item in videos if item.get("video_id") == video_id]
    if not chosen:
        known = ", ".join(item.get("video_id", "?") for item in videos) or "—"
        print(
            f"Видео {video_id} нет в config/data_split.json. Известные: {known}",
            file=sys.stderr,
        )
        return None
    return chosen


def repo_local(raw):
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def display_local(raw):
    if not raw:
        return "—"
    return str(raw)


def as_path(raw):
    if not raw:
        return None
    return Path(raw)


def video_nas_dir(video, videos_dir):
    raw = video.get("nas_dir")
    if raw:
        return Path(raw)
    return videos_dir / (video.get("video_id") or "unknown")


def passport_file(video, nas_dir):
    raw = video.get("nas_passport_path")
    if raw:
        return Path(raw)
    return nas_dir / "passport.json"


def iter_slots(video):
    """Слоты raw / sdr / clip: локальный путь, Windows-архив, путь на NAS.

    SDR-слот есть только если задан nas_sdr_path или рабочий path отличается
    от raw_path. Если SDR нет (IMG_0291, GoPro), path считается локальной
    стороной raw.
    """
    raw_local = video.get("raw_path") or None
    work_local = video.get("path") or None
    nas_raw = video.get("nas_raw_path") or None
    nas_sdr = video.get("nas_sdr_path") or None
    win_raw = video.get("win_archive_path") or None

    if raw_local or nas_raw or (work_local and not nas_sdr):
        if raw_local:
            local_raw = raw_local
        elif not nas_sdr:
            local_raw = work_local
        else:
            local_raw = None
        yield {
            "role": "raw",
            "clip_id": None,
            "local_raw": local_raw,
            "win_raw": win_raw,
            "nas_raw": nas_raw,
        }

    sdr_distinct = bool(nas_sdr) or bool(
        work_local and raw_local and work_local != raw_local
    )
    if sdr_distinct:
        yield {
            "role": "sdr",
            "clip_id": None,
            "local_raw": work_local,
            "win_raw": None,
            "nas_raw": nas_sdr,
        }

    for clip in video.get("clips") or []:
        if not isinstance(clip, dict):
            continue
        yield {
            "role": "clip",
            "clip_id": clip.get("clip_id"),
            "local_raw": clip.get("path"),
            "win_raw": clip.get("win_archive_path"),
            "nas_raw": clip.get("nas_path"),
        }


def slot_label(slot):
    if slot.get("clip_id"):
        return f"clip {slot['clip_id']}"
    name = None
    for raw in (slot.get("local_raw"), slot.get("nas_raw"), slot.get("win_raw")):
        if raw:
            name = Path(str(raw)).name
            break
    role = slot.get("role") or "—"
    if name:
        return f"{role} {name}"
    return role


def probe(path):
    if path is None:
        return None
    info = {
        "path": path,
        "exists": False,
        "size": None,
        "kind": "missing",
    }
    try:
        if path.is_symlink():
            info["kind"] = "symlink"
            if not path.exists():
                info["kind"] = "broken-symlink"
                return info
        elif path.is_dir():
            info["kind"] = "dir"
            return info
        elif not path.exists():
            return info
        if path.is_file():
            info["exists"] = True
            if info["kind"] != "symlink":
                info["kind"] = "file"
            info["size"] = int(path.stat().st_size)
            return info
        info["kind"] = "not-a-file"
    except OSError as exc:
        info["kind"] = "error"
        info["error"] = str(exc)
    return info


def format_size(size):
    if size is None:
        return "—"
    value = int(size)
    units = ("B", "KB", "MB", "GB", "TB")
    idx = 0
    tenths = 0
    while value >= 1024 and idx < len(units) - 1:
        tenths = (value % 1024) * 10 // 1024
        value //= 1024
        idx += 1
    if idx == 0:
        return f"{value} B"
    return f"{value}.{tenths} {units[idx]}"


def presence_cell(configured, info):
    if not configured:
        return "—"
    if info is None:
        return "н/д"
    if info["kind"] == "broken-symlink":
        return "BROKEN"
    if info["kind"] == "error":
        return "ERROR"
    if info["kind"] == "dir":
        return "DIR"
    if not info["exists"]:
        return "MISSING"
    size = format_size(info["size"])
    if info["kind"] == "symlink":
        return f"SYMLINK {size}"
    return f"OK {size}"


def pair_status(local_info, win_info, nas_info, reachable):
    if not reachable:
        return "NAS_НЕ_СМОНТИРОВАН"
    local_ok = bool(local_info and local_info["exists"])
    win_ok = bool(win_info and win_info["exists"])
    nas_ok = bool(nas_info and nas_info["exists"])
    source = local_info if local_ok else (win_info if win_ok else None)
    if source and nas_ok:
        if source["size"] == nas_info["size"]:
            return "СОВПАДАЕТ"
        return "РАЗМЕР"
    if local_ok:
        return "ТОЛЬКО_LOCAL"
    if win_ok:
        return "ТОЛЬКО_WIN"
    if nas_ok:
        return "ТОЛЬКО_NAS"
    return "НЕТ"


def collect_rows(videos):
    rows = []
    for video in videos:
        video_id = video.get("video_id") or "?"
        slots = list(iter_slots(video))
        passport_path = as_path(video.get("nas_passport_path"))
        if not slots:
            rows.append(
                {
                    "video_id": video_id,
                    "video": video,
                    "role": "—",
                    "slot_label": "—",
                    "local_disp": "—",
                    "nas_disp": "—",
                    "status": "НЕТ_ПУТЕЙ",
                    "local_path": None,
                    "win_path": None,
                    "nas_path": None,
                    "passport_path": passport_path,
                }
            )
            continue
        for slot in slots:
            local_raw = slot.get("local_raw")
            nas_raw = slot.get("nas_raw")
            win_raw = slot.get("win_raw")
            nas_path = as_path(nas_raw)
            rows.append(
                {
                    "video_id": video_id,
                    "video": video,
                    "role": slot["role"],
                    "clip_id": slot.get("clip_id"),
                    "slot_label": slot_label(slot),
                    "local_raw": local_raw,
                    "local_disp": display_local(local_raw),
                    "win_disp": win_raw or "—",
                    "nas_disp": str(nas_path) if nas_path else "—",
                    "local_path": repo_local(local_raw),
                    "win_path": as_path(win_raw),
                    "nas_path": nas_path,
                    "passport_path": passport_path,
                }
            )
    return rows


def annotate_rows(rows, reachable):
    counts = {
        "СОВПАДАЕТ": 0,
        "РАЗМЕР": 0,
        "ТОЛЬКО_LOCAL": 0,
        "ТОЛЬКО_WIN": 0,
        "ТОЛЬКО_NAS": 0,
        "НЕТ": 0,
        "NAS_НЕ_СМОНТИРОВАН": 0,
        "НЕТ_ПУТЕЙ": 0,
    }
    for row in rows:
        if row.get("status") == "НЕТ_ПУТЕЙ":
            counts["НЕТ_ПУТЕЙ"] += 1
            row["local_cell"] = "—"
            row["win_cell"] = "—"
            row["nas_cell"] = "—"
            row["passport_cell"] = "—"
            continue
        local_info = probe(row["local_path"]) if row["local_path"] else None
        win_info = probe(row["win_path"]) if row["win_path"] else None
        if reachable and row["nas_path"] is not None:
            nas_info = probe(row["nas_path"])
        else:
            nas_info = None
        row["local_info"] = local_info
        row["win_info"] = win_info
        row["nas_info"] = nas_info
        row["local_cell"] = presence_cell(row["local_path"] is not None, local_info)
        row["win_cell"] = presence_cell(row["win_path"] is not None, win_info)
        if not reachable:
            row["nas_cell"] = "н/д" if row["nas_path"] is not None else "—"
        else:
            row["nas_cell"] = presence_cell(row["nas_path"] is not None, nas_info)
        passport_path = row.get("passport_path")
        if not reachable:
            row["passport_cell"] = "н/д"
        elif passport_path is not None and passport_path.is_file():
            row["passport_cell"] = "OK"
        else:
            row["passport_cell"] = "MISSING"
        status = pair_status(local_info, win_info, nas_info, reachable)
        row["status"] = status
        counts[status] = counts.get(status, 0) + 1
    return counts


def print_mount_report(mount_point, project_root, info):
    if not info["exists"]:
        mount_text = "нет такой папки"
    elif info["mounted"]:
        mount_text = "смонтирована"
    else:
        mount_text = "папка есть, но не смонтирована"
    if info["project_accessible"]:
        project_text = "доступен"
    elif info["mounted"] and info["project_exists"]:
        project_text = "есть, но нет доступа на чтение"
    elif info["mounted"]:
        project_text = "ещё не создан"
    else:
        project_text = "недоступен"
    write_text = "да" if info["writable"] else "нет"
    registry = "есть" if PASSPORTS_PATH.is_file() else "нет"
    print(f"точка монтирования: {mount_point} — {mount_text}")
    print(f"каталог проекта:    {project_root} — {project_text}")
    print(f"запись на NAS:      {write_text}")
    print(f"реестр паспортов:   config/video_passports.json — {registry}")
    print()


def print_table(rows):
    headers = ("видео", "слот", "LOCAL", "WIN_DISK", "NAS", "passport", "статус")
    keys = (
        "video_id",
        "slot_label",
        "local_cell",
        "win_cell",
        "nas_cell",
        "passport_cell",
        "status",
    )
    widths = [len(title) for title in headers]
    rendered = []
    for row in rows:
        cells = [str(row.get(key, "—")) for key in keys]
        rendered.append(cells)
        for idx, cell in enumerate(cells):
            widths[idx] = max(widths[idx], len(cell))
    header = "  ".join(title.ljust(widths[idx]) for idx, title in enumerate(headers))
    print(header)
    print("  ".join("-" * width for width in widths))
    for cells in rendered:
        print("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells)))


def print_summary(video_count, counts):
    print()
    print(
        "итого: "
        f"видео {video_count}, "
        f"совпадает {counts.get('СОВПАДАЕТ', 0)}, "
        f"размер отличается {counts.get('РАЗМЕР', 0)}, "
        f"только LOCAL {counts.get('ТОЛЬКО_LOCAL', 0)}, "
        f"только WIN_DISK {counts.get('ТОЛЬКО_WIN', 0)}, "
        f"только NAS {counts.get('ТОЛЬКО_NAS', 0)}, "
        f"нет файла {counts.get('НЕТ', 0)}, "
        f"NAS не смонтирован {counts.get('NAS_НЕ_СМОНТИРОВАН', 0)}"
    )


def same_size(src, dst):
    try:
        if not dst.exists() or not dst.is_file():
            return False
        return int(src.stat().st_size) == int(dst.stat().st_size)
    except OSError:
        return False


def same_inode(src, dst):
    try:
        return src.exists() and dst.exists() and src.samefile(dst)
    except OSError:
        return False


def ensure_video_tree(nas_dir, dry_run):
    for name in TREE_DIRS:
        directory = nas_dir / name
        if dry_run:
            print(f"[dry-run] каталог {directory}")
            continue
        directory.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        joined = ",".join(TREE_DIRS)
        print(f"каталог готов: {nas_dir}/{{{joined}}}")


def resolve_push_source(local_path, win_path, nas_path):
    """Файл для копирования на NAS: сначала sources/, затем Windows-архив."""
    if local_path is not None and local_path.is_file():
        if nas_path is not None and same_inode(local_path, nas_path):
            return None, "symlink-nas"
        return local_path, "LOCAL"
    if win_path is not None and win_path.is_file():
        if nas_path is not None and same_inode(win_path, nas_path):
            return None, "symlink-nas"
        return win_path, "WIN_DISK"
    return None, None


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = int(src.stat().st_size)
    copied = 0
    last_report = 0
    partial = dst.with_name(f".{dst.name}.partial")
    try:
        with src.open("rb") as source, partial.open("wb") as target:
            while True:
                chunk = source.read(COPY_CHUNK)
                if not chunk:
                    break
                target.write(chunk)
                copied += len(chunk)
                if copied - last_report >= COPY_REPORT or copied == total:
                    print(
                        f"  ... {format_size(copied)} / {format_size(total)}",
                        flush=True,
                    )
                    last_report = copied
        os.replace(partial, dst)
    except Exception:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass
        raise
    try:
        shutil.copystat(src, dst, follow_symlinks=True)
    except OSError:
        pass


def push_rows(rows, dry_run):
    errors = 0
    copied = 0
    skipped = 0
    for row in rows:
        local_path = row["local_path"]
        win_path = row["win_path"]
        nas_path = row["nas_path"]
        label = f"{row['video_id']} {row['slot_label']}"
        if nas_path is None:
            print(f"пропуск {label}: на NAS путь не задан")
            skipped += 1
            continue
        source, origin = resolve_push_source(local_path, win_path, nas_path)
        if origin == "symlink-nas":
            print(f"пропуск {label}: локальный путь уже указывает на файл NAS")
            skipped += 1
            continue
        if source is None:
            print(f"пропуск {label}: нет файла в LOCAL и WIN_DISK")
            skipped += 1
            continue
        if same_size(source, nas_path):
            print(
                f"пропуск {label}: размер уже совпадает "
                f"({format_size(source.stat().st_size)}) {nas_path.name}"
            )
            skipped += 1
            continue
        overwrite = nas_path.exists()
        verb = "перезапись" if overwrite else "копирование"
        if dry_run:
            print(f"[dry-run] {verb} ({origin}): {source} -> {nas_path}")
            copied += 1
            continue
        print(f"{verb} ({origin}): {source} -> {nas_path}", flush=True)
        try:
            copy_file(source, nas_path)
        except OSError as exc:
            print(f"ОШИБКА {label}: {exc}", file=sys.stderr)
            errors += 1
            continue
        src_size = int(source.stat().st_size)
        try:
            dst_size = int(nas_path.stat().st_size)
        except OSError as exc:
            print(f"ОШИБКА {label}: не прочитан размер NAS: {exc}", file=sys.stderr)
            errors += 1
            continue
        if src_size != dst_size:
            print(
                f"ОШИБКА {label}: после копирования размер {dst_size} != {src_size}",
                file=sys.stderr,
            )
            errors += 1
            continue
        print(f"готово {label}: {format_size(dst_size)}", flush=True)
        copied += 1
    print(f"push: операций {copied}, пропущено {skipped}, ошибок {errors}")
    return copied, skipped, errors


def pull_rows(rows, symlink, dry_run):
    errors = 0
    done = 0
    skipped = 0
    for row in rows:
        local_path = row["local_path"]
        nas_path = row["nas_path"]
        label = f"{row['video_id']} {row['slot_label']}"
        if local_path is None:
            print(f"пропуск {label}: локальный путь не задан")
            skipped += 1
            continue
        if local_path.exists():
            print(f"пропуск {label}: локально уже есть {row['local_disp']}")
            skipped += 1
            continue
        if nas_path is None or not nas_path.is_file():
            print(f"пропуск {label}: на NAS нет файла {row['nas_disp']}")
            skipped += 1
            continue
        action = "symlink" if symlink else "копирование"
        if dry_run:
            print(f"[dry-run] {action}: {nas_path} -> {row['local_disp']}")
            done += 1
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.is_symlink():
            local_path.unlink()
        if symlink:
            local_path.symlink_to(nas_path)
            print(f"symlink: {row['local_disp']} -> {nas_path}")
            done += 1
            continue
        try:
            copy_file(nas_path, local_path)
        except OSError as exc:
            print(f"ОШИБКА {label}: {exc}", file=sys.stderr)
            errors += 1
            continue
        src_size = int(nas_path.stat().st_size)
        dst_size = int(local_path.stat().st_size)
        if src_size != dst_size:
            print(
                f"ОШИБКА {label}: после копирования размер {dst_size} != {src_size}",
                file=sys.stderr,
            )
            errors += 1
            continue
        print(
            f"копирование: {nas_path.name} -> {row['local_disp']} "
            f"({format_size(dst_size)})"
        )
        done += 1
    mode = "symlink" if symlink else "copy"
    print(f"pull ({mode}): операций {done}, пропущено {skipped}, ошибок {errors}")
    return 1 if errors else 0


def parse_rate(rate):
    if not rate or rate in {"0/0", "N/A"}:
        return None
    try:
        if "/" in str(rate):
            num, den = str(rate).split("/", 1)
            den_value = float(den)
            if den_value == 0:
                return None
            return float(num) / den_value
        return float(rate)
    except (TypeError, ValueError):
        return None


def run_ffprobe(path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,color_space,color_transfer,avg_frame_rate,r_frame_rate,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"ffprobe недоступен для {path}: {exc}", file=sys.stderr)
        return None
    if completed.returncode != 0:
        err = (completed.stderr or "").strip()
        print(f"ffprobe ошибка {path}: {err}", file=sys.stderr)
        return None
    try:
        return json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        print(f"ffprobe вернул не JSON для {path}", file=sys.stderr)
        return None


def blank_probe(size_bytes):
    return {
        "codec_name": None,
        "width": None,
        "height": None,
        "pix_fmt": None,
        "color_space": None,
        "color_transfer": None,
        "fps": None,
        "duration_sec": None,
        "size_bytes": size_bytes,
    }


def clean_text(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "unknown":
        return None
    return text


def probe_media(path):
    """Живые данные файла: ffprobe плюс размер с диска."""
    if path is None or not path.is_file():
        return None
    try:
        size_bytes = int(path.stat().st_size)
    except OSError:
        size_bytes = None
    payload = run_ffprobe(path)
    if not payload:
        return blank_probe(size_bytes)
    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = payload.get("format") or {}
    fps = parse_rate(stream.get("avg_frame_rate"))
    if not fps or fps <= 0:
        fps = parse_rate(stream.get("r_frame_rate"))
    duration = None
    for raw in (fmt.get("duration"), stream.get("duration")):
        parsed = parse_rate(raw) if raw is not None else None
        if parsed and parsed > 0:
            duration = parsed
            break
    width = stream.get("width")
    height = stream.get("height")
    if size_bytes is None and fmt.get("size"):
        try:
            size_bytes = int(fmt["size"])
        except (TypeError, ValueError):
            size_bytes = None
    return {
        "codec_name": clean_text(stream.get("codec_name")),
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "pix_fmt": clean_text(stream.get("pix_fmt")),
        "color_space": clean_text(stream.get("color_space")),
        "color_transfer": clean_text(stream.get("color_transfer")),
        "fps": round(fps, 3) if fps else None,
        "duration_sec": round(duration, 3) if duration else None,
        "size_bytes": size_bytes,
    }


def needs_hdr_tonemap(probe_info):
    transfer = ""
    if probe_info:
        transfer = str(probe_info.get("color_transfer") or "")
    return transfer.casefold() in HDR_TRANSFERS


def existing_file(paths):
    seen = set()
    for path in paths:
        if path is None:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def probe_candidates(video):
    """Порядок: NAS raw, локальный raw, Windows-архив, NAS sdr, локальный sdr."""
    raw_local = repo_local(video.get("raw_path"))
    work_local = repo_local(video.get("path"))
    paths = [
        as_path(video.get("nas_raw_path")),
        raw_local,
        as_path(video.get("win_archive_path")),
        as_path(video.get("nas_sdr_path")),
    ]
    if work_local is not None and work_local != raw_local:
        paths.append(work_local)
    elif raw_local is None:
        paths.append(work_local)
    return paths


def window_record(window):
    return {
        "window_id": window.get("window_id"),
        "start_sec": window.get("start_sec"),
        "end_sec": window.get("end_sec"),
        "role": window.get("role"),
        "dataset_split": window.get("dataset_split"),
    }


def clip_record(clip):
    local_path = repo_local(clip.get("path"))
    win_path = as_path(clip.get("win_archive_path"))
    nas_path = as_path(clip.get("nas_path"))
    local_info = probe(local_path) if local_path else None
    win_info = probe(win_path) if win_path else None
    nas_info = probe(nas_path) if nas_path else None
    size_bytes = None
    for info in (nas_info, local_info, win_info):
        if info and info.get("exists") and info.get("size") is not None:
            size_bytes = info["size"]
            break
    if nas_path is None:
        nas_status = "—"
    elif nas_info and nas_info.get("exists"):
        source_size = None
        for info in (local_info, win_info):
            if info and info.get("exists") and info.get("size") is not None:
                source_size = info["size"]
                break
        if source_size is not None and source_size != nas_info.get("size"):
            nas_status = "SIZE_DIFF"
        else:
            nas_status = "OK"
    else:
        nas_status = "MISSING"
    return {
        "clip_id": clip.get("clip_id"),
        "local_path": clip.get("path"),
        "win_archive_path": clip.get("win_archive_path"),
        "nas_path": clip.get("nas_path"),
        "size_bytes": size_bytes,
        "nas_status": nas_status,
    }


def build_passport(video, probe_info=None):
    if probe_info is None:
        probe_info = probe_media(existing_file(probe_candidates(video)))
    local_path = video.get("path") or video.get("raw_path")
    return {
        "video_id": video.get("video_id"),
        "camera": video.get("camera"),
        "camera_angle": video.get("camera_angle"),
        "players_swapped": video.get("players_swapped"),
        "resolution": video.get("resolution"),
        "fps": video.get("fps"),
        "duration_sec": video.get("duration_sec"),
        "color_profile": video.get("color_profile"),
        "needs_hdr_tonemap": needs_hdr_tonemap(probe_info),
        "probe": probe_info,
        "paths": {
            "nas_raw_path": video.get("nas_raw_path"),
            "nas_sdr_path": video.get("nas_sdr_path"),
            "local_path": local_path,
            "win_archive_path": video.get("win_archive_path"),
            "clips": [clip_record(clip) for clip in video.get("clips") or []],
        },
        "windows": [window_record(item) for item in video.get("windows") or []],
        "updated_at": iso_now(),
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.replace(tmp, path)
    except OSError:
        if path.exists():
            path.unlink()
        os.replace(tmp, path)


def load_passport_registry():
    if not PASSPORTS_PATH.is_file():
        return []
    try:
        data = json.loads(PASSPORTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"не прочитан {PASSPORTS_PATH}: {exc}", file=sys.stderr)
        return []
    items = data.get("passports")
    if isinstance(items, dict):
        return [item for item in items.values() if isinstance(item, dict)]
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    return []


def upsert_passports(new_items):
    by_id = {}
    for item in load_passport_registry():
        video_id = item.get("video_id")
        if video_id:
            by_id[video_id] = item
    for item in new_items:
        video_id = item.get("video_id")
        if video_id:
            by_id[video_id] = item
    ordered = [by_id[key] for key in sorted(by_id)]
    payload = {
        "schema_version": "1.0",
        "updated_at": iso_now(),
        "passports": ordered,
    }
    write_json(PASSPORTS_PATH, payload)
    print(f"реестр паспортов: config/video_passports.json ({len(ordered)})")
    return ordered


def write_video_passport(video, videos_dir, dry_run):
    nas_dir = video_nas_dir(video, videos_dir)
    path = passport_file(video, nas_dir)
    if dry_run:
        print(f"[dry-run] паспорт {path}")
        return {"video_id": video.get("video_id")}
    passport = build_passport(video)
    write_json(path, passport)
    print(f"паспорт: {path}")
    return passport


def push_videos(videos, videos_dir, dry_run):
    errors = 0
    copied = 0
    skipped = 0
    passports = []
    for video in videos:
        nas_dir = video_nas_dir(video, videos_dir)
        ensure_video_tree(nas_dir, dry_run)
        rows = collect_rows([video])
        slot_copied, slot_skipped, slot_errors = push_rows(rows, dry_run)
        copied += slot_copied
        skipped += slot_skipped
        errors += slot_errors
        passport = write_video_passport(video, videos_dir, dry_run)
        if passport is not None:
            passports.append(passport)
    if passports and not dry_run:
        upsert_passports(passports)
    elif dry_run:
        print(
            f"[dry-run] реестр паспортов: config/video_passports.json ({len(passports)})"
        )
    print(
        f"push всего: копий {copied}, пропущено {skipped}, "
        f"ошибок {errors}, паспортов {len(passports)}"
    )
    return 1 if errors else 0


def inbox_video_id(path):
    stem = path.stem
    if stem.lower().endswith("_sdr"):
        stem = stem[: -len("_sdr")]
    return stem


def iter_inbox_videos(inbox_dir):
    if not inbox_dir.is_dir():
        return
    entries = sorted(inbox_dir.iterdir(), key=lambda item: item.name.lower())
    for path in entries:
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        yield path


def build_inbox_passport(video_id, dest, probe_info):
    transfer = None
    if probe_info:
        transfer = probe_info.get("color_transfer")
    width = probe_info.get("width") if probe_info else None
    height = probe_info.get("height") if probe_info else None
    resolution = f"{width}x{height}" if width and height else None
    hdr = needs_hdr_tonemap(probe_info)
    if hdr and transfer:
        color_profile = f"HDR {transfer}"
    else:
        color_profile = transfer
    return {
        "video_id": video_id,
        "camera": None,
        "camera_angle": None,
        "players_swapped": None,
        "resolution": resolution,
        "fps": probe_info.get("fps") if probe_info else None,
        "duration_sec": probe_info.get("duration_sec") if probe_info else None,
        "color_profile": color_profile,
        "needs_hdr_tonemap": hdr,
        "probe": probe_info,
        "paths": {
            "nas_raw_path": str(dest),
            "nas_sdr_path": None,
            "local_path": None,
            "win_archive_path": None,
            "clips": [],
        },
        "windows": [],
        "updated_at": iso_now(),
    }


def ingest_inbox(inbox_dir, videos_dir, video_id, dry_run):
    if not inbox_dir.is_dir():
        print(f"Нет каталога inbox: {inbox_dir}", file=sys.stderr)
        return 1
    files = list(iter_inbox_videos(inbox_dir))
    if video_id:
        files = [path for path in files if inbox_video_id(path) == video_id]
    if not files:
        print(f"inbox пуст: {inbox_dir}")
        return 0
    errors = 0
    passports = []
    for path in files:
        found_id = inbox_video_id(path)
        nas_dir = videos_dir / found_id
        dest = nas_dir / "raw" / path.name
        label = f"{found_id} {path.name}"
        if dest.exists() and not same_size(path, dest):
            print(
                f"ОШИБКА {label}: на NAS уже есть {dest} другого размера",
                file=sys.stderr,
            )
            errors += 1
            continue
        if dry_run:
            print(f"[dry-run] перенос {path} -> {dest}")
            ensure_video_tree(nas_dir, True)
            print(f"[dry-run] паспорт {nas_dir / 'passport.json'}")
            continue
        ensure_video_tree(nas_dir, False)
        if dest.exists() and same_size(path, dest):
            print(f"пропуск {label}: файл уже лежит в {dest}")
            media_path = dest
        else:
            print(f"перенос: {path} -> {dest}", flush=True)
            try:
                shutil.move(str(path), str(dest))
            except OSError as exc:
                print(f"ОШИБКА {label}: {exc}", file=sys.stderr)
                errors += 1
                continue
            media_path = dest
        probe_info = probe_media(media_path)
        passport = build_inbox_passport(found_id, dest, probe_info)
        write_json(nas_dir / "passport.json", passport)
        hdr = "да" if passport["needs_hdr_tonemap"] else "нет"
        print(f"паспорт: {nas_dir / 'passport.json'} (needs_hdr_tonemap={hdr})")
        passports.append(passport)
    if passports:
        upsert_passports(passports)
    print(f"inbox: принято {len(passports)}, ошибок {errors}")
    return 1 if errors else 0


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    data = load_split()
    if data is None:
        return 1

    mount_point, project_root, inbox_dir, videos_dir, _datasets = storage_paths(
        data["nas_storage"]
    )
    mount_info = inspect_mount(mount_point, project_root)

    if args.ingest_inbox:
        print(f"data_split: {DATA_SPLIT_PATH.relative_to(REPO_ROOT)}")
        print_mount_report(mount_point, project_root, mount_info)
        if not mount_info["mounted"] or not mount_info["writable"]:
            return fail_nas_unavailable()
        return ingest_inbox(inbox_dir, videos_dir, args.video, args.dry_run)

    videos = select_videos(data["videos"], args.video)
    if videos is None:
        return 1
    rows = collect_rows(videos)

    if args.status:
        print(f"data_split: {DATA_SPLIT_PATH.relative_to(REPO_ROOT)}")
        print_mount_report(mount_point, project_root, mount_info)
        counts = annotate_rows(rows, mount_info["reachable"])
        print_table(rows)
        print_summary(len(videos), counts)
        return 0

    if not mount_info["mounted"] or not mount_info["writable"]:
        return fail_nas_unavailable()

    if args.push:
        return push_videos(videos, videos_dir, args.dry_run)
    annotate_rows(rows, True)
    return pull_rows(rows, args.symlink, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
