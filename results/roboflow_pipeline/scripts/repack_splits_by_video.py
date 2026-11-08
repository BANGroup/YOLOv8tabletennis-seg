#!/usr/bin/env python3
"""Пересборка YOLO-сплитов по окнам config/data_split.json без удаления исходников.

Без --output (или с --dry-run) только печатает аудит. Копирование — shutil.copy2.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

_NAMED_FRAME_RE = re.compile(r"^(?P<video>.+)_f(?P<idx>\d+)")
_LEGACY_FRAME_RE = re.compile(r"^frame_(?P<idx>\d+)")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_SPLITS = ("train", "valid", "test")
_DEFAULT_NAMES = ["Ball", "Net", "Player1", "Player2", "TableLeft", "TableRight"]
_LEAK_HARD_SEC = 0.5
_LEAK_WARN_SEC = 1.0


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "data_split.json").is_file():
            return parent
    return Path.cwd()


def _resolve_path(raw: str, root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    return (root / path).resolve()


def _parse_frame_identity(name: str) -> tuple[str | None, int | None, bool]:
    """Вернуть (video_id, frame_idx, legacy). legacy frame_* относится к IMG_0291."""
    base = Path(name).name
    legacy = _LEGACY_FRAME_RE.match(base)
    if legacy:
        return "IMG_0291", int(legacy.group("idx")), True
    named = _NAMED_FRAME_RE.match(base)
    if named:
        return named.group("video"), int(named.group("idx")), False
    return None, None, False


def _load_split_config(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Не найден split-config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _video_index(split_cfg: dict) -> dict[str, dict]:
    return {str(video["video_id"]): video for video in split_cfg.get("videos", []) if video.get("video_id")}


def _fps_for(video_id: str | None, videos: dict[str, dict]) -> float:
    if video_id and video_id in videos and videos[video_id].get("fps"):
        return float(videos[video_id]["fps"])
    if video_id == "IMG_0291":
        return 30.0
    return 30.0


def _window_for(video: dict, time_sec: float) -> dict | None:
    windows = list(video.get("windows") or [])
    last_idx = len(windows) - 1
    for idx, window in enumerate(windows):
        start = float(window["start_sec"])
        end = float(window["end_sec"])
        if start <= time_sec < end or (idx == last_idx and start <= time_sec <= end):
            return window
    return None


def _label_rank(label_path: Path | None) -> tuple[int, int]:
    """(число полигонов, число непустых строк). Больше — лучше."""
    if label_path is None or not label_path.is_file():
        return (0, 0)
    polygons = 0
    lines = 0
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        lines += 1
        if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
            polygons += 1
    return (polygons, lines)


def _iter_split_images(dataset_dir: Path):
    split_dirs = [("valid", dataset_dir / "val")] if (dataset_dir / "val" / "images").is_dir() else []
    for split in _SPLITS:
        split_dirs.append((split, dataset_dir / split))
    seen: set[Path] = set()
    for split, split_dir in split_dirs:
        images_dir = split_dir / "images"
        if not images_dir.is_dir():
            continue
        resolved = images_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        labels_dir = split_dir / "labels"
        for image in sorted(images_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            label = labels_dir / f"{image.stem}.txt"
            yield split, image, label if label.is_file() else None


def _read_class_spec(dataset_dir: Path) -> tuple[int, list[str]]:
    path = dataset_dir / "data.yaml"
    if not path.is_file():
        return len(_DEFAULT_NAMES), list(_DEFAULT_NAMES)
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return len(_DEFAULT_NAMES), list(_DEFAULT_NAMES)
    names = data.get("names") or list(_DEFAULT_NAMES)
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda item: int(item))]
    names = [str(name) for name in names]
    nc = int(data.get("nc") or len(names))
    return nc, names


def _blank_counts() -> dict[str, int]:
    return {split: 0 for split in _SPLITS}


def _bump(counts: dict[str, int], split: str | None) -> None:
    if split in counts:
        counts[split] += 1


def _collect(sources: list[Path], videos: dict[str, dict], preserve_existing_v3: bool) -> dict:
    current_counts = _blank_counts()
    chosen: dict[tuple, dict] = {}
    duplicates = 0
    unknown_names = 0

    for source in sources:
        if not source.is_dir():
            raise SystemExit(f"Датасет не найден: {source}")
        for split, image, label in _iter_split_images(source):
            _bump(current_counts, split)
            video_id, frame_idx, legacy = _parse_frame_identity(image.name)
            if video_id is None or frame_idx is None:
                unknown_names += 1
            fps = _fps_for(video_id, videos)
            time_sec = None if frame_idx is None else float(frame_idx) / fps
            strict_split = None
            window_id = None
            if video_id in videos and time_sec is not None:
                window = _window_for(videos[video_id], time_sec)
                if window is not None:
                    strict_split = window.get("dataset_split")
                    window_id = window.get("window_id")
            rank = _label_rank(label)
            preserved = bool(preserve_existing_v3 and legacy and video_id == "IMG_0291")
            if preserved:
                target_split = split
            elif strict_split in _SPLITS:
                target_split = strict_split
            else:
                target_split = split
            record = {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "legacy": legacy,
                "time_sec": time_sec,
                "current_split": split,
                "strict_split": strict_split,
                "window_id": window_id,
                "target_split": target_split,
                "preserved": preserved,
                "image": image,
                "label": label,
                "source": source,
                "rank": rank,
            }
            if video_id is None or frame_idx is None:
                key = ("file", str(image.resolve()))
            else:
                key = (video_id, frame_idx)
            prev = chosen.get(key)
            if prev is None:
                chosen[key] = record
                continue
            duplicates += 1
            if rank > prev["rank"]:
                chosen[key] = record

    records = list(chosen.values())
    target_counts = _blank_counts()
    strict_counts = _blank_counts()
    moved = 0
    preserved_count = 0
    strict_disagreements = 0
    unassigned_strict = 0
    for record in records:
        _bump(target_counts, record["target_split"])
        if record["strict_split"] in _SPLITS:
            _bump(strict_counts, record["strict_split"])
        else:
            unassigned_strict += 1
        if record["target_split"] != record["current_split"]:
            moved += 1
        if record["preserved"]:
            preserved_count += 1
        if record["strict_split"] in _SPLITS and record["strict_split"] != record["current_split"]:
            strict_disagreements += 1

    return {
        "current_counts": current_counts,
        "target_counts": target_counts,
        "strict_counts": strict_counts,
        "records": records,
        "duplicates": duplicates,
        "unknown_names": unknown_names,
        "moved": moved,
        "preserved_count": preserved_count,
        "strict_disagreements": strict_disagreements,
        "unassigned_strict": unassigned_strict,
        "file_count": sum(current_counts.values()),
    }


def _cross_split_pairs(records: list[dict], split_key: str, max_dt: float) -> list[dict]:
    by_video: dict[str, list[dict]] = {}
    for record in records:
        video_id = record.get("video_id")
        if not video_id or record.get("time_sec") is None or record.get(split_key) not in _SPLITS:
            continue
        by_video.setdefault(video_id, []).append(record)
    pairs: list[dict] = []
    for video_id, group in by_video.items():
        group.sort(key=lambda item: float(item["time_sec"]))
        for idx, left in enumerate(group):
            left_t = float(left["time_sec"])
            left_split = left[split_key]
            for right in group[idx + 1 :]:
                dt = float(right["time_sec"]) - left_t
                if dt >= max_dt:
                    break
                right_split = right[split_key]
                if left_split == right_split:
                    continue
                splits = tuple(sorted((left_split, right_split)))
                pairs.append(
                    {
                        "video_id": video_id,
                        "dt_sec": dt,
                        "left_frame": left["frame_idx"],
                        "right_frame": right["frame_idx"],
                        "left_split": left_split,
                        "right_split": right_split,
                        "splits": splits,
                        "train_vs_holdout": "train" in splits and splits != ("train", "train"),
                    }
                )
    pairs.sort(key=lambda item: item["dt_sec"])
    return pairs


def _format_counts(counts: dict[str, int]) -> str:
    parts = [f"{split}={counts.get(split, 0)}" for split in _SPLITS]
    total = sum(counts.get(split, 0) for split in _SPLITS)
    return f"{', '.join(parts)} (всего {total})"


def _summarize_pairs(pairs: list[dict]) -> str:
    if not pairs:
        return "нет"
    hard = sum(1 for pair in pairs if pair["dt_sec"] < _LEAK_HARD_SEC)
    warn = sum(1 for pair in pairs if pair["dt_sec"] < _LEAK_WARN_SEC and pair["train_vs_holdout"])
    return (
        f"{len(pairs)} пар ближе {_LEAK_WARN_SEC:.1f} с между разными сплитами; "
        f"из них < {_LEAK_HARD_SEC:.1f} с: {hard}; "
        f"train vs valid/test < {_LEAK_WARN_SEC:.1f} с: {warn}"
    )


def _format_pair_examples(pairs: list[dict], limit: int = 12) -> list[str]:
    lines: list[str] = []
    for pair in pairs[:limit]:
        lines.append(
            f"  - {pair['video_id']} f{pair['left_frame']} ({pair['left_split']}) ↔ "
            f"f{pair['right_frame']} ({pair['right_split']}), dt={pair['dt_sec']:.3f} с"
        )
    hidden = len(pairs) - limit
    if hidden > 0:
        lines.append(f"  - … ещё {hidden} пар")
    return lines


def build_report(collected: dict, sources: list[Path], preserve_existing_v3: bool, dry_run: bool) -> str:
    records = collected["records"]
    target_pairs = _cross_split_pairs(records, "target_split", _LEAK_WARN_SEC)
    current_pairs = _cross_split_pairs(records, "current_split", _LEAK_WARN_SEC)
    by_video = Counter(record["video_id"] or "unknown" for record in records)
    moved_examples = [record for record in records if record["target_split"] != record["current_split"]]
    moved_examples.sort(key=lambda item: (item["video_id"] or "", item["frame_idx"] or -1))

    lines = [
        "Аудит сплитов YOLO по config/data_split.json",
        f"Режим: {'dry-run, исходники не изменялись' if dry_run else 'копирование в output (исходники не удаляются)'}",
        f"preserve-existing-v3: {preserve_existing_v3} "
        "(legacy frame_* IMG_0291 сохраняют текущий сплит; <VIDEO_ID>_f* идут по окнам)",
        "Источники:",
    ]
    lines.extend(f"  - {source}" for source in sources)
    lines.append(f"Файлов изображений (до склейки дублей): {collected['file_count']}")
    lines.append(f"Уникальных кадров после склейки: {len(records)}")
    lines.append(f"Дубликаты одного (video_id, frame_idx): {collected['duplicates']}")
    lines.append(f"Имена без video/frame: {collected['unknown_names']}")
    lines.append("Кадры по видео: " + ", ".join(f"{key}={value}" for key, value in sorted(by_video.items())))
    lines.append(f"Сейчас на диске: {_format_counts(collected['current_counts'])}")
    lines.append(f"Итоговый сплит (с preserve): {_format_counts(collected['target_counts'])}")
    lines.append(f"Строго по окнам data_split.json: {_format_counts(collected['strict_counts'])}")
    lines.append(f"Без окна data_split.json: {collected['unassigned_strict']}")
    lines.append(f"Legacy IMG_0291 сохранено на месте: {collected['preserved_count']}")
    lines.append(
        f"Переедет фактически: {collected['moved']} "
        f"(строгое окно разошлось с текущим сплитом у {collected['strict_disagreements']} кадров)"
    )
    if moved_examples:
        lines.append("Примеры переезда:")
        for record in moved_examples[:12]:
            lines.append(
                f"  - {record['image'].name}: {record['current_split']} -> {record['target_split']} "
                f"(окно {record['window_id']}, t={record['time_sec']})"
            )
        if len(moved_examples) > 12:
            lines.append(f"  - … ещё {len(moved_examples) - 12}")
    lines.append(f"Утечки итогового сплита: {_summarize_pairs(target_pairs)}")
    lines.extend(_format_pair_examples(target_pairs))
    if current_pairs != target_pairs:
        lines.append(f"Утечки текущего сплита (до переноса): {_summarize_pairs(current_pairs)}")
        lines.extend(_format_pair_examples(current_pairs))
    return "\n".join(lines)


def _write_data_yaml(output: Path, nc: int, names: list[str]) -> None:
    quoted = ", ".join("'" + name.replace("'", "''") + "'" for name in names)
    text = (
        f"path: {output.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        f"nc: {nc}\n"
        f"names: [{quoted}]\n"
    )
    (output / "data.yaml").write_text(text, encoding="utf-8")


def _copy_dataset(output: Path, records: list[dict], nc: int, names: list[str]) -> None:
    for split in _SPLITS:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)
    for record in records:
        split = record["target_split"]
        if split not in _SPLITS:
            continue
        image: Path = record["image"]
        dst_image = output / split / "images" / image.name
        dst_label = output / split / "labels" / f"{image.stem}.txt"
        shutil.copy2(image, dst_image)
        label = record["label"]
        if label is not None and label.is_file():
            shutil.copy2(label, dst_label)
        else:
            dst_label.write_text("", encoding="utf-8")
    _write_data_yaml(output, nc, names)


def _guard_output(output: Path, sources: list[Path]) -> None:
    out = output.resolve()
    for source in sources:
        src = source.resolve()
        if out == src or src in out.parents or out in src.parents:
            raise SystemExit(
                f"Отказ: output {out} пересекается с источником {src}. "
                "Исходный экспорт не перезаписывается."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Аудит и неразрушающая пересборка YOLO-сплитов по config/data_split.json"
    )
    parser.add_argument("--source", required=True, help="Экспортированный датасет YOLO")
    parser.add_argument("--extra-sources", nargs="*", default=[], help="Дополнительные датасеты для склейки")
    parser.add_argument("--output", default=None, help="Куда скопировать пересобранный датасет")
    parser.add_argument("--split-config", default="config/data_split.json")
    parser.add_argument("--dry-run", action="store_true", help="Только аудит, без копирования")
    parser.add_argument(
        "--preserve-existing-v3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Оставить текущий сплит legacy-кадров IMG_0291 (frame_NNNNNN)",
    )
    args = parser.parse_args()

    root = _find_project_root()
    split_path = _resolve_path(args.split_config, root)
    split_cfg = _load_split_config(split_path)
    videos = _video_index(split_cfg)
    sources = [_resolve_path(args.source, root)]
    sources.extend(_resolve_path(item, root) for item in args.extra_sources)

    dry_run = bool(args.dry_run or not args.output)
    collected = _collect(sources, videos, preserve_existing_v3=bool(args.preserve_existing_v3))
    report = build_report(collected, sources, bool(args.preserve_existing_v3), dry_run)
    print(report)
    print(f"split-config: {split_path}")

    if dry_run:
        print("Копирование не выполнялось.")
        return

    output = _resolve_path(args.output, root)
    _guard_output(output, sources)
    nc, names = _read_class_spec(sources[0])
    _copy_dataset(output, collected["records"], nc, names)
    print(f"Скопировано кадров: {len(collected['records'])} -> {output}")
    print(f"data.yaml: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
