#!/usr/bin/env python3
"""Санация осколков TableLeft/TableRight в экспорте tt_seg_v3. Дата: 2026-11-08.

Исходный каталог tt_seg_v3_export/dataset только читается.
Очищенная копия пишется в tt_seg_v3_cleaned/.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "valid", "test")
CLASS_NAMES = ("Ball", "Net", "Player1", "Player2", "TableLeft", "TableRight")
CLS_BALL = 0
CLS_NET = 1
CLS_PLAYER1 = 2
CLS_PLAYER2 = 3
CLS_TABLE_LEFT = 4
CLS_TABLE_RIGHT = 5
TABLE_CLASSES = (CLS_TABLE_LEFT, CLS_TABLE_RIGHT)
PLAYER_CLASSES = (CLS_PLAYER1, CLS_PLAYER2)

SHARD_WIDTH = np.float32(0.08)
SHARD_AREA = np.float32(0.005)
PREVIEW_COUNT = 10
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

VIS_CLASS_BGR: dict[str, tuple[int, int, int]] = {
    "Ball": (0, 255, 255),
    "Net": (255, 255, 0),
    "Player1": (0, 255, 0),
    "Player2": (0, 165, 255),
    "TableLeft": (255, 100, 0),
    "TableRight": (180, 0, 255),
}
VIS_FILL_ALPHA = 0.35


@dataclass
class Ann:
    cls_id: int | None
    line: str
    width: np.float32
    area: np.float32
    parsed: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _shoelace_area(xs: np.ndarray, ys: np.ndarray) -> np.float32:
    cross = np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys, dtype=np.float32)
    return np.float32(np.abs(cross) * np.float32(0.5))


def _parse_line(line: str) -> Ann:
    raw = line.strip()
    if not raw:
        return Ann(None, line, np.float32(0.0), np.float32(0.0), False)
    parts = raw.split()
    try:
        cls_id = int(parts[0])
    except ValueError:
        return Ann(None, raw, np.float32(0.0), np.float32(0.0), False)
    vals = np.asarray(parts[1:], dtype=np.float32)
    if vals.size == 4:
        width = np.float32(vals[2])
        height = np.float32(vals[3])
        return Ann(cls_id, raw, width, np.float32(width * height), True)
    if vals.size < 6 or vals.size % 2 != 0:
        return Ann(cls_id, raw, np.float32(0.0), np.float32(0.0), False)
    xs = vals[0::2]
    ys = vals[1::2]
    width = np.float32(np.max(xs) - np.min(xs))
    return Ann(cls_id, raw, width, _shoelace_area(xs, ys), True)


def _is_table_shard(ann: Ann) -> bool:
    return bool(ann.width < SHARD_WIDTH or ann.area < SHARD_AREA)


def _largest(items: list[Ann]) -> tuple[Ann | None, int]:
    """Оставить один полигон с максимальной площадью. При равенстве — первый."""
    if not items:
        return None, 0
    best = items[0]
    for item in items[1:]:
        if item.area > best.area:
            best = item
    return best, len(items) - 1


def sanitize_annotations(lines: list[str]) -> tuple[list[str], dict[str, int], bool, bool]:
    """Вернуть очищенные строки, счётчики удалений и флаг полного стола."""
    anns = [_parse_line(line) for line in lines if line.strip()]
    removed = {
        "table_shards": 0,
        "table_extras": 0,
        "player_extras": 0,
        "net_micros": 0,
    }
    duplicate_table = False
    kept: list[Ann] = []

    by_cls: dict[int, list[Ann]] = {}
    passthrough: list[Ann] = []
    for ann in anns:
        if not ann.parsed or ann.cls_id is None:
            passthrough.append(ann)
            continue
        by_cls.setdefault(ann.cls_id, []).append(ann)

    for cls_id in TABLE_CLASSES:
        group = by_cls.get(cls_id, [])
        if len(group) >= 2:
            duplicate_table = True
        survivors = [ann for ann in group if not _is_table_shard(ann)]
        removed["table_shards"] += len(group) - len(survivors)
        best, extra = _largest(survivors)
        removed["table_extras"] += extra
        if best is not None:
            kept.append(best)

    for cls_id in PLAYER_CLASSES:
        best, extra = _largest(by_cls.get(cls_id, []))
        removed["player_extras"] += extra
        if best is not None:
            kept.append(best)

    nets = by_cls.get(CLS_NET, [])
    if len(nets) > 1 and any(ann.area < SHARD_AREA for ann in nets):
        best, extra = _largest(nets)
        removed["net_micros"] += extra
        if best is not None:
            kept.append(best)
    else:
        kept.extend(nets)

    for cls_id, group in by_cls.items():
        if cls_id in TABLE_CLASSES or cls_id in PLAYER_CLASSES or cls_id == CLS_NET:
            continue
        kept.extend(group)

    kept_ids = {id(ann) for ann in kept}
    kept_ids.update(id(ann) for ann in passthrough)
    out_lines = [ann.line for ann in anns if id(ann) in kept_ids]

    has_left = any(ann.cls_id == CLS_TABLE_LEFT for ann in kept)
    has_right = any(ann.cls_id == CLS_TABLE_RIGHT for ann in kept)
    return out_lines, removed, has_left and has_right, duplicate_table


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    for path in images_dir.iterdir():
        if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_EXTS:
            return path
    return None


def _reset_output(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for name in ("dataset", "visualized_sample", "quarantine_incomplete_table", "cleaning_report.json"):
        path = out_root / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def _write_label(path: Path, lines: list[str]) -> None:
    text = ("\n".join(lines) + "\n") if lines else ""
    path.write_text(text, encoding="utf-8")


def _contour_int(pts: np.ndarray) -> np.ndarray:
    return np.rint(pts).astype(np.int32).reshape(-1, 1, 2)


def _put_label(image: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, org, font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, font, 0.5, color, 1, cv2.LINE_AA)


def _save_preview(image_path: Path, lines: list[str], dest: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    img_h, img_w = image.shape[:2]
    scale_w = np.float32(img_w)
    scale_h = np.float32(img_h)
    parsed: list[tuple[str, tuple[int, int, int], np.ndarray]] = []
    for line in lines:
        ann = _parse_line(line)
        if not ann.parsed or ann.cls_id is None:
            continue
        parts = ann.line.split()
        vals = np.asarray(parts[1:], dtype=np.float32)
        if vals.size == 4:
            cx, cy, bw, bh = vals
            half = np.float32(0.5)
            x1 = (cx - bw * half) * scale_w
            y1 = (cy - bh * half) * scale_h
            x2 = (cx + bw * half) * scale_w
            y2 = (cy + bh * half) * scale_h
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        else:
            xs = vals[0::2] * scale_w
            ys = vals[1::2] * scale_h
            pts = np.stack((xs, ys), axis=1).astype(np.float32, copy=False)
        name = CLASS_NAMES[ann.cls_id] if 0 <= ann.cls_id < len(CLASS_NAMES) else str(ann.cls_id)
        parsed.append((name, VIS_CLASS_BGR.get(name, (255, 255, 255)), pts))

    overlay = image.copy()
    for _name, color, pts in parsed:
        cv2.fillPoly(overlay, [_contour_int(pts)], color)
    canvas = cv2.addWeighted(overlay, VIS_FILL_ALPHA, image, 1.0 - VIS_FILL_ALPHA, 0.0)
    seen: list[str] = []
    for name, color, pts in parsed:
        cv2.polylines(canvas, [_contour_int(pts)], True, color, 2, cv2.LINE_AA)
        if name not in seen:
            seen.append(name)
    summary = f"{image_path.name} | {', '.join(seen) if seen else 'none'}"
    _put_label(canvas, summary, (8, 22), (255, 255, 255))
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), canvas)


def _sample_indices(count: int, limit: int) -> list[int]:
    if count <= 0 or limit <= 0:
        return []
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [0]
    return [int(round(i * (count - 1) / (limit - 1))) for i in range(limit)]


def _write_data_yaml(dataset_dir: Path) -> None:
    yaml_text = (
        f"path: {dataset_dir.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        "nc: 6\n"
        "names: ['Ball', 'Net', 'Player1', 'Player2', 'TableLeft', 'TableRight']\n"
    )
    (dataset_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def clean_dataset(src: Path, out_root: Path) -> dict:
    if src.resolve() == (out_root / "dataset").resolve():
        raise RuntimeError("отказ писать в исходный датасет")

    _reset_output(out_root)
    dataset_dir = out_root / "dataset"
    quarantine_images = out_root / "quarantine_incomplete_table" / "images"
    quarantine_labels = out_root / "quarantine_incomplete_table" / "labels"
    preview_dir = out_root / "visualized_sample"
    for split in SPLITS:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    quarantine_images.mkdir(parents=True, exist_ok=True)
    quarantine_labels.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    removed_totals = {
        "table_shards": 0,
        "table_extras": 0,
        "player_extras": 0,
        "net_micros": 0,
    }
    class_counts = {name: 0 for name in CLASS_NAMES}
    by_split: dict[str, dict[str, int]] = {}
    quarantined: list[str] = []
    kept_previews: list[tuple[Path, list[str]]] = []
    warnings: list[str] = []
    frames_seen = 0
    frames_kept = 0
    frames_with_duplicate_table = 0

    for split in SPLITS:
        labels_dir = src / split / "labels"
        images_dir = src / split / "images"
        split_stats = {"seen": 0, "kept": 0, "quarantined": 0}
        label_paths = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
        for label_path in label_paths:
            raw_lines = label_path.read_text(encoding="utf-8").splitlines()
            cleaned, removed, table_complete, duplicate_table = sanitize_annotations(raw_lines)
            frames_seen += 1
            split_stats["seen"] += 1
            if duplicate_table:
                frames_with_duplicate_table += 1
            for key, value in removed.items():
                removed_totals[key] += value

            image_path = _find_image(images_dir, label_path.stem)
            frame_name = f"{split}/{image_path.name if image_path is not None else label_path.name}"
            if image_path is None:
                warnings.append(f"нет картинки для {split}/{label_path.name}")

            if not table_complete:
                split_stats["quarantined"] += 1
                quarantined.append(frame_name)
                _write_label(quarantine_labels / label_path.name, cleaned)
                if image_path is not None:
                    shutil.copy2(image_path, quarantine_images / image_path.name)
                continue

            frames_kept += 1
            split_stats["kept"] += 1
            _write_label(dataset_dir / split / "labels" / label_path.name, cleaned)
            if image_path is not None:
                copied = dataset_dir / split / "images" / image_path.name
                shutil.copy2(image_path, copied)
                kept_previews.append((copied, cleaned))
            for line in cleaned:
                ann = _parse_line(line)
                if ann.parsed and ann.cls_id is not None and 0 <= ann.cls_id < len(CLASS_NAMES):
                    class_counts[CLASS_NAMES[ann.cls_id]] += 1
        by_split[split] = split_stats

    _write_data_yaml(dataset_dir)

    preview_names: list[str] = []
    for index in _sample_indices(len(kept_previews), PREVIEW_COUNT):
        image_path, lines = kept_previews[index]
        dest = preview_dir / image_path.name
        _save_preview(image_path, lines, dest)
        preview_names.append(image_path.name)

    report = {
        "source_dataset": str(src),
        "cleaned_dataset": str(dataset_dir),
        "frames_seen": frames_seen,
        "frames_kept": frames_kept,
        "frames_quarantined": len(quarantined),
        "frames_with_duplicate_table_half": frames_with_duplicate_table,
        "quarantined_frames": quarantined,
        "removed": removed_totals,
        "by_split": by_split,
        "cleaned_class_counts": class_counts,
        "preview_frames": preview_names,
        "warnings": warnings,
        "rules": {
            "table_shard": "width < 0.08 or area < 0.005, then max 1 polygon per TableLeft and TableRight",
            "incomplete_table": "missing TableLeft or TableRight after cleaning -> quarantine",
            "players": "max 1 Player1 and 1 Player2, largest area",
            "net": "keep all unless a micro-shard (area < 0.005) is present, then largest only",
            "ball": "unchanged",
        },
    }
    report_path = out_root / "cleaning_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = _repo_root()
    src = root / "results" / "roboflow_pipeline" / "runs" / "tt_seg_v3_export" / "dataset"
    out_root = root / "results" / "roboflow_pipeline" / "runs" / "tt_seg_v3_cleaned"
    if not src.is_dir():
        raise SystemExit(f"нет исходного датасета: {src}")
    report = clean_dataset(src, out_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
