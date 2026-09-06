#!/usr/bin/env python3
"""Сборка YOLO zip (images/ + labels/) для Roboflow upload. Дата: 2026-05-26."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from roboflow_client import find_project_root, load_pipeline_config, resolve_run_dir

DET_CLASSES = ["Ball", "Net", "Player1", "Player2", "Table"]
SEG_CLASSES = ["TableLeft", "TableRight", "Net", "Player1", "Player2"]
SEG_V2_CLASSES = ["Ball", "Net", "Player1", "Player2", "TableLeft", "TableRight"]
SEG_BALL_CLASS_ID = 5


def _filter_seg_lines(label_text: str) -> str:
    """Оставить только seg-полигоны (class 0-4), исключить Ball bbox (class 5)."""
    kept: list[str] = []
    for line in label_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        cls_id = int(float(parts[0]))
        if cls_id == SEG_BALL_CLASS_ID:
            continue
        if cls_id < 0 or cls_id >= len(SEG_CLASSES):
            continue
        kept.append(line.strip())
    return "\n".join(kept) + ("\n" if kept else "")


def _keep_seg_v2_lines(label_text: str) -> str:
    """Roboflow v2: все 6 классов как polygon lines (включая Ball cls 0)."""
    kept: list[str] = []
    for line in label_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(float(parts[0]))
        if cls_id < 0 or cls_id >= len(SEG_V2_CLASSES):
            continue
        kept.append(line.strip())
    return "\n".join(kept) + ("\n" if kept else "")


def build_yolo_zip(
    images_dir: Path,
    labels_dir: Path,
    zip_path: Path,
    classes: list[str] | None = None,
    label_format: str = "seg",
) -> dict:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if label_format == "seg":
        class_names = classes or SEG_CLASSES
    elif label_format == "seg_v2":
        class_names = classes or SEG_V2_CLASSES
    else:
        class_names = classes or DET_CLASSES

    image_exts = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in image_exts)
    if not images:
        raise RuntimeError(f"Нет изображений в {images_dir}")

    data_yaml = "\n".join(
        [
            "path: .",
            "train: images",
            f"nc: {len(class_names)}",
            f"names: {class_names}",
            "",
        ]
    )

    paired = 0
    missing_labels = 0
    filtered_ball_lines = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", data_yaml)
        for img in images:
            label = labels_dir / f"{img.stem}.txt"
            zf.write(img, arcname=f"images/{img.name}")
            if label.is_file():
                raw = label.read_text(encoding="utf-8")
                if label_format == "seg":
                    before = sum(1 for ln in raw.splitlines() if ln.strip())
                    content = _filter_seg_lines(raw)
                    after = sum(1 for ln in content.splitlines() if ln.strip())
                    filtered_ball_lines += max(0, before - after)
                elif label_format == "seg_v2":
                    content = _keep_seg_v2_lines(raw)
                else:
                    content = raw if raw.endswith("\n") or not raw else raw + "\n"
                zf.writestr(f"labels/{label.name}", content)
                paired += 1
            else:
                missing_labels += 1

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "zip_path": str(zip_path),
        "format": label_format,
        "classes": class_names,
        "images": len(images),
        "labels_paired": paired,
        "missing_labels": missing_labels,
        "filtered_ball_lines": filtered_ball_lines if label_format == "seg" else 0,
        "size_bytes": zip_path.stat().st_size,
    }
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLO upload zip")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="default: run_dir/upload.zip")
    parser.add_argument(
        "--format",
        choices=["seg", "seg_v2", "det"],
        default=None,
        help="seg: 5-class without Ball; seg_v2: 6-class Roboflow v2 with Ball polygons; det: legacy",
    )
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_pipeline_config(root)
    run_dir = resolve_run_dir(args.run_dir, root)

    label_mode = cfg.get("label_mode", "det")
    label_format = args.format or ("seg" if label_mode == "hybrid_seg" else "det")
    classes = cfg.get("classes_seg") if label_format == "seg" else cfg.get("classes")
    if label_format == "seg_v2":
        classes = SEG_V2_CLASSES

    zip_path = Path(args.output) if args.output else run_dir / "upload.zip"
    meta = build_yolo_zip(run_dir / "images", run_dir / "labels", zip_path, classes, label_format)
    meta_path = run_dir / "upload_zip_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Built zip ({label_format}): {zip_path} "
        f"({meta['images']} images, {meta['size_bytes']} bytes)"
    )
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
