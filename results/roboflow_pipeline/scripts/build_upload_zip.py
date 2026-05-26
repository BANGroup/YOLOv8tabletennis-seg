#!/usr/bin/env python3
"""Сборка YOLO zip (images/ + labels/) для Roboflow upload. Дата: 2026-05-26."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from roboflow_client import find_project_root


def build_yolo_zip(images_dir: Path, labels_dir: Path, zip_path: Path) -> dict:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in image_exts)
    if not images:
        raise RuntimeError(f"Нет изображений в {images_dir}")

    paired = 0
    missing_labels = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            label = labels_dir / f"{img.stem}.txt"
            zf.write(img, arcname=f"images/{img.name}")
            if label.is_file():
                zf.write(label, arcname=f"labels/{label.name}")
                paired += 1
            else:
                missing_labels += 1

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "zip_path": str(zip_path),
        "images": len(images),
        "labels_paired": paired,
        "missing_labels": missing_labels,
        "size_bytes": zip_path.stat().st_size,
    }
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLO upload zip")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="default: run_dir/upload.zip")
    args = parser.parse_args()

    root = find_project_root()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / "results" / "roboflow_pipeline" / run_dir

    zip_path = Path(args.output) if args.output else run_dir / "upload.zip"
    meta = build_yolo_zip(run_dir / "images", run_dir / "labels", zip_path)
    meta_path = run_dir / "upload_zip_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Built zip: {zip_path} ({meta['images']} images, {meta['size_bytes']} bytes)")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
