#!/usr/bin/env python3
"""Pre-label YOLO detection bbox из train12 seg модели. Дата: 2026-05-26."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from roboflow_client import find_project_root

# Roboflow classes (SSoT scripts/config/roboflow.json)
CLASS_NAMES = ["Ball", "Net", "Player1", "Player2", "Table"]


def xyxy_to_yolo(line_cls: int, xyxy: np.ndarray, img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{line_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def prelabel_images(
    images_dir: Path,
    labels_dir: Path,
    model_path: Path,
    conf: float = 0.03,
) -> dict:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"Нет изображений в {images_dir}")

    stats = {"images": 0, "boxes": 0, "by_class": {name: 0 for name in CLASS_NAMES}, "files": []}
    for img_path in image_paths:
        results = model(str(img_path), conf=conf, verbose=False)[0]
        h, w = results.orig_shape
        lines: list[str] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                continue
            xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)
            lines.append(xyxy_to_yolo(cls_id, xyxy, w, h))
            stats["by_class"][CLASS_NAMES[cls_id]] += 1
            stats["boxes"] += 1

        label_path = labels_dir / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
        stats["files"].append({"image": str(img_path), "label": str(label_path), "boxes": len(lines)})

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "format": "yolo_detection",
        "classes": CLASS_NAMES,
        "confidence": conf,
        "stats": stats,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-label images with train12 detection bboxes")
    parser.add_argument("--run-dir", type=str, required=True, help="runs/<batch>")
    parser.add_argument("--images", type=str, default=None, help="override images dir")
    parser.add_argument("--labels", type=str, default=None, help="override labels dir")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.03)
    args = parser.parse_args()

    root = find_project_root()
    cfg_path = root / "results" / "roboflow_pipeline" / "config" / "pipeline.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / "results" / "roboflow_pipeline" / run_dir

    images_dir = Path(args.images) if args.images else run_dir / "images"
    labels_dir = Path(args.labels) if args.labels else run_dir / "labels"
    model_path = Path(args.model) if args.model else root / cfg["local_model"]

    report = prelabel_images(images_dir, labels_dir, model_path, args.conf)
    report_path = run_dir / "prelabel_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Pre-labeled {report['stats']['images']} images, {report['stats']['boxes']} boxes")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
