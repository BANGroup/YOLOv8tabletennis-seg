#!/usr/bin/env python3
"""Fine-tune YOLOv8-seg from train14 weights. fliplr=0: классы сторон не зеркалятся. Дата: 2026-11-08."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def find_project_root() -> Path:
    """Walk up from script location to repo root (contains runs/segment)."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "runs" / "segment").is_dir():
            return parent
    raise RuntimeError("Project root not found (expected runs/segment)")


def resolve_workers(requested: int | None) -> int:
    if requested is not None and requested >= 0:
        return requested
    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu - 1))


def print_training_summary(save_dir: Path) -> None:
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    results_csv = save_dir / "results.csv"

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Run directory : {save_dir}")
    print(f"best.pt path  : {best_pt}")
    print(f"best.pt exists: {best_pt.is_file()}")
    if best_pt.is_file():
        print(f"best.pt size  : {best_pt.stat().st_size:,} bytes")
    print(f"last.pt exists: {last_pt.is_file()}")

    if results_csv.is_file():
        with results_csv.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            last = rows[-1]
            epoch = last.get("epoch", "?")
            print(f"epochs logged : {len(rows)} (last epoch={epoch})")
            metric_keys = [
                k
                for k in last
                if k.startswith(("metrics/mAP50", "metrics/precision", "metrics/recall"))
                or k.startswith(("train/box", "train/seg", "val/box", "val/seg"))
            ]
            for key in sorted(metric_keys):
                if last.get(key) not in (None, ""):
                    print(f"  {key}: {last[key]}")
        else:
            print("results.csv   : empty")
    else:
        print("results.csv   : not found")
    print("=" * 60)


def run_train(
    model_path: Path,
    data_yaml: Path,
    project: Path,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    fliplr: float = 0.0,
) -> Path:
    os.environ.setdefault("YOLO_OFFLINE", "1")

    from ultralytics import YOLO

    if not model_path.is_file():
        raise FileNotFoundError(f"Base weights not found: {model_path}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    print(f"Base model : {model_path}")
    print(f"Data yaml  : {data_yaml}")
    print(f"Project    : {project}")
    print(f"Name       : {name}")
    print(
        f"Epochs     : {epochs}, imgsz={imgsz}, batch={batch}, "
        f"workers={workers}, fliplr={fliplr}"
    )
    print(f"YOLO_OFFLINE={os.environ.get('YOLO_OFFLINE')}")

    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name=name,
        exist_ok=True,
        flipud=0.0,
        fliplr=fliplr,
        workers=workers,
        amp=False,
        pretrained=False,
    )

    save_dir = project / name
    print_training_summary(save_dir)
    return save_dir


def main() -> int:
    root = find_project_root()
    default_data = root / "results/roboflow_pipeline/runs/tt_seg_v3_export/dataset/data.yaml"
    default_model = root / "runs/segment/train14_ttseg_v3/weights/best.pt"

    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8-seg on tt-seg v3 export (from train14, fliplr off)"
    )
    parser.add_argument("--data", type=Path, default=default_data, help="dataset data.yaml")
    parser.add_argument("--model", type=Path, default=default_model, help="base .pt weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--name", type=str, default="train15_ttseg_v4")
    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.0,
        help="horizontal flip probability (0: TableLeft/Right and Player1/2 stay side-aware)",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=root / "runs/segment",
        help="ultralytics project dir (output: project/name)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="dataloader workers (default: min(8, cpu-1))",
    )
    args = parser.parse_args()

    workers = resolve_workers(args.workers)
    attempts = [
        {"batch": args.batch, "imgsz": args.imgsz, "label": "primary"},
        {"batch": 2, "imgsz": 960, "label": "oom-fallback"},
    ]

    last_error: Exception | None = None
    for idx, cfg in enumerate(attempts):
        if idx == 1 and (cfg["batch"] == args.batch and cfg["imgsz"] == args.imgsz):
            continue
        try:
            if idx > 0:
                print(
                    f"\nRetry after OOM: batch={cfg['batch']}, imgsz={cfg['imgsz']} "
                    f"({cfg['label']})"
                )
            run_train(
                model_path=args.model.resolve(),
                data_yaml=args.data.resolve(),
                project=args.project.resolve(),
                name=args.name,
                epochs=args.epochs,
                imgsz=cfg["imgsz"],
                batch=cfg["batch"],
                workers=workers,
                fliplr=args.fliplr,
            )
            return 0
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                last_error = exc
                print(f"CUDA OOM detected: {exc}", file=sys.stderr)
                continue
            raise
        except Exception as exc:
            last_error = exc
            raise

    if last_error is not None:
        raise last_error
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
