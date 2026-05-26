#!/usr/bin/env python3
"""Sample кадров для Roboflow pipeline. Дата: 2026-05-26."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from roboflow_client import find_project_root


def load_pipeline_config(root: Path) -> dict:
    cfg_path = root / "results" / "roboflow_pipeline" / "config" / "pipeline.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _label_has_ball(label_path: Path) -> bool:
    if not label_path.is_file():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts and parts[0] == "0":
            return True
    return False


def sample_from_dataset(
    dataset_dir: Path,
    output_dir: Path,
    count: int,
    name_pattern: str = "frame_000",
) -> list[dict]:
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    candidates: list[tuple[str, Path]] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        if not _label_has_ball(label_path):
            continue
        stem = label_path.stem
        if name_pattern and name_pattern not in stem:
            continue
        for img_path in images_dir.glob(f"{stem}.*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                candidates.append((stem, img_path))
                break
    if len(candidates) < count:
        raise RuntimeError(f"Найдено {len(candidates)} кадров с Ball, нужно {count}")
    selected = candidates[:count]
    out_images = output_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    frames: list[dict] = []
    for idx, (stem, src) in enumerate(selected):
        dst_name = f"sample_{idx:04d}{src.suffix.lower()}"
        dst = out_images / dst_name
        shutil.copy2(src, dst)
        frames.append(
            {
                "index": idx,
                "source_stem": stem,
                "source_image": str(src),
                "output_image": str(dst),
                "strategy": "dataset_fallback",
            }
        )
    return frames


def _pick_uniform(total: int, max_frames: int, stride: int) -> set[int]:
    indices = set(range(0, total, max(1, stride)))
    if len(indices) > max_frames:
        step = max(1, total // max_frames)
        indices = set(range(0, total, step))
    return set(sorted(indices)[:max_frames])


def sample_from_video(
    video_path: Path,
    output_dir: Path,
    model_path: Path,
    strategy: str,
    max_frames: int,
    sampling_cfg: dict,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> list[dict]:
    out_images = output_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec else total_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    conf = float(sampling_cfg.get("confidence", 0.03))
    low_min = float(sampling_cfg.get("low_conf_min", 0.03))
    low_max = float(sampling_cfg.get("low_conf_max", 0.25))
    missed_max = int(sampling_cfg.get("missed_max_consecutive", 5))
    stride = int(sampling_cfg.get("uniform_stride", 30))

    span = max(1, end_frame - start_frame)
    uniform_set = _pick_uniform(span, max_frames, stride)

    selected: dict[int, dict] = {}
    last_detection: int | None = None
    consecutive_misses = 0
    frame_idx = start_frame
    local_idx = 0

    while frame_idx < end_frame and len(selected) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        local_idx = frame_idx - start_frame
        results = model(frame, conf=conf, verbose=False)[0]

        ball_detected = False
        ball_conf = 0.0
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                ball_detected = True
                ball_conf = float(box.conf[0])
                break

        reasons: list[str] = []
        if strategy in {"uniform", "combined"} and local_idx in uniform_set:
            reasons.append("uniform")
        if strategy in {"missed", "combined"}:
            if last_detection is not None and not ball_detected and consecutive_misses <= missed_max:
                reasons.append("missed")
        if strategy in {"low_conf", "combined"}:
            if ball_detected and low_min <= ball_conf <= low_max:
                reasons.append("low_conf")

        if ball_detected:
            last_detection = frame_idx
            consecutive_misses = 0
        else:
            consecutive_misses += 1

        if reasons and frame_idx not in selected:
            out_name = f"frame_{frame_idx:06d}.jpg"
            out_path = out_images / out_name
            cv2.imwrite(str(out_path), frame)
            selected[frame_idx] = {
                "frame_index": frame_idx,
                "time_sec": float(frame_idx / fps),
                "output_image": str(out_path),
                "ball_detected": ball_detected,
                "ball_confidence": float(ball_conf),
                "strategies": reasons,
            }
        frame_idx += 1

    cap.release()
    return [selected[k] for k in sorted(selected.keys())]


def write_manifest(output_dir: Path, meta: dict) -> Path:
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample frames for Roboflow labeling pipeline")
    parser.add_argument("--video", type=str, default=None, help="Путь к видео")
    parser.add_argument("--from-dataset", action="store_true", help="Взять кадры из dataset/train")
    parser.add_argument("--dataset-dir", type=str, default="dataset/train")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=str, required=True, help="runs/<batch> или images dir")
    parser.add_argument("--strategy", choices=["uniform", "missed", "low_conf", "combined"], default="combined")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_pipeline_config(root)
    output_dir = Path(args.output)
    if output_dir.name == "images":
        run_dir = output_dir.parent
    else:
        run_dir = output_dir
        output_dir = run_dir / "images"

    if args.from_dataset:
        fb = cfg["fallback"]["dataset_sample"]
        dataset_dir = root / (args.dataset_dir or fb["dataset_dir"])
        pattern = fb.get("name_pattern", "frame_000")
        frames = sample_from_dataset(dataset_dir, run_dir, args.count, pattern)
        source = "dataset"
    elif args.video:
        model_path = root / cfg["local_model"]
        frames = sample_from_video(
            root / args.video if not Path(args.video).is_absolute() else Path(args.video),
            run_dir,
            model_path,
            args.strategy,
            min(args.count, cfg["sampling"]["max_frames"]),
            cfg["sampling"],
            args.start,
            args.end,
        )
        source = "video"
    else:
        raise SystemExit("Укажите --video или --from-dataset")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "frame_count": len(frames),
        "frames": frames,
        "output_dir": str(run_dir),
        "images_dir": str(output_dir),
    }
    manifest_path = write_manifest(run_dir, manifest)
    print(f"Sampled {len(frames)} frames -> {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
