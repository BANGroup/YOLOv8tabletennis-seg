#!/usr/bin/env python3
"""Сравнение train12 vs train13 vs train14 на видео (inference only). Дата: 2026-09-01."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from roboflow_client import find_project_root

# train12: 0 Ball, 1 Net, 2 P1, 3 P2, 4 Table
TRAIN12_BALL = 0
TRAIN12_NET = 1
TRAIN12_P1 = 2
TRAIN12_P2 = 3
TRAIN12_TABLE = 4

# train13/train14: 0 Ball, 1 Net, 2 P1, 3 P2, 4 TableLeft, 5 TableRight
TRAIN13_BALL = 0
TRAIN13_NET = 1
TRAIN13_P1 = 2
TRAIN13_P2 = 3
TRAIN13_TABLE_LEFT = 4
TRAIN13_TABLE_RIGHT = 5

MODELS = {
    "train12": {
        "path": "runs/segment/train12_ball68/weights/best.pt",
        "ball_cls": TRAIN12_BALL,
        "net_cls": TRAIN12_NET,
        "p1_cls": TRAIN12_P1,
        "p2_cls": TRAIN12_P2,
        "table_cls_ids": (TRAIN12_TABLE,),
    },
    "train13": {
        "path": "runs/segment/train13_ttseg_v2/weights/best.pt",
        "ball_cls": TRAIN13_BALL,
        "net_cls": TRAIN13_NET,
        "p1_cls": TRAIN13_P1,
        "p2_cls": TRAIN13_P2,
        "table_cls_ids": (TRAIN13_TABLE_LEFT, TRAIN13_TABLE_RIGHT),
    },
    "train14": {
        "path": "runs/segment/train14_ttseg_v3/weights/best.pt",
        "ball_cls": TRAIN13_BALL,
        "net_cls": TRAIN13_NET,
        "p1_cls": TRAIN13_P1,
        "p2_cls": TRAIN13_P2,
        "table_cls_ids": (TRAIN13_TABLE_LEFT, TRAIN13_TABLE_RIGHT),
    },
}

WINDOWS = {
    "A_in_domain": {"start_sec": 45, "end_sec": 180},
    "B_held_out": {"start_sec": 180, "end_sec": 357},
}

STRIDE = 15
PRIMARY_CONF = 0.25
RECALL_CONF = 0.10


def _frame_indices(start_sec: float, end_sec: float, fps: float, stride: int) -> list[int]:
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    return list(range(start_frame, end_frame, max(1, stride)))


def _analyze_frame(result, model_cfg: dict) -> dict:
    ball_cls = model_cfg["ball_cls"]
    net_cls = model_cfg["net_cls"]
    p1_cls = model_cfg["p1_cls"]
    p2_cls = model_cfg["p2_cls"]
    table_ids = set(model_cfg["table_cls_ids"])

    ball_confs: list[float] = []
    has_net = False
    has_table = False
    has_p1 = False
    has_p2 = False

    if result.boxes is None or len(result.boxes) == 0:
        return {
            "frames_with_ball": False,
            "ball_count": 0,
            "best_ball_conf": None,
            "has_net": False,
            "has_table": False,
            "has_player1": False,
            "has_player2": False,
        }

    for idx in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[idx])
        conf = float(result.boxes.conf[idx])
        if cls_id == ball_cls:
            ball_confs.append(conf)
        elif cls_id == net_cls:
            has_net = True
        elif cls_id in table_ids:
            has_table = True
        elif cls_id == p1_cls:
            has_p1 = True
        elif cls_id == p2_cls:
            has_p2 = True

    best_conf = max(ball_confs) if ball_confs else None
    return {
        "frames_with_ball": len(ball_confs) > 0,
        "ball_count": len(ball_confs),
        "best_ball_conf": best_conf,
        "has_net": has_net,
        "has_table": has_table,
        "has_player1": has_p1,
        "has_player2": has_p2,
    }


def _aggregate(frame_stats: list[dict]) -> dict:
    frames = len(frame_stats)
    with_ball = [s for s in frame_stats if s["frames_with_ball"]]
    best_confs = [s["best_ball_conf"] for s in with_ball if s["best_ball_conf"] is not None]
    ball_boxes = sum(s["ball_count"] for s in frame_stats)
    multi_ball = sum(1 for s in frame_stats if s["ball_count"] >= 2)

    return {
        "frames": frames,
        "frames_with_ball": len(with_ball),
        "ball_boxes_total": ball_boxes,
        "mean_best_ball_conf": float(statistics.mean(best_confs)) if best_confs else None,
        "median_best_ball_conf": float(statistics.median(best_confs)) if best_confs else None,
        "frames_with_2plus_balls": multi_ball,
        "frames_with_net": sum(1 for s in frame_stats if s["has_net"]),
        "frames_with_table": sum(1 for s in frame_stats if s["has_table"]),
        "frames_with_player1": sum(1 for s in frame_stats if s["has_player1"]),
        "frames_with_player2": sum(1 for s in frame_stats if s["has_player2"]),
    }


def run_window(
    model: YOLO,
    model_name: str,
    model_cfg: dict,
    cap: cv2.VideoCapture,
    frame_indices: list[int],
    conf: float,
) -> dict:
    frame_stats: list[dict] = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        result = model(frame, conf=conf, verbose=False)[0]
        frame_stats.append(_analyze_frame(result, model_cfg))
    agg = _aggregate(frame_stats)
    agg["model"] = model_name
    agg["confidence"] = conf
    return agg


def print_table(report: dict) -> None:
    header = (
        f"{'model':<8} {'window':<14} {'conf':>5} {'frames':>6} {'ball_fr':>7} "
        f"{'boxes':>6} {'mean_c':>7} {'med_c':>7} {'2+ball':>6} "
        f"{'net':>5} {'table':>5} {'p1':>5} {'p2':>5}"
    )
    print(header)
    print("-" * len(header))
    for window_name in WINDOWS:
        for model_name in MODELS:
            primary = report["results"][window_name][model_name]["conf_0.25"]
            mean_c = primary["mean_best_ball_conf"]
            med_c = primary["median_best_ball_conf"]
            mean_s = f"{mean_c:.3f}" if mean_c is not None else "  -  "
            med_s = f"{med_c:.3f}" if med_c is not None else "  -  "
            print(
                f"{model_name:<8} {window_name:<14} {PRIMARY_CONF:>5.2f} "
                f"{primary['frames']:>6} {primary['frames_with_ball']:>7} "
                f"{primary['ball_boxes_total']:>6} {mean_s:>7} {med_s:>7} "
                f"{primary['frames_with_2plus_balls']:>6} "
                f"{primary['frames_with_net']:>5} {primary['frames_with_table']:>5} "
                f"{primary['frames_with_player1']:>5} {primary['frames_with_player2']:>5}"
            )
    print()
    print("Ball recall pass (conf=0.10):")
    for window_name in WINDOWS:
        for model_name in MODELS:
            recall = report["results"][window_name][model_name]["conf_0.10"]
            print(
                f"  {model_name} {window_name}: "
                f"frames_with_ball={recall['frames_with_ball']}/{recall['frames']}, "
                f"boxes={recall['ball_boxes_total']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare train12 vs train13 vs train14 inference on video")
    parser.add_argument("--video", type=str, default="sources/IMG_0291.MOV")
    parser.add_argument(
        "--output",
        type=str,
        default="results/roboflow_pipeline/runs/compare_train12_train13_train14",
    )
    args = parser.parse_args()

    os.environ.setdefault("YOLO_OFFLINE", "1")
    root = find_project_root()
    video_path = root / args.video if not Path(args.video).is_absolute() else Path(args.video)
    out_dir = (
        Path(args.output)
        if Path(args.output).is_absolute()
        else root / args.output
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    loaded_models: dict[str, YOLO] = {}
    for name, cfg in MODELS.items():
        model_path = root / cfg["path"]
        if not model_path.is_file():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        loaded_models[name] = YOLO(str(model_path))

    report: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "stride": STRIDE,
        "yolo_offline": os.environ.get("YOLO_OFFLINE"),
        "windows": WINDOWS,
        "results": {},
    }

    for window_name, wcfg in WINDOWS.items():
        indices = _frame_indices(wcfg["start_sec"], wcfg["end_sec"], fps, STRIDE)
        report["results"][window_name] = {}
        for model_name, model_cfg in MODELS.items():
            model = loaded_models[model_name]
            primary = run_window(model, model_name, model_cfg, cap, indices, PRIMARY_CONF)
            recall = run_window(model, model_name, model_cfg, cap, indices, RECALL_CONF)
            report["results"][window_name][model_name] = {
                "conf_0.25": primary,
                "conf_0.10": recall,
                "frame_indices_count": len(indices),
            }

    cap.release()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report saved: {report_path}")
    print()
    print_table(report)


if __name__ == "__main__":
    main()
