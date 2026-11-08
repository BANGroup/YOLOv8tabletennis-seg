#!/usr/bin/env python3
"""Sample кадров для Roboflow pipeline. Дата: 2026-05-26."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from roboflow_client import find_project_root, resolve_run_dir


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


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _ball_center_y_norm(box_xyxy: np.ndarray, img_h: int) -> float:
    cy = (float(box_xyxy[1]) + float(box_xyxy[3])) / 2.0
    return cy / float(img_h)


def _ball_speed_px(
    prev_center: tuple[float, float] | None,
    center: tuple[float, float],
) -> float:
    if prev_center is None:
        return 0.0
    dx = center[0] - prev_center[0]
    dy = center[1] - prev_center[1]
    return float(np.hypot(dx, dy))


def _passes_gameplay_gate(
    ball_detected: bool,
    ball_xyxy: np.ndarray | None,
    img_h: int,
    ball_y_min_norm: float,
    ball_speed_px: float,
    ball_speed_thresh: float,
) -> bool:
    if not ball_detected or ball_xyxy is None:
        return False
    in_table_zone = _ball_center_y_norm(ball_xyxy, img_h) > ball_y_min_norm
    fast_ball = ball_speed_px > ball_speed_thresh
    return in_table_zone or fast_ball


def _collect_reasons(
    strategy: str,
    local_idx: int,
    uniform_set: set[int],
    ball_detected: bool,
    ball_conf: float,
    last_detection: int | None,
    consecutive_misses: int,
    low_min: float,
    low_max: float,
    missed_max: int,
    frame_idx: int,
    missing_table_half: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if strategy in {"uniform", "combined"} and local_idx in uniform_set:
        reasons.append("uniform")
    if strategy in {"missed", "combined"}:
        if last_detection is not None and not ball_detected and consecutive_misses <= missed_max:
            reasons.append("missed")
    if strategy in {"low_conf", "combined"}:
        if ball_detected and low_min <= ball_conf <= low_max:
            reasons.append("low_conf")
    if strategy in {"missing_table_half", "combined"} and missing_table_half:
        reasons.append("missing_table_half")
    return reasons


def _candidate_score(reasons: list[str], gameplay_ok: bool) -> float:
    score = float(len(reasons)) * 10.0
    if "low_conf" in reasons:
        score += 2.0
    if "uniform" in reasons:
        score += 1.0
    if "missed" in reasons:
        score += 1.0
    if "missing_table_half" in reasons:
        score += 4.0
    if gameplay_ok:
        score += 5.0
    return score


def _time_to_bin(time_sec: float, start_sec: float, end_sec: float, num_bins: int) -> int:
    span = max(1e-6, end_sec - start_sec)
    rel = (time_sec - start_sec) / span
    rel = min(max(rel, 0.0), 0.999999)
    return min(num_bins - 1, int(rel * num_bins))


_NAMED_FRAME_RE = re.compile(r"^(?P<video>.+)_f(?P<idx>\d+)")
_LEGACY_FRAME_RE = re.compile(r"^frame_(?P<idx>\d+)")
_V3_EXPORT_REL = Path("results/roboflow_pipeline/runs/tt_seg_v3_export/dataset")


def _resolve_video_id(video_path: Path) -> str:
    """Канонический video_id: IMG_0291.MOV -> IMG_0291, IMG_1043_sdr.mp4 -> IMG_1043."""
    stem = Path(video_path).stem
    if stem.lower().endswith("_sdr"):
        stem = stem[: -len("_sdr")]
    return stem


def _parse_frame_identity(name: str) -> tuple[str | None, int | None]:
    base = Path(name).name
    legacy = _LEGACY_FRAME_RE.match(base)
    if legacy:
        return "IMG_0291", int(legacy.group("idx"))
    named = _NAMED_FRAME_RE.match(base)
    if named:
        return named.group("video"), int(named.group("idx"))
    return None, None


def _frame_index_from_name(name: str, target_video_id: str | None = None) -> int | None:
    """Индекс кадра из <VIDEO_ID>_fNNNNNN или legacy frame_NNNNNN (это IMG_0291)."""
    video_id, frame_idx = _parse_frame_identity(name)
    if frame_idx is None:
        return None
    if target_video_id is None or video_id == target_video_id:
        return frame_idx
    return None


def _iter_image_names(directory: Path):
    if not directory.is_dir():
        return
    for img in directory.iterdir():
        if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield img.name


def collect_exclude_indices(
    exclude_dirs: list[Path],
    export_dir: Path | None = None,
    target_video_id: str | None = None,
) -> set[int]:
    """Собрать frame indices уже размеченных кадров, чтобы не нарезать их повторно.

    По умолчанию учитывается tt_seg_v3_export/dataset, если каталог есть на диске.
    """
    excluded: set[int] = set()
    root = find_project_root()
    export_dirs: list[Path] = []
    default_v3 = root / _V3_EXPORT_REL
    if default_v3.is_dir():
        export_dirs.append(default_v3)
    if export_dir is not None:
        resolved = export_dir.resolve()
        if all(resolved != existing.resolve() for existing in export_dirs):
            export_dirs.append(export_dir)

    def _add_from_images(images_dir: Path) -> None:
        for name in _iter_image_names(images_dir):
            idx = _frame_index_from_name(name, target_video_id)
            if idx is not None:
                excluded.add(idx)

    for folder in exclude_dirs:
        images_dir = folder / "images" if (folder / "images").is_dir() else folder
        _add_from_images(images_dir)
    for dataset in export_dirs:
        for split in ("train", "valid", "test"):
            _add_from_images(dataset / split / "images")
    return excluded


def _respects_min_gap(time_sec: float, selected: list[dict], min_gap_sec: float) -> bool:
    gap = float(min_gap_sec)
    for prev in selected:
        if abs(float(time_sec) - float(prev["time_sec"])) < gap:
            return False
    return True


def sample_from_video(
    video_path: Path,
    output_dir: Path,
    model_path: Path,
    strategy: str,
    max_frames: int,
    sampling_cfg: dict,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    min_start_sec: float = 0.0,
    stratified: bool = True,
    require_table_ball: bool = False,
    max_per_bin: int = 1,
    exclude_frame_indices: set[int] | None = None,
    min_gap_sec: float = 0.5,
) -> list[dict]:
    exclude_frame_indices = exclude_frame_indices or set()
    out_images = output_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_start = max(start_sec, min_start_sec)
    end_sec_val = end_sec if end_sec is not None else total_frames / fps
    start_frame = int(effective_start * fps)
    end_frame = min(int(end_sec_val * fps), total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    conf = float(sampling_cfg.get("confidence", 0.03))
    low_min = float(sampling_cfg.get("low_conf_min", 0.03))
    low_max = float(sampling_cfg.get("low_conf_max", 0.25))
    missed_max = int(sampling_cfg.get("missed_max_consecutive", 5))
    stride = int(sampling_cfg.get("uniform_stride", 30))
    ball_y_min_norm = float(sampling_cfg.get("ball_y_min_norm", 0.45))
    ball_speed_thresh = float(sampling_cfg.get("ball_speed_thresh_px", 8.0))
    table_conf_min = float(sampling_cfg.get("table_conf_min", 0.25))
    n_classes = len(model.names) if getattr(model, "names", None) else 0
    video_id = _resolve_video_id(video_path)

    span = max(1, end_frame - start_frame)
    uniform_set = _pick_uniform(span, max_frames, stride)
    num_bins = max_frames

    bin_candidates: list[list[dict]] = [[] for _ in range(num_bins)]
    last_detection: int | None = None
    consecutive_misses = 0
    prev_ball_center: tuple[float, float] | None = None
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        local_idx = frame_idx - start_frame
        time_sec = float(frame_idx / fps)
        img_h, img_w = frame.shape[:2]
        results = model(frame, conf=conf, verbose=False)[0]

        ball_detected = False
        ball_conf = 0.0
        ball_xyxy: np.ndarray | None = None
        has_table_left = False
        has_table_right = False
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf_i = float(box.conf[0])
                if cls_id == 0 and not ball_detected:
                    ball_detected = True
                    ball_conf = conf_i
                    ball_xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)
                if n_classes >= 6 and conf_i >= table_conf_min:
                    if cls_id == 4:
                        has_table_left = True
                    elif cls_id == 5:
                        has_table_right = True
        missing_half = n_classes >= 6 and not (has_table_left and has_table_right)

        ball_center: tuple[float, float] | None = None
        ball_speed_px = 0.0
        if ball_xyxy is not None:
            ball_center = (
                (float(ball_xyxy[0]) + float(ball_xyxy[2])) / 2.0,
                (float(ball_xyxy[1]) + float(ball_xyxy[3])) / 2.0,
            )
            ball_speed_px = _ball_speed_px(prev_ball_center, ball_center)
            prev_ball_center = ball_center

        reasons = _collect_reasons(
            strategy,
            local_idx,
            uniform_set,
            ball_detected,
            ball_conf,
            last_detection,
            consecutive_misses,
            low_min,
            low_max,
            missed_max,
            frame_idx,
            missing_table_half=missing_half,
        )

        if ball_detected:
            last_detection = frame_idx
            consecutive_misses = 0
        else:
            consecutive_misses += 1

        if not reasons:
            frame_idx += 1
            continue

        if frame_idx in exclude_frame_indices:
            frame_idx += 1
            continue

        gameplay_ok = _passes_gameplay_gate(
            ball_detected,
            ball_xyxy,
            img_h,
            ball_y_min_norm,
            ball_speed_px,
            ball_speed_thresh,
        )
        if require_table_ball and not gameplay_ok:
            frame_idx += 1
            continue

        candidate = {
            "frame_index": frame_idx,
            "time_sec": time_sec,
            "ball_detected": ball_detected,
            "ball_confidence": float(ball_conf),
            "strategies": reasons,
            "gameplay_ok": gameplay_ok,
            "score": _candidate_score(reasons, gameplay_ok),
            "_frame": frame.copy(),
        }

        if stratified:
            bin_idx = _time_to_bin(time_sec, effective_start, end_sec_val, num_bins)
            bucket = bin_candidates[bin_idx]
            if len(bucket) < max_per_bin:
                bucket.append(candidate)
            else:
                worst_i = min(range(len(bucket)), key=lambda i: bucket[i]["score"])
                if candidate["score"] > bucket[worst_i]["score"]:
                    bucket[worst_i] = candidate
        else:
            bin_candidates[0].append(candidate)

        frame_idx += 1

    cap.release()

    selected: list[dict] = []
    min_gap = float(min_gap_sec)
    for bin_idx, bucket in enumerate(bin_candidates):
        bucket.sort(key=lambda c: c["score"], reverse=True)
        taken_in_bin = 0
        for cand in bucket:
            if taken_in_bin >= max_per_bin or len(selected) >= max_frames:
                break
            if not _respects_min_gap(float(cand["time_sec"]), selected, min_gap):
                continue
            out_name = f"{video_id}_f{cand['frame_index']:06d}.jpg"
            out_path = out_images / out_name
            cv2.imwrite(str(out_path), cand.pop("_frame"))
            selected.append(
                {
                    "frame_index": cand["frame_index"],
                    "time_sec": cand["time_sec"],
                    "video_id": video_id,
                    "output_image": str(out_path),
                    "ball_detected": cand["ball_detected"],
                    "ball_confidence": cand["ball_confidence"],
                    "strategies": cand["strategies"],
                    "bin_index": bin_idx,
                    "gameplay_ok": cand["gameplay_ok"],
                }
            )
            taken_in_bin += 1
        if len(selected) >= max_frames:
            break

    return sorted(selected, key=lambda f: f["frame_index"])


def write_manifest(output_dir: Path, meta: dict) -> Path:
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def _resolve_video_sampling(cfg: dict, video_arg: str | None) -> dict:
    vs = cfg.get("video_sampling", {})
    key = "IMG_0291"
    if video_arg:
        video_id = _resolve_video_id(Path(video_arg))
        if video_id in vs:
            key = video_id
    return dict(vs.get(key, vs.get("IMG_0291", {})))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample frames for Roboflow labeling pipeline")
    parser.add_argument("--video", type=str, default=None, help="Путь к видео")
    parser.add_argument("--from-dataset", action="store_true", help="Взять кадры из dataset/train")
    parser.add_argument("--dataset-dir", type=str, default="dataset/train")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--output", type=str, required=True, help="runs/<batch> или images dir")
    parser.add_argument(
        "--strategy",
        choices=["uniform", "missed", "low_conf", "missing_table_half", "combined"],
        default=None,
    )
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--min-start-sec", type=float, default=None, help="Минимум времени от начала видео")
    parser.add_argument("--stratified", action="store_true", default=None, help="Стратифицированная выборка по времени")
    parser.add_argument("--no-stratified", action="store_true", help="Отключить стратификацию")
    parser.add_argument(
        "--require-table-ball",
        action="store_true",
        default=None,
        help="Только кадры с мячом в зоне стола (y>0.45) или быстрый мяч",
    )
    parser.add_argument(
        "--exclude-dirs",
        type=str,
        nargs="*",
        default=None,
        help="Run dirs или images/ dirs — frame indices из имён будут исключены",
    )
    parser.add_argument(
        "--exclude-export",
        type=str,
        default=None,
        help="Путь к tt_seg_v2_export/dataset для исключения уже экспортированных кадров",
    )
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_pipeline_config(root)
    video_cfg = _resolve_video_sampling(cfg, args.video)
    sampling_cfg = cfg.get("sampling", {})

    output_dir = Path(args.output)
    if output_dir.name == "images":
        run_dir = output_dir.parent
        if not output_dir.is_absolute():
            run_dir = resolve_run_dir(run_dir, root)
            output_dir = run_dir / "images"
    else:
        run_dir = resolve_run_dir(output_dir, root) if not output_dir.is_absolute() else output_dir
        output_dir = run_dir / "images"

    if args.from_dataset:
        fb = cfg["fallback"]["dataset_sample"]
        dataset_dir = root / (args.dataset_dir or fb["dataset_dir"])
        pattern = fb.get("name_pattern", "frame_000")
        count = args.count or fb.get("count", 20)
        frames = sample_from_dataset(dataset_dir, run_dir, count, pattern)
        source = "dataset"
        sampling_meta = {"mode": "dataset_fallback"}
    elif args.video:
        model_path = root / cfg["local_model"]
        max_frames = args.count or video_cfg.get("max_frames") or sampling_cfg.get("max_frames", 80)
        strategy = args.strategy or video_cfg.get("strategy") or sampling_cfg.get("default_strategy", "combined")
        start_sec = args.start if args.start is not None else float(video_cfg.get("start_sec", 0.0))
        end_sec = args.end if args.end is not None else video_cfg.get("end_sec")
        min_start_sec = (
            args.min_start_sec
            if args.min_start_sec is not None
            else float(video_cfg.get("min_start_sec", start_sec))
        )
        if args.no_stratified:
            stratified = False
        elif args.stratified:
            stratified = True
        else:
            stratified = bool(video_cfg.get("stratified", True))
        require_table_ball = (
            args.require_table_ball
            if args.require_table_ball is not None
            else bool(video_cfg.get("require_table_ball", False))
        )
        exclude_dirs: list[Path] = []
        if args.exclude_dirs:
            for d in args.exclude_dirs:
                p = Path(d)
                exclude_dirs.append(p if p.is_absolute() else root / p)
        export_dir = None
        if args.exclude_export:
            export_dir = Path(args.exclude_export)
            if not export_dir.is_absolute():
                export_dir = root / export_dir
        video_path = root / args.video if not Path(args.video).is_absolute() else Path(args.video)
        video_id = _resolve_video_id(video_path)
        min_gap_sec = float(video_cfg.get("min_gap_sec", sampling_cfg.get("min_gap_sec", 0.5)))
        exclude_indices = collect_exclude_indices(
            exclude_dirs,
            export_dir,
            target_video_id=video_id,
        )
        frames = sample_from_video(
            video_path,
            run_dir,
            model_path,
            strategy,
            max_frames,
            sampling_cfg,
            start_sec,
            end_sec,
            min_start_sec,
            stratified,
            require_table_ball,
            max_per_bin=int(video_cfg.get("max_per_bin", 1)),
            exclude_frame_indices=exclude_indices,
            min_gap_sec=min_gap_sec,
        )
        source = "video"
        times = [f["time_sec"] for f in frames]
        sampling_meta = {
            "mode": "video_stratified" if stratified else "video_sequential",
            "video_id": video_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "min_start_sec": min_start_sec,
            "min_gap_sec": min_gap_sec,
            "stratified": stratified,
            "require_table_ball": require_table_ball,
            "excluded_frame_indices_count": len(exclude_indices),
            "time_sec_min": min(times) if times else None,
            "time_sec_max": max(times) if times else None,
            "bins_covered": len({f.get("bin_index", -1) for f in frames}),
            "bins_total": max_frames,
        }
    else:
        raise SystemExit("Укажите --video или --from-dataset")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "frame_count": len(frames),
        "frames": frames,
        "output_dir": str(run_dir),
        "images_dir": str(output_dir),
        "sampling": sampling_meta,
    }
    manifest_path = write_manifest(run_dir, manifest)
    print(f"Sampled {len(frames)} frames -> {output_dir}")
    if source == "video" and sampling_meta.get("time_sec_min") is not None:
        print(
            f"Time range: {sampling_meta['time_sec_min']:.1f}s - {sampling_meta['time_sec_max']:.1f}s "
            f"(bins {sampling_meta['bins_covered']}/{sampling_meta['bins_total']})"
        )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
