#!/usr/bin/env python3
"""Read-only audit of trained seg models and table-label geometry.

Writes report, GT overlays and inference collages under this directory.
Does not modify datasets, weights, sources or scripts/.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

os.environ["YOLO_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SAMPLES = OUT / "samples"
INFER = OUT / "inference"
FRAMES = OUT / "frames"

NAMES6 = {
    0: "Ball",
    1: "Net",
    2: "Player1",
    3: "Player2",
    4: "TableLeft",
    5: "TableRight",
}
NAMES5 = {
    0: "Ball",
    1: "Net",
    2: "Player1",
    3: "Player2",
    4: "Table",
}

DATASETS = [
    {
        "id": "dataset",
        "root": ROOT / "dataset",
        "layout": "splits",
        "names": NAMES5,
        "note": "5 классов, база train12_ball68",
    },
    {
        "id": "tt_seg_v2_export",
        "root": ROOT / "results/roboflow_pipeline/runs/tt_seg_v2_export/dataset",
        "layout": "splits",
        "names": NAMES6,
        "note": "6 классов, база train13_ttseg_v2",
    },
    {
        "id": "tt_seg_v3_export",
        "root": ROOT / "results/roboflow_pipeline/runs/tt_seg_v3_export/dataset",
        "layout": "splits",
        "names": NAMES6,
        "note": "6 классов, база train14_ttseg_v3",
    },
    {
        "id": "img0291_balls_180_357",
        "root": ROOT / "results/roboflow_pipeline/runs/img0291_balls_180_357",
        "layout": "flat",
        "names": NAMES6,
        "note": "преразметка IMG_0291 (layout roboflow_v2: TableLeft=4, TableRight=5)",
    },
]

MODELS = [
    {
        "id": "train10",
        "run": ROOT / "runs/segment/train10",
        "weights": ROOT / "runs/segment/train10/weights/best.pt",
        "infer": False,
    },
    {
        "id": "train12_ball68",
        "run": ROOT / "runs/segment/train12_ball68",
        "weights": ROOT / "runs/segment/train12_ball68/weights/best.pt",
        "infer": True,
        "imgsz_fallback": 640,
    },
    {
        "id": "train13_ttseg_v2",
        "run": ROOT / "runs/segment/train13_ttseg_v2",
        "weights": ROOT / "runs/segment/train13_ttseg_v2/weights/best.pt",
        "infer": True,
        "imgsz_fallback": 1280,
    },
    {
        "id": "train14_ttseg_v3",
        "run": ROOT / "runs/segment/train14_ttseg_v3",
        "weights": ROOT / "runs/segment/train14_ttseg_v3/weights/best.pt",
        "infer": True,
        "imgsz_fallback": 1280,
    },
]

VIDEOS = [
    ("IMG_0291", ROOT / "sources/IMG_0291.MOV", [60.0, 65.0]),
    ("IMG_0973_sdr", ROOT / "sources/IMG_0973_sdr.mp4", [120.0, 125.0]),
    ("IMG_1043_sdr", ROOT / "sources/IMG_1043_sdr.mp4", [60.0, 65.0]),
    ("GX010025", ROOT / "sources/GX010025.MP4", [25.0, 30.0]),
    ("GX010138", ROOT / "sources/GX010138.MP4", [300.0, 305.0]),
]

COLORS_BGR = {
    "Ball": (0, 255, 255),
    "Net": (255, 255, 0),
    "Player1": (0, 180, 0),
    "Player2": (0, 140, 255),
    "Table": (255, 0, 0),
    "TableLeft": (255, 0, 0),
    "TableRight": (180, 0, 180),
}

FRAME_RE = re.compile(r"frame_(\d+)")
SMALL_AREA = 0.03
SLIVER_WIDTH = 0.06


def f32(v) -> float:
    return float(np.float32(v))


def rnd(v, n=5):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return round(float(v), n)
    return v


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def polygon_area(pts: np.ndarray) -> float:
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0].astype(np.float32)
    y = pts[:, 1].astype(np.float32)
    acc = np.float32(0.0)
    for i in range(pts.shape[0]):
        j = (i + 1) % pts.shape[0]
        acc += x[i] * y[j] - x[j] * y[i]
    return float(abs(acc) * np.float32(0.5))


def parse_label_line(line: str) -> dict | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(float(parts[0]))
    nums = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
    if nums.size == 4:
        cx, cy, w, h = nums
        x1, y1 = cx - w / np.float32(2), cy - h / np.float32(2)
        x2, y2 = cx + w / np.float32(2), cy + h / np.float32(2)
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        kind = "bbox"
    elif nums.size >= 6 and nums.size % 2 == 0:
        pts = nums.reshape(-1, 2)
        kind = "polygon"
    else:
        return None
    xs = pts[:, 0]
    ys = pts[:, 1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    return {
        "cls_id": cls_id,
        "kind": kind,
        "pts": pts,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "width": xmax - xmin,
        "height": ymax - ymin,
        "area": polygon_area(pts),
        "cx": (xmin + xmax) / 2.0,
        "cy": (ymin + ymax) / 2.0,
    }


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        im = cv2.imread(str(path))
        if im is None:
            return None
        h, w = im.shape[:2]
        return int(w), int(h)


def iter_pairs(root: Path, layout: str):
    pairs = []
    missing_images = []
    if layout == "flat":
        splits = [("flat", root / "images", root / "labels")]
    else:
        splits = []
        for split in ("train", "valid", "test"):
            splits.append((split, root / split / "images", root / split / "labels"))
    for split, img_dir, lbl_dir in splits:
        if not lbl_dir.is_dir() and not img_dir.is_dir():
            continue
        labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
        images_by_stem = {}
        if img_dir.is_dir():
            for p in img_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    images_by_stem[p.stem] = p
        seen = set()
        for lp in labels:
            ip = images_by_stem.get(lp.stem)
            if ip is None:
                missing_images.append(str(lp.name))
            pairs.append((split, ip, lp))
            seen.add(lp.stem)
        if img_dir.is_dir():
            for stem, ip in images_by_stem.items():
                if stem not in seen:
                    pairs.append((split, ip, None))
    return pairs, missing_images


def box_of(instances: list[dict]) -> dict | None:
    if not instances:
        return None
    xmin = min(i["xmin"] for i in instances)
    xmax = max(i["xmax"] for i in instances)
    ymin = min(i["ymin"] for i in instances)
    ymax = max(i["ymax"] for i in instances)
    return {
        "n": len(instances),
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "width": xmax - xmin,
        "height": ymax - ymin,
        "area_sum": float(sum(i["area"] for i in instances)),
        "area_max": float(max(i["area"] for i in instances)),
        "width_max": float(max(i["width"] for i in instances)),
        "width_min": float(min(i["width"] for i in instances)),
        "cx": (xmin + xmax) / 2.0,
        "cy": (ymin + ymax) / 2.0,
    }


def y_overlap(a: dict, b: dict) -> float:
    return max(0.0, min(a["ymax"], b["ymax"]) - max(a["ymin"], b["ymin"]))


def percentile_stats(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, dtype=np.float32)
    out = {
        "n": int(a.size),
        "min": rnd(float(a.min())),
        "p10": rnd(float(np.percentile(a, 10))),
        "p50": rnd(float(np.percentile(a, 50))),
        "mean": rnd(float(a.mean())),
        "p90": rnd(float(np.percentile(a, 90))),
        "max": rnd(float(a.max())),
    }
    return out


def width_hist(values: list[float]) -> dict:
    bins = [0.0, 0.2, 0.35, 0.5, 0.7, 1.01]
    labels = ["0-0.20", "0.20-0.35", "0.35-0.50", "0.50-0.70", "0.70-1.00"]
    counts = {k: 0 for k in labels}
    for v in values:
        for i in range(len(labels)):
            if bins[i] <= v < bins[i + 1]:
                counts[labels[i]] += 1
                break
    return counts


def compact_box(b: dict | None) -> dict | None:
    if b is None:
        return None
    return {k: rnd(b[k]) if isinstance(b[k], float) else b[k] for k in b}


def analyze_frame(split: str, image: Path | None, label: Path | None, names: dict) -> dict:
    rec = {
        "split": split,
        "image": None if image is None else image.name,
        "label": None if label is None else label.name,
        "frame_id": None,
        "wh": None,
        "empty": True,
        "counts": Counter(),
        "by_class": defaultdict(list),
    }
    src_name = (image.name if image is not None else label.name if label is not None else "")
    m = FRAME_RE.search(src_name)
    if m:
        rec["frame_id"] = int(m.group(1))
    if image is not None:
        rec["wh"] = image_size(image)
    if label is None or not label.is_file():
        rec["counts"] = {}
        rec["by_class"] = {}
        return rec
    text = label.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        inst = parse_label_line(line)
        if inst is None:
            continue
        name = names.get(inst["cls_id"], f"class_{inst['cls_id']}")
        inst["cls_name"] = name
        rec["by_class"][name].append(inst)
        rec["counts"][name] += 1
    rec["empty"] = sum(rec["counts"].values()) == 0
    rec["counts"] = dict(rec["counts"])
    return rec


def geometry_flags(rec: dict) -> dict:
    lefts = rec["by_class"].get("TableLeft", [])
    rights = rec["by_class"].get("TableRight", [])
    tables = rec["by_class"].get("Table", [])
    nets = rec["by_class"].get("Net", [])
    left = box_of(lefts)
    right = box_of(rights)
    table = box_of(tables)
    net = box_of(nets)
    halves = []
    if left:
        halves.append(left)
    if right:
        halves.append(right)
    union = box_of(lefts + rights) if (lefts or rights) else table
    flags = []
    only_left = bool(left) and not right
    only_right = bool(right) and not left
    both = bool(left) and bool(right)
    if only_left:
        flags.append("only_TableLeft")
    if only_right:
        flags.append("only_TableRight")
    if not left and not right and not table:
        flags.append("no_table")
    small = False
    for inst in lefts + rights + tables:
        if inst["area"] < SMALL_AREA:
            small = True
            break
    if small:
        flags.append("area_lt_0.03")
    sliver = False
    for inst in lefts + rights:
        if inst["width"] < SLIVER_WIDTH:
            sliver = True
            break
    if sliver:
        flags.append("sliver_width_lt_0.06")
    same_side = False
    net_between = None
    net_y_disjoint = None
    cut_gap = None
    order_swapped = None
    net_x_within_union = None
    if both and net:
        same_side = (left["cx"] - net["cx"]) * (right["cx"] - net["cx"]) > 0
        if same_side:
            flags.append("both_halves_same_side_of_net")
        net_between = (min(left["xmax"], right["xmax"]) <= net["cx"] <= max(left["xmin"], right["xmin"])) or (
            min(left["xmin"], right["xmin"]) <= net["cx"] <= max(left["xmax"], right["xmax"])
            and not same_side
        )
        ov = y_overlap(union, net)
        net_h = max(net["height"], 1e-6)
        net_y_disjoint = ov / net_h < 0.05
        if net_y_disjoint:
            flags.append("net_y_disjoint")
    if both:
        # gap between inner edges when left box is actually left of right box
        if left["cx"] <= right["cx"]:
            cut_gap = right["xmin"] - left["xmax"]
            order_swapped = False
        else:
            cut_gap = left["xmin"] - right["xmax"]
            order_swapped = True
            flags.append("TableLeft_cx_right_of_TableRight")
        if abs(cut_gap) <= 0.02:
            flags.append("halves_share_vertical_cut")
    if union and union["width"] < 0.35 and (both or only_left or only_right or table):
        flags.append("narrow_union_width_lt_0.35")
    if union and net and union["width"] > 1e-6:
        net_x_within_union = (net["cx"] - union["xmin"]) / union["width"]
    multi_left = len(lefts) >= 2
    multi_right = len(rights) >= 2
    multi_table = len(tables) >= 2
    if multi_left:
        flags.append("multi_TableLeft")
    if multi_right:
        flags.append("multi_TableRight")
    if multi_table:
        flags.append("multi_Table")
    return {
        "flags": flags,
        "only_left": only_left,
        "only_right": only_right,
        "both": both,
        "left": compact_box(left),
        "right": compact_box(right),
        "table": compact_box(table),
        "net": compact_box(net),
        "union": compact_box(union),
        "same_side": same_side,
        "net_y_disjoint": net_y_disjoint,
        "cut_gap": rnd(cut_gap),
        "order_swapped": order_swapped,
        "net_x_within_union": rnd(net_x_within_union),
        "n_table_left": len(lefts),
        "n_table_right": len(rights),
        "n_table": len(tables),
    }


def summarize_dataset(spec: dict) -> dict:
    pairs, missing_images = iter_pairs(spec["root"], spec["layout"])
    class_counts = Counter()
    kind_counts = Counter()
    n_images = 0
    n_labels = 0
    n_empty = 0
    n_missing_label = 0
    size_counts = Counter()
    frames = []
    per_class_area = defaultdict(list)
    per_class_width = defaultdict(list)
    per_class_xmin = defaultdict(list)
    per_class_xmax = defaultdict(list)
    union_w, union_area, union_xmin, union_xmax = [], [], [], []
    union_w_by_size = defaultdict(list)
    table_w_by_size = defaultdict(list)
    for split, image, label in pairs:
        if image is not None:
            n_images += 1
        if label is None:
            n_missing_label += 1
        else:
            n_labels += 1
        rec = analyze_frame(split, image, label, spec["names"])
        if rec["empty"]:
            n_empty += 1
        if rec["wh"]:
            size_counts[f"{rec['wh'][0]}x{rec['wh'][1]}"] += 1
        for name, insts in rec["by_class"].items():
            class_counts[name] += len(insts)
            for inst in insts:
                kind_counts[inst["kind"]] += 1
                per_class_area[name].append(inst["area"])
                per_class_width[name].append(inst["width"])
                per_class_xmin[name].append(inst["xmin"])
                per_class_xmax[name].append(inst["xmax"])
        geo = geometry_flags(rec)
        if geo["union"]:
            union_w.append(geo["union"]["width"])
            union_area.append(geo["union"]["area_sum"])
            union_xmin.append(geo["union"]["xmin"])
            union_xmax.append(geo["union"]["xmax"])
            if rec["wh"]:
                key = f"{rec['wh'][0]}x{rec['wh'][1]}"
                union_w_by_size[key].append(geo["union"]["width"])
                if geo["table"]:
                    table_w_by_size[key].append(geo["table"]["width"])
                elif geo["union"]:
                    table_w_by_size[key].append(geo["union"]["width"])
        frames.append(
            {
                "split": rec["split"],
                "image": rec["image"],
                "label": rec["label"],
                "frame_id": rec["frame_id"],
                "wh": rec["wh"],
                "counts": rec["counts"],
                **{k: geo[k] for k in (
                    "flags", "only_left", "only_right", "both", "left", "right",
                    "table", "net", "union", "same_side", "net_y_disjoint",
                    "cut_gap", "order_swapped", "net_x_within_union",
                    "n_table_left", "n_table_right", "n_table",
                )},
                "_by_class": rec["by_class"],
                "_image_path": None if image is None else str(image),
            }
        )
    n_both = sum(1 for f in frames if f["both"])
    n_only_l = sum(1 for f in frames if f["only_left"])
    n_only_r = sum(1 for f in frames if f["only_right"])
    n_same = sum(1 for f in frames if f["same_side"])
    n_disjoint = sum(1 for f in frames if f["net_y_disjoint"])
    n_small = sum(1 for f in frames if "area_lt_0.03" in f["flags"])
    n_sliver = sum(1 for f in frames if "sliver_width_lt_0.06" in f["flags"])
    n_cut = sum(1 for f in frames if "halves_share_vertical_cut" in f["flags"])
    n_swap = sum(1 for f in frames if f["order_swapped"])
    n_narrow = sum(1 for f in frames if "narrow_union_width_lt_0.35" in f["flags"])
    n_multi_l = sum(1 for f in frames if f["n_table_left"] >= 2)
    n_no_table = sum(1 for f in frames if "no_table" in f["flags"])
    cls_stats = {}
    for name in sorted(set(list(class_counts) + list(per_class_width))):
        cls_stats[name] = {
            "objects": int(class_counts.get(name, 0)),
            "width": percentile_stats(per_class_width.get(name, [])),
            "area": percentile_stats(per_class_area.get(name, [])),
            "xmin": percentile_stats(per_class_xmin.get(name, [])),
            "xmax": percentile_stats(per_class_xmax.get(name, [])),
        }
    summary = {
        "id": spec["id"],
        "root": rel(spec["root"]),
        "note": spec["note"],
        "names": {str(k): v for k, v in spec["names"].items()},
        "n_images": n_images,
        "n_label_files": n_labels,
        "n_empty_labels": n_empty,
        "n_images_missing_label": n_missing_label,
        "n_labels_missing_image": len(missing_images),
        "image_sizes": dict(size_counts),
        "label_kinds": dict(kind_counts),
        "class_object_counts": {k: int(v) for k, v in sorted(class_counts.items())},
        "class_geometry": cls_stats,
        "table_presence": {
            "frames": len(frames),
            "both_halves": n_both,
            "only_TableLeft": n_only_l,
            "only_TableRight": n_only_r,
            "no_table": n_no_table,
            "both_halves_same_side_of_net": n_same,
            "net_y_disjoint_among_labeled": n_disjoint,
            "any_table_polygon_area_lt_0.03": n_small,
            "any_half_width_lt_0.06": n_sliver,
            "halves_share_vertical_cut": n_cut,
            "TableLeft_right_of_TableRight": n_swap,
            "narrow_union_width_lt_0.35": n_narrow,
            "frames_with_2plus_TableLeft": n_multi_l,
        },
        "union_x": {
            "width": percentile_stats(union_w),
            "area_sum": percentile_stats(union_area),
            "xmin": percentile_stats(union_xmin),
            "xmax": percentile_stats(union_xmax),
            "width_hist": width_hist(union_w),
        },
        "union_width_by_image_size": {k: percentile_stats(v) for k, v in union_w_by_size.items()},
        "frames": frames,
    }
    return summary


def score_frame(f: dict) -> int:
    s = 0
    flags = set(f["flags"])
    if "only_TableLeft" in flags or "only_TableRight" in flags:
        s += 50
    if "both_halves_same_side_of_net" in flags:
        s += 40
    if "net_y_disjoint" in flags:
        s += 25
    if "area_lt_0.03" in flags:
        s += 20
    if "narrow_union_width_lt_0.35" in flags:
        s += 20
    if "sliver_width_lt_0.06" in flags:
        s += 15
    if "multi_TableLeft" in flags or "multi_TableRight" in flags:
        s += 10
    if "TableLeft_cx_right_of_TableRight" in flags:
        s += 8
    return s


def primary_flag(f: dict) -> str:
    order = [
        "only_TableLeft",
        "only_TableRight",
        "both_halves_same_side_of_net",
        "net_y_disjoint",
        "narrow_union_width_lt_0.35",
        "area_lt_0.03",
        "sliver_width_lt_0.06",
        "multi_TableLeft",
        "TableLeft_cx_right_of_TableRight",
    ]
    for name in order:
        if name in f["flags"]:
            return name
    return "typical"


def draw_instances(image: np.ndarray, by_class: dict, title_lines: list[str]) -> np.ndarray:
    canvas = image.copy()
    overlay = canvas.copy()
    draw_order = ["Table", "TableLeft", "TableRight", "Player1", "Player2", "Net", "Ball"]
    seen = set()
    for name in draw_order + sorted(by_class):
        if name in seen:
            continue
        seen.add(name)
        color = COLORS_BGR.get(name, (200, 200, 200))
        for inst in by_class.get(name, []):
            h, w = canvas.shape[:2]
            pts = inst["pts"].astype(np.float32).copy()
            pts[:, 0] *= np.float32(w)
            pts[:, 1] *= np.float32(h)
            poly = np.round(pts).astype(np.int32)
            if name in {"Ball"}:
                cx = int(round(float(inst["cx"]) * w))
                cy = int(round(float(inst["cy"]) * h))
                cv2.circle(canvas, (cx, cy), 8, color, 2)
            else:
                cv2.fillPoly(overlay, [poly], color)
                cv2.polylines(canvas, [poly], True, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
    # redraw contours on top of blend
    for name, insts in by_class.items():
        color = COLORS_BGR.get(name, (200, 200, 200))
        for inst in insts:
            h, w = canvas.shape[:2]
            pts = inst["pts"].astype(np.float32).copy()
            pts[:, 0] *= np.float32(w)
            pts[:, 1] *= np.float32(h)
            poly = np.round(pts).astype(np.int32)
            if name != "Ball":
                cv2.polylines(canvas, [poly], True, color, 2, cv2.LINE_AA)
            if name in {"TableLeft", "TableRight", "Table", "Net"}:
                cx = int(round(float(inst["cx"]) * w))
                cy = int(round(float(inst["cy"]) * h))
                cv2.putText(canvas, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    bar_h = 18 + 18 * len(title_lines)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], bar_h), (0, 0, 0), -1)
    for i, line in enumerate(title_lines):
        cv2.putText(
            canvas, line[:180], (8, 16 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return canvas


def resize_max_w(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    return cv2.resize(img, (max_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def save_samples(datasets: dict) -> list[dict]:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    chosen = []

    def take(ds_id: str, frame: dict, reason: str):
        if len(chosen) >= 8:
            return
        path = frame.get("_image_path")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            return
        flags = ",".join(frame["flags"][:6]) or "none"
        union = frame.get("union") or {}
        title = [
            f"{ds_id} | {reason}",
            f"{frame['image']} | flags: {flags}",
            (
                f"union x=[{union.get('xmin')},{union.get('xmax')}] w={union.get('width')} "
                f"area_sum={union.get('area_sum')} L={frame['n_table_left']} R={frame['n_table_right']} T={frame['n_table']}"
            ),
        ]
        vis = draw_instances(img, frame["_by_class"], title)
        vis = resize_max_w(vis, 1400)
        fname = f"{len(chosen)+1:02d}_{ds_id}_{reason}.jpg"
        out = SAMPLES / fname
        cv2.imwrite(str(out), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        chosen.append({
            "file": rel(out),
            "dataset": ds_id,
            "image": frame["image"],
            "reason": reason,
            "flags": frame["flags"],
            "union": frame.get("union"),
            "left": frame.get("left"),
            "right": frame.get("right"),
            "table": frame.get("table"),
            "net": frame.get("net"),
            "frame_id": frame.get("frame_id"),
            "wh": frame.get("wh"),
        })

    v3 = datasets["tt_seg_v3_export"]["frames"]
    ranked = sorted(v3, key=score_frame, reverse=True)
    used_flags = set()
    v3_taken = 0
    for frame in ranked:
        if v3_taken >= 5:
            break
        flag = primary_flag(frame)
        if flag in used_flags:
            continue
        if score_frame(frame) <= 0:
            continue
        used_flags.add(flag)
        before = len(chosen)
        take("tt_seg_v3_export", frame, flag)
        v3_taken += len(chosen) - before
    for frame in ranked:
        if v3_taken >= 5:
            break
        if score_frame(frame) <= 0:
            continue
        if any(c["image"] == frame["image"] and c["dataset"] == "tt_seg_v3_export" for c in chosen):
            continue
        before = len(chosen)
        take("tt_seg_v3_export", frame, primary_flag(frame) + "_extra")
        v3_taken += len(chosen) - before

    # one wide v3 frame as contrast if we still have room later
    wide = [f for f in v3 if f.get("union") and f["_image_path"]]
    wide.sort(key=lambda f: f["union"]["width"], reverse=True)
    if wide:
        take("tt_seg_v3_export", wide[0], "widest_union")

    ds = datasets["dataset"]["frames"]
    with_table = [f for f in ds if f.get("table") and f["_image_path"] and f.get("wh")]
    by_size = defaultdict(list)
    for f in with_table:
        by_size[f"{f['wh'][0]}x{f['wh'][1]}"].append(f)
    # most common size: median-width frame and narrowest
    if by_size:
        common = max(by_size, key=lambda k: len(by_size[k]))
        group = sorted(by_size[common], key=lambda f: f["table"]["width"])
        mid = group[len(group) // 2]
        take("dataset", mid, f"median_Table_{common}")
        take("dataset", group[0], f"narrowest_Table_{common}")
        # a second resolution if present, widest table
        others = [k for k in by_size if k != common]
        if others:
            other = max(others, key=lambda k: len(by_size[k]))
            group2 = sorted(by_size[other], key=lambda f: f["table"]["width"])
            take("dataset", group2[len(group2) // 2], f"median_Table_{other}")

    balls = datasets["img0291_balls_180_357"]["frames"]
    balls_ranked = sorted(
        [f for f in balls if f["_image_path"]],
        key=score_frame,
        reverse=True,
    )
    if balls_ranked and len(chosen) < 8:
        take("img0291_balls_180_357", balls_ranked[0], primary_flag(balls_ranked[0]))

    if chosen:
        thumbs = []
        for c in chosen:
            im = cv2.imread(str(ROOT / c["file"]))
            if im is None:
                continue
            thumbs.append(cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA))
        if thumbs:
            cols = 4
            rows = (len(thumbs) + cols - 1) // cols
            sheet = np.zeros((rows * 360, cols * 640, 3), dtype=np.uint8)
            for i, th in enumerate(thumbs):
                r, c = divmod(i, cols)
                sheet[r * 360:(r + 1) * 360, c * 640:(c + 1) * 640] = th
            cv2.imwrite(str(SAMPLES / "contact_sheet.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return chosen


def strip_private(frames: list[dict]) -> list[dict]:
    out = []
    for f in frames:
        item = {k: v for k, v in f.items() if not k.startswith("_")}
        out.append(item)
    return out


def match_frame_ids(a_frames: list[dict], b_frames: list[dict]) -> dict:
    a = {}
    b = {}
    for f in a_frames:
        if f.get("frame_id") is None or not f.get("union"):
            continue
        a.setdefault(f["frame_id"], []).append(f)
    for f in b_frames:
        if f.get("frame_id") is None or not f.get("union"):
            continue
        b.setdefault(f["frame_id"], []).append(f)
    shared = sorted(set(a) & set(b))
    dw, dx0, dx1 = [], [], []
    identical_width = 0
    for fid in shared:
        ua = a[fid][0]["union"]
        ub = b[fid][0]["union"]
        dw.append(ua["width"] - ub["width"])
        dx0.append(ua["xmin"] - ub["xmin"])
        dx1.append(ua["xmax"] - ub["xmax"])
        if abs(ua["width"] - ub["width"]) < 0.005 and abs(ua["xmin"] - ub["xmin"]) < 0.005:
            identical_width += 1
    return {
        "n_shared_frame_ids": len(shared),
        "n_a": len(a),
        "n_b": len(b),
        "delta_union_width_a_minus_b": percentile_stats(dw),
        "delta_xmin": percentile_stats(dx0),
        "delta_xmax": percentile_stats(dx1),
        "n_near_identical_union_x": identical_width,
        "shared_frame_ids_sample": shared[:30],
    }


def parse_models() -> list[dict]:
    rows_out = []
    for spec in MODELS:
        args = load_yaml(spec["run"] / "args.yaml")
        csv_path = spec["run"] / "results.csv"
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        def metric(row, key):
            return float(row[key])
        best = max(
            rows,
            key=lambda r: (metric(r, "metrics/mAP50(M)"), metric(r, "metrics/mAP50-95(M)"), -int(float(r["epoch"]))),
        )
        last = rows[-1]

        def pack(row):
            return {
                "epoch": int(float(row["epoch"])),
                "box_mAP50": rnd(metric(row, "metrics/mAP50(B)"), 4),
                "box_mAP50_95": rnd(metric(row, "metrics/mAP50-95(B)"), 4),
                "mask_mAP50": rnd(metric(row, "metrics/mAP50(M)"), 4),
                "mask_mAP50_95": rnd(metric(row, "metrics/mAP50-95(M)"), 4),
                "precision_M": rnd(metric(row, "metrics/precision(M)"), 4),
                "recall_M": rnd(metric(row, "metrics/recall(M)"), 4),
            }

        rows_out.append({
            "id": spec["id"],
            "epochs_planned": args.get("epochs"),
            "epochs_ran": len(rows),
            "imgsz": args.get("imgsz"),
            "data": args.get("data"),
            "model_init": args.get("model"),
            "weights": rel(spec["weights"]),
            "best_by_mask_mAP50": pack(best),
            "last_epoch": pack(last),
        })
    return rows_out


def model_names() -> dict:
    from ultralytics import YOLO

    out = {}
    for spec in MODELS:
        model = YOLO(str(spec["weights"]))
        names = getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            ordered = [names[k] for k in sorted(names)]
        else:
            ordered = list(names)
        out[spec["id"]] = ordered
        del model
    return out


def extract_frame(video: Path, t: float, dest: Path) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not video.exists():
        return False, f"missing video {video}"
    cmd = [
        "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video),
        "-frames:v", "1", "-q:v", "2", str(dest),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timeout"
    if proc.returncode != 0 or not dest.is_file():
        err = (proc.stderr or "")[-500:]
        return False, err
    return True, "ok"


def det_from_result(result) -> list[dict]:
    dets = []
    names = result.names if isinstance(result.names, dict) else {i: n for i, n in enumerate(result.names)}
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    h, w = result.orig_shape
    masks_xy = None
    if result.masks is not None and result.masks.xy is not None:
        masks_xy = result.masks.xy
    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i])
        conf = float(result.boxes.conf[i])
        xyxy = result.boxes.xyxy[i].detach().cpu().numpy().astype(np.float32)
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        if masks_xy is not None and i < len(masks_xy) and len(masks_xy[i]) >= 3:
            pts = np.asarray(masks_xy[i], dtype=np.float32)
            norm = np.stack([pts[:, 0] / np.float32(w), pts[:, 1] / np.float32(h)], axis=1).astype(np.float32)
            area = polygon_area(norm)
            xmin = float(pts[:, 0].min()) / float(w)
            xmax = float(pts[:, 0].max()) / float(w)
            ymin = float(pts[:, 1].min()) / float(h)
            ymax = float(pts[:, 1].max()) / float(h)
            kind = "mask"
            pts_norm = norm
        else:
            area = max(0.0, (x2 - x1) * (y2 - y1) / float(w * h))
            xmin, xmax = x1 / float(w), x2 / float(w)
            ymin, ymax = y1 / float(h), y2 / float(h)
            kind = "bbox"
            pts_norm = None
        dets.append({
            "cls_id": cls_id,
            "cls": names.get(cls_id, str(cls_id)),
            "conf": rnd(conf, 4),
            "kind": kind,
            "xmin": rnd(xmin),
            "xmax": rnd(xmax),
            "ymin": rnd(ymin),
            "ymax": rnd(ymax),
            "width": rnd(xmax - xmin),
            "height": rnd(ymax - ymin),
            "area": rnd(area),
            "_pts": pts_norm,
        })
    return dets


def draw_dets(image: np.ndarray, dets: list[dict], title: str) -> np.ndarray:
    by_class = defaultdict(list)
    for d in dets:
        pts = d.get("_pts")
        if pts is None:
            x1, y1, x2, y2 = d["xmin"], d["ymin"], d["xmax"], d["ymax"]
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        by_class[d["cls"]].append({
            "pts": pts,
            "cx": (d["xmin"] + d["xmax"]) / 2.0,
            "cy": (d["ymin"] + d["ymax"]) / 2.0,
            "cls_name": d["cls"],
        })
    counts = Counter(d["cls"] for d in dets)
    tables = [d for d in dets if d["cls"] in {"Table", "TableLeft", "TableRight"}]
    extra = " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    tw = ""
    if tables:
        xmin = min(d["xmin"] for d in tables)
        xmax = max(d["xmax"] for d in tables)
        tw = f" table_union_x=[{xmin:.3f},{xmax:.3f}] w={xmax-xmin:.3f}"
    return draw_instances(image, by_class, [title, extra + tw])


def run_inference(model_meta: list[dict]) -> dict:
    from ultralytics import YOLO

    FRAMES.mkdir(parents=True, exist_ok=True)
    INFER.mkdir(parents=True, exist_ok=True)
    imgsz_by_id = {m["id"]: int(m["imgsz"]) for m in model_meta}
    loaded = {}
    for spec in MODELS:
        if not spec["infer"]:
            continue
        loaded[spec["id"]] = YOLO(str(spec["weights"]))
    report = {"conf": 0.15, "frames": []}
    for video_id, video, times in VIDEOS:
        for t in times:
            dest = FRAMES / f"{video_id}_t{int(t):04d}.jpg"
            ok, msg = extract_frame(video, t, dest)
            entry = {
                "video_id": video_id,
                "video": rel(video) if video.exists() else str(video),
                "t_sec": t,
                "frame": rel(dest) if ok else None,
                "extract_ok": ok,
                "extract_error": None if ok else msg[-300:],
                "models": {},
            }
            if not ok:
                report["frames"].append(entry)
                print(f"FRAME FAIL {video_id} t={t}: {msg[-200:]}", flush=True)
                continue
            image = cv2.imread(str(dest))
            panels = []
            for spec in MODELS:
                if not spec["infer"]:
                    continue
                model = loaded[spec["id"]]
                imgsz = imgsz_by_id.get(spec["id"]) or spec["imgsz_fallback"]
                result = model.predict(image, conf=0.15, verbose=False, device=0, imgsz=imgsz)[0]
                dets = det_from_result(result)
                counts = dict(Counter(d["cls"] for d in dets))
                table_dets = [d for d in dets if str(d["cls"]).startswith("Table")]
                union = None
                if table_dets:
                    xmin = min(d["xmin"] for d in table_dets)
                    xmax = max(d["xmax"] for d in table_dets)
                    ymin = min(d["ymin"] for d in table_dets)
                    ymax = max(d["ymax"] for d in table_dets)
                    union = {
                        "xmin": rnd(xmin), "xmax": rnd(xmax),
                        "ymin": rnd(ymin), "ymax": rnd(ymax),
                        "width": rnd(xmax - xmin), "height": rnd(ymax - ymin),
                    }
                entry["models"][spec["id"]] = {
                    "imgsz": imgsz,
                    "counts": counts,
                    "table_union": union,
                    "detections": dets,
                }
                vis = draw_dets(image, dets, f"{spec['id']} imgsz={imgsz} conf=0.15")
                for d in dets:
                    d.pop("_pts", None)
                vis = resize_max_w(vis, 960)
                panel_path = INFER / f"{video_id}_t{int(t):04d}_{spec['id']}.jpg"
                cv2.imwrite(str(panel_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                entry["models"][spec["id"]]["image"] = rel(panel_path)
                panels.append(cv2.resize(vis, (640, 360), interpolation=cv2.INTER_AREA))
                print(
                    f"INFER {video_id} t={t} {spec['id']} counts={counts} union={union}",
                    flush=True,
                )
            if panels:
                collage = np.hstack(panels)
                collage_path = INFER / f"{video_id}_t{int(t):04d}_collage.jpg"
                cv2.imwrite(str(collage_path), collage, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                entry["collage"] = rel(collage_path)
            report["frames"].append(entry)
    return report


def hypothesis_block(datasets: dict) -> dict:
    def grab(ds_id: str) -> dict:
        s = datasets[ds_id]
        return {
            "id": ds_id,
            "n_images": s["n_images"],
            "table_presence": s["table_presence"],
            "union_x": s["union_x"],
            "TableLeft": s["class_geometry"].get("TableLeft"),
            "TableRight": s["class_geometry"].get("TableRight"),
            "Table": s["class_geometry"].get("Table"),
        }

    v3 = datasets["tt_seg_v3_export"]
    ds = datasets["dataset"]
    # compare union width on the shared 1280x720 bucket if present
    def p50_width(summary, size_key):
        block = summary["union_width_by_image_size"].get(size_key)
        if not block:
            return None
        return block.get("p50")

    size_keys = sorted(set(ds["union_width_by_image_size"]) | set(v3["union_width_by_image_size"]))
    by_size = {}
    for key in size_keys:
        a = p50_width(ds, key)
        b = p50_width(v3, key)
        ratio = None
        if a and b and a > 0:
            ratio = rnd(b / a, 4)
        by_size[key] = {
            "dataset_union_width_p50": a,
            "v3_union_width_p50": b,
            "v3_over_dataset": ratio,
        }
    matches = {
        "balls_vs_v3": match_frame_ids(
            datasets["img0291_balls_180_357"]["frames"],
            datasets["tt_seg_v3_export"]["frames"],
        ),
        "balls_vs_v2": match_frame_ids(
            datasets["img0291_balls_180_357"]["frames"],
            datasets["tt_seg_v2_export"]["frames"],
        ),
        "v2_vs_v3": match_frame_ids(
            datasets["tt_seg_v2_export"]["frames"],
            datasets["tt_seg_v3_export"]["frames"],
        ),
        "dataset_vs_v3": match_frame_ids(
            datasets["dataset"]["frames"],
            datasets["tt_seg_v3_export"]["frames"],
        ),
    }
    return {
        "code_path": (
            "results/roboflow_pipeline/scripts/prelabel_yolo_seg.py: "
            "если у модели < 6 классов, каждый полигон класса Table (id 4) режется "
            "split_table_polygon вертикалью x=центр Net на TableLeft и TableRight. "
            "Вторая физическая половина попадает в разметку только если она уже была внутри этого полигона Table."
        ),
        "datasets": {
            "dataset": grab("dataset"),
            "tt_seg_v2_export": grab("tt_seg_v2_export"),
            "tt_seg_v3_export": grab("tt_seg_v3_export"),
            "img0291_balls_180_357": grab("img0291_balls_180_357"),
        },
        "width_ratio_by_image_size": by_size,
        "frame_id_matches": matches,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== models csv/args ===", flush=True)
    models = parse_models()
    print("=== model.names ===", flush=True)
    names = model_names()
    for m in models:
        m["model_names"] = names.get(m["id"])
        print(m["id"], m["model_names"], "best", m["best_by_mask_mAP50"], "last", m["last_epoch"], flush=True)

    print("=== labels ===", flush=True)
    datasets = {}
    for spec in DATASETS:
        print("dataset", spec["id"], flush=True)
        datasets[spec["id"]] = summarize_dataset(spec)
        tp = datasets[spec["id"]]["table_presence"]
        ux = datasets[spec["id"]]["union_x"]["width"]
        print(" ", datasets[spec["id"]]["class_object_counts"], tp, "union_w", ux, flush=True)

    print("=== samples ===", flush=True)
    samples = save_samples(datasets)
    for s in samples:
        print(" sample", s["file"], s["reason"], flush=True)

    hypo = hypothesis_block(datasets)

    # drop heavy per-frame geometry from the JSON, keep a compact anomaly index
    compact_datasets = {}
    for ds_id, summary in datasets.items():
        anomalies = []
        for f in summary["frames"]:
            if score_frame(f) <= 0 and "multi_Table" not in f["flags"]:
                continue
            anomalies.append({
                "image": f["image"],
                "split": f["split"],
                "frame_id": f["frame_id"],
                "wh": f["wh"],
                "flags": f["flags"],
                "union": f["union"],
                "left": f["left"],
                "right": f["right"],
                "table": f["table"],
                "net": f["net"],
                "counts": f["counts"],
                "net_x_within_union": f["net_x_within_union"],
                "cut_gap": f["cut_gap"],
            })
        anomalies.sort(key=lambda x: (
            -int("only_TableLeft" in x["flags"] or "only_TableRight" in x["flags"]),
            -int("both_halves_same_side_of_net" in x["flags"]),
            x["image"] or "",
        ))
        compact = {k: v for k, v in summary.items() if k != "frames"}
        compact["n_anomaly_frames"] = len(anomalies)
        compact["anomalies_top"] = anomalies[:40]
        compact_datasets[ds_id] = compact

    print("=== inference ===", flush=True)
    inference = run_inference(models)

    report = {
        "created_for": "model_and_labels_audit_20261108",
        "models": models,
        "datasets": compact_datasets,
        "hypothesis": hypo,
        "samples": samples,
        "inference": inference,
    }
    out_json = OUT / "audit_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE", rel(out_json), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
