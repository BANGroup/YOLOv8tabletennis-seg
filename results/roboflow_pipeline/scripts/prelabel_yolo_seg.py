#!/usr/bin/env python3
"""Pre-label YOLO seg + Ball bbox из train12 seg модели. Дата: 2026-05-26."""

from __future__ import annotations

import argparse
import functools
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from roboflow_client import find_project_root, load_pipeline_config, resolve_run_dir

# train12 model class ids
TRAIN12_BALL = 0
TRAIN12_NET = 1
TRAIN12_PLAYER1 = 2
TRAIN12_PLAYER2 = 3
TRAIN12_TABLE = 4

# merged local / Roboflow seg export class ids
SEG_TABLE_LEFT = 0
SEG_TABLE_RIGHT = 1
SEG_NET = 2
SEG_PLAYER1 = 3
SEG_PLAYER2 = 4
SEG_BALL = 5

SEG_CLASS_NAMES = ["TableLeft", "TableRight", "Net", "Player1", "Player2"]
TRAIN12_TO_SEG = {
    TRAIN12_NET: SEG_NET,
    TRAIN12_PLAYER1: SEG_PLAYER1,
    TRAIN12_PLAYER2: SEG_PLAYER2,
}

# Roboflow v2 export order (tt-seg-img0291 dataset v2)
RFV2_BALL = 0
RFV2_NET = 1
RFV2_PLAYER1 = 2
RFV2_PLAYER2 = 3
RFV2_TABLE_LEFT = 4
RFV2_TABLE_RIGHT = 5
RFV2_CLASS_NAMES = ["Ball", "Net", "Player1", "Player2", "TableLeft", "TableRight"]

# BGR. Ball — жёлтый кружок r=10, чтобы мелкий мяч был виден на превью.
VIS_CLASS_BGR: dict[str, tuple[int, int, int]] = {
    "Ball": (0, 255, 255),
    "Net": (255, 255, 0),
    "Player1": (0, 255, 0),
    "Player2": (0, 165, 255),
    "TableLeft": (255, 100, 0),
    "TableRight": (180, 0, 255),
}
VIS_FILL_ALPHA = 0.35


def bbox_to_poly4_xyxy(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def poly_to_yolo_seg(cls_id: int, poly_xy: np.ndarray, img_w: int, img_h: int) -> str:
    pts = poly_xy.astype(np.float32)
    parts = [str(cls_id)]
    for x, y in pts:
        parts.append(f"{x / img_w:.6f}")
        parts.append(f"{y / img_h:.6f}")
    return " ".join(parts)


def xyxy_to_yolo_bbox(cls_id: int, xyxy: np.ndarray, img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _clip_polygon_half_plane(
    poly: np.ndarray,
    keep_left: bool,
    split_x: float,
) -> np.ndarray | None:
    """Sutherland-Hodgman clip polygon against vertical line x=split_x."""
    if poly.shape[0] < 3:
        return None

    def inside(p: np.ndarray) -> bool:
        return p[0] <= split_x if keep_left else p[0] >= split_x

    def intersect(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        dx = p2[0] - p1[0]
        if abs(dx) < 1e-6:
            return np.array([split_x, p1[1]], dtype=np.float32)
        t = (split_x - p1[0]) / dx
        t = float(np.clip(t, 0.0, 1.0))
        y = p1[1] + t * (p2[1] - p1[1])
        return np.array([split_x, y], dtype=np.float32)

    output: list[np.ndarray] = []
    n = poly.shape[0]
    for i in range(n):
        curr = poly[i]
        prev = poly[i - 1]
        curr_in = inside(curr)
        prev_in = inside(prev)
        if curr_in:
            if not prev_in:
                output.append(intersect(prev, curr))
            output.append(curr)
        elif prev_in:
            output.append(intersect(prev, curr))

    if len(output) < 3:
        return None
    return np.array(output, dtype=np.float32)


_TABLE_SHARD_WIDTH_FRAC = np.float32(0.08)
_TABLE_SHARD_AREA_FRAC = np.float32(0.005)


def _polygon_width_x(poly: np.ndarray) -> np.float32:
    xs = poly[:, 0]
    return np.float32(np.max(xs) - np.min(xs))


def _polygon_area(poly: np.ndarray) -> np.float32:
    xs = poly[:, 0]
    ys = poly[:, 1]
    cross = np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys, dtype=np.float32)
    return np.float32(np.abs(cross) * np.float32(0.5))


def _is_table_shard(poly: np.ndarray, img_w: int, img_h: int) -> bool:
    """Узкий осколок вдоль сетки: ширина по X < 0.08 кадра или площадь < 0.005 кадра."""
    if poly.shape[0] < 3:
        return True
    min_width = np.float32(img_w) * _TABLE_SHARD_WIDTH_FRAC
    min_area = np.float32(img_w) * np.float32(img_h) * _TABLE_SHARD_AREA_FRAC
    return bool(_polygon_width_x(poly) < min_width or _polygon_area(poly) < min_area)


def _reject_table_shard(
    poly: np.ndarray | None,
    img_w: int,
    img_h: int,
) -> np.ndarray | None:
    if poly is None or poly.shape[0] < 3:
        return None
    if _is_table_shard(poly, img_w, img_h):
        return None
    return poly


def _largest_table_polygon(
    polys: list[np.ndarray],
    img_w: int,
    img_h: int,
) -> np.ndarray | None:
    """Один полигон половины стола: осколки отброшены, остаётся крупнейший по площади."""
    best: np.ndarray | None = None
    best_area = np.float32(-1.0)
    for poly in polys:
        kept = _reject_table_shard(poly, img_w, img_h)
        if kept is None:
            continue
        area = _polygon_area(kept)
        if best is None or area > best_area:
            best = kept
            best_area = area
    return best


def split_table_polygon(
    poly_xy: np.ndarray,
    split_x: float,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Split table polygon into TableLeft / TableRight by vertical line at split_x.

    Узкая половина (ширина по X < 0.08 * img_w или площадь < 0.005 * img_w * img_h)
    отбрасывается как осколок вдоль сетки.
    """
    poly = poly_xy.astype(np.float32)
    left = _clip_polygon_half_plane(poly, keep_left=True, split_x=split_x)
    right = _clip_polygon_half_plane(poly, keep_left=False, split_x=split_x)
    max_x = np.float32(img_w)
    max_y = np.float32(img_h)

    if left is not None and left.shape[0] >= 3:
        left[:, 0] = np.clip(left[:, 0], np.float32(0.0), max_x)
        left[:, 1] = np.clip(left[:, 1], np.float32(0.0), max_y)
    else:
        left = None

    if right is not None and right.shape[0] >= 3:
        right[:, 0] = np.clip(right[:, 0], np.float32(0.0), max_x)
        right[:, 1] = np.clip(right[:, 1], np.float32(0.0), max_y)
    else:
        right = None

    return _reject_table_shard(left, img_w, img_h), _reject_table_shard(right, img_w, img_h)


def resolve_split_x(
    net_boxes: list[np.ndarray],
    img_w: int,
) -> float:
    if net_boxes:
        centers = [float((b[0] + b[2]) / 2.0) for b in net_boxes]
        return float(np.mean(centers))
    return float(img_w) / 2.0


def _bbox_area(xyxy: np.ndarray) -> float:
    return max(0.0, float(xyxy[2] - xyxy[0])) * max(0.0, float(xyxy[3] - xyxy[1]))


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _dedupe_players_same_side(
    players: list[tuple[np.ndarray, np.ndarray]],
    iou_thresh: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Оставить крупнейший bbox при IoU > thresh на одной стороне."""
    players = sorted(players, key=lambda p: _bbox_area(p[0]), reverse=True)
    kept: list[tuple[np.ndarray, np.ndarray]] = []
    for xyxy, poly in players:
        if any(_bbox_iou(xyxy, k[0]) > iou_thresh for k in kept):
            continue
        kept.append((xyxy, poly))
    return kept


def _assign_player_side(center_x: float, split_x: float, p1_cls: int, p2_cls: int, players_swapped: bool) -> int:
    on_left = center_x < split_x
    if players_swapped:
        return p2_cls if on_left else p1_cls
    return p1_cls if on_left else p2_cls


def _assign_player_seg_cls(xyxy: np.ndarray, split_x: float, players_swapped: bool = False) -> int:
    center_x = (float(xyxy[0]) + float(xyxy[2])) / 2.0
    return _assign_player_side(center_x, split_x, SEG_PLAYER1, SEG_PLAYER2, players_swapped)


def _assign_player_rfv2_cls(xyxy: np.ndarray, split_x: float, players_swapped: bool = False) -> int:
    center_x = (float(xyxy[0]) + float(xyxy[2])) / 2.0
    return _assign_player_side(center_x, split_x, RFV2_PLAYER1, RFV2_PLAYER2, players_swapped)


_NAMED_FRAME_RE = re.compile(r"^(?P<video>.+)_f(?P<idx>\d+)")
_LEGACY_FRAME_RE = re.compile(r"^frame_(?P<idx>\d+)")


def _video_id_from_image_name(name: str) -> str | None:
    base = Path(name).name
    if _LEGACY_FRAME_RE.match(base):
        return "IMG_0291"
    named = _NAMED_FRAME_RE.match(base)
    if named:
        return named.group("video")
    upper = base.upper()
    for token in ("IMG_1043", "IMG_0973", "IMG_0291"):
        if upper.startswith(token):
            return token
    return None


def _load_split_config(root: Path | None = None) -> dict:
    root = root or find_project_root()
    rel = "config/data_split.json"
    pipe_path = root / "results" / "roboflow_pipeline" / "config" / "pipeline.json"
    if pipe_path.is_file():
        try:
            pipe = json.loads(pipe_path.read_text(encoding="utf-8"))
            rel = str(pipe.get("data_split_config") or rel)
        except json.JSONDecodeError:
            pass
    path = Path(rel)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return {"videos": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _players_swapped_for_image(name: str, override: bool | None, split_cfg: dict) -> bool:
    """CLI override либо players_swapped из data_split.json (IMG_1043 -> True)."""
    if override is not None:
        return bool(override)
    video_id = _video_id_from_image_name(name)
    for video in split_cfg.get("videos", []):
        if video.get("video_id") == video_id:
            return bool(video.get("players_swapped", False))
    return video_id == "IMG_1043"


def _select_players_spatial(
    raw_players: list[tuple[np.ndarray, np.ndarray]],
    split_x: float,
    max_per_side: int = 2,
    assign_cls=None,
    p1_cls: int = SEG_PLAYER1,
    p2_cls: int = SEG_PLAYER2,
) -> list[tuple[int, np.ndarray]]:
    if assign_cls is None:
        assign_cls = _assign_player_seg_cls
    left: list[tuple[np.ndarray, np.ndarray]] = []
    right: list[tuple[np.ndarray, np.ndarray]] = []
    for xyxy, poly in raw_players:
        if assign_cls(xyxy, split_x) == p1_cls:
            left.append((xyxy, poly))
        else:
            right.append((xyxy, poly))
    left = _dedupe_players_same_side(left)[:max_per_side]
    right = _dedupe_players_same_side(right)[:max_per_side]
    out: list[tuple[int, np.ndarray]] = []
    for xyxy, poly in left:
        out.append((p1_cls, poly))
    for xyxy, poly in right:
        out.append((p2_cls, poly))
    return out


def _line_to_polygon(line: str, img_w: int, img_h: int) -> tuple[int, np.ndarray] | None:
    """YOLO-строка (полигон или bbox) → класс и точки в пикселях float32."""
    parts = line.split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    vals = np.asarray(parts[1:], dtype=np.float32)
    scale_w = np.float32(img_w)
    scale_h = np.float32(img_h)
    if vals.size == 4:
        cx, cy, bw, bh = vals
        half = np.float32(0.5)
        x1 = (cx - bw * half) * scale_w
        y1 = (cy - bh * half) * scale_h
        x2 = (cx + bw * half) * scale_w
        y2 = (cy + bh * half) * scale_h
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        return cls_id, pts
    if vals.size < 6 or vals.size % 2 != 0:
        return None
    xs = vals[0::2] * scale_w
    ys = vals[1::2] * scale_h
    pts = np.stack((xs, ys), axis=1).astype(np.float32, copy=False)
    if pts.shape[0] < 3:
        return None
    return cls_id, pts


def _contour_int(pts: np.ndarray) -> np.ndarray:
    return np.rint(pts).astype(np.int32).reshape(-1, 1, 2)


def _put_label(image: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.5) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, org, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, font, scale, color, 1, cv2.LINE_AA)


def _save_prelabel_preview(
    img_path: Path,
    lines: list[str],
    vis_dir: Path | None,
    class_names: list[str],
    img_w: int,
    img_h: int,
) -> None:
    """Полупрозрачная заливка и контур полигонов/боксов, записанных в lines."""
    if vis_dir is None:
        return
    image = cv2.imread(str(img_path))
    if image is None:
        return

    parsed: list[tuple[str, tuple[int, int, int], np.ndarray]] = []
    for line in lines:
        parsed_line = _line_to_polygon(line, img_w, img_h)
        if parsed_line is None:
            continue
        cls_id, pts = parsed_line
        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        color = VIS_CLASS_BGR.get(name, (255, 255, 255))
        parsed.append((name, color, pts))

    overlay = image.copy()
    for _name, color, pts in parsed:
        cv2.fillPoly(overlay, [_contour_int(pts)], color)
    canvas = cv2.addWeighted(overlay, VIS_FILL_ALPHA, image, 1.0 - VIS_FILL_ALPHA, 0.0)

    seen: list[str] = []
    for name, color, pts in parsed:
        cv2.polylines(canvas, [_contour_int(pts)], True, color, 2, cv2.LINE_AA)
        top_i = int(np.argmin(pts[:, 1]))
        tx = int(pts[top_i, 0])
        ty = int(pts[top_i, 1])
        text_x = max(0, min(tx, canvas.shape[1] - 1))
        text_y = ty - 6 if ty > 18 else ty + 16
        _put_label(canvas, name, (text_x, max(text_y, 14)), color)
        if name == "Ball":
            n = np.float32(pts.shape[0])
            cx = pts[:, 0].sum(dtype=np.float32) / n
            cy = pts[:, 1].sum(dtype=np.float32) / n
            cv2.circle(
                canvas,
                (int(np.rint(cx)), int(np.rint(cy))),
                10,
                color,
                2,
                cv2.LINE_AA,
            )
        if name not in seen:
            seen.append(name)

    summary = f"{img_path.name} | classes: {', '.join(seen) if seen else 'none'}"
    _put_label(canvas, summary, (8, 22), (255, 255, 255), scale=0.55)
    cv2.imwrite(str(vis_dir / img_path.name), canvas)


def extract_instance_poly(result, idx: int, img_w: int, img_h: int) -> np.ndarray:
    if result.masks is not None and len(result.masks) > idx:
        xy = result.masks.xy[idx]
        if xy is not None and len(xy) >= 3:
            return np.array(xy, dtype=np.float32)
    xyxy = result.boxes.xyxy[idx].cpu().numpy().astype(np.float32)
    return bbox_to_poly4_xyxy(xyxy)


def prelabel_images(
    images_dir: Path,
    labels_dir: Path,
    model_path: Path,
    conf: float = 0.03,
    class_layout: str = "roboflow_v2",
    players_swapped: bool | None = None,
    split_config: dict | None = None,
    vis_dir: Path | None = None,
) -> dict:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if vis_dir is not None:
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
    use_rfv2 = class_layout == "roboflow_v2"

    model = YOLO(str(model_path))
    native_dual_table = len(getattr(model, "names", {}) or {}) >= 6
    split_cfg = split_config if split_config is not None else _load_split_config()
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"Нет изображений в {images_dir}")

    class_names = RFV2_CLASS_NAMES if use_rfv2 else SEG_CLASS_NAMES + ["Ball"]
    stats = {
        "images": 0,
        "seg_polygons": 0,
        "ball_boxes": 0,
        "ball_polygons": 0,
        "table_splits": 0,
        "native_table_polygons": 0,
        "by_class": {name: 0 for name in class_names},
        "files": [],
        "visualized_dir": str(vis_dir) if vis_dir else None,
    }

    for img_path in image_paths:
        results = model(str(img_path), conf=conf, verbose=False)[0]
        h, w = results.orig_shape
        lines: list[str] = []
        net_boxes: list[np.ndarray] = []
        swapped = _players_swapped_for_image(img_path.name, players_swapped, split_cfg)

        if results.boxes is None or len(results.boxes) == 0:
            label_path = labels_dir / f"{img_path.stem}.txt"
            label_path.write_text("", encoding="utf-8")
            stats["images"] += 1
            stats["files"].append(
                {
                    "image": str(img_path),
                    "label": str(label_path),
                    "lines": 0,
                    "players_swapped": swapped,
                }
            )
            _save_prelabel_preview(img_path, lines, vis_dir, class_names, w, h)
            continue

        table_polys: list[np.ndarray] = []
        direct_left: list[np.ndarray] = []
        direct_right: list[np.ndarray] = []
        raw_players: list[tuple[np.ndarray, np.ndarray]] = []
        deferred_seg: list[tuple[int, np.ndarray]] = []

        for idx in range(len(results.boxes)):
            cls_id = int(results.boxes.cls[idx])
            xyxy = results.boxes.xyxy[idx].cpu().numpy().astype(np.float32)

            if cls_id == TRAIN12_BALL:
                if use_rfv2:
                    poly = extract_instance_poly(results, idx, w, h)
                    lines.append(poly_to_yolo_seg(RFV2_BALL, poly, w, h))
                    stats["ball_polygons"] += 1
                    stats["by_class"]["Ball"] += 1
                else:
                    lines.append(xyxy_to_yolo_bbox(SEG_BALL, xyxy, w, h))
                    stats["ball_boxes"] += 1
                    stats["by_class"]["Ball"] += 1
                continue

            if cls_id == TRAIN12_NET:
                net_boxes.append(xyxy)
                poly = extract_instance_poly(results, idx, w, h)
                net_cls = RFV2_NET if use_rfv2 else SEG_NET
                deferred_seg.append((net_cls, poly))
                continue

            # train14: TableLeft/TableRight уже отдельные классы — полигон берём как есть.
            if native_dual_table and cls_id == RFV2_TABLE_LEFT:
                direct_left.append(extract_instance_poly(results, idx, w, h))
                continue
            if native_dual_table and cls_id == RFV2_TABLE_RIGHT:
                direct_right.append(extract_instance_poly(results, idx, w, h))
                continue

            # train12: один класс Table, режется по split_x.
            if (not native_dual_table) and cls_id == TRAIN12_TABLE:
                table_polys.append(extract_instance_poly(results, idx, w, h))
                continue

            if cls_id in (TRAIN12_PLAYER1, TRAIN12_PLAYER2):
                poly = extract_instance_poly(results, idx, w, h)
                raw_players.append((xyxy, poly))
                continue

            seg_cls = None if native_dual_table else TRAIN12_TO_SEG.get(cls_id)
            if seg_cls is None:
                continue
            poly = extract_instance_poly(results, idx, w, h)
            deferred_seg.append((seg_cls, poly))

        split_x = resolve_split_x(net_boxes, w)
        tl_cls = RFV2_TABLE_LEFT if use_rfv2 else SEG_TABLE_LEFT
        tr_cls = RFV2_TABLE_RIGHT if use_rfv2 else SEG_TABLE_RIGHT

        def _emit_poly(cls_out: int, poly: np.ndarray, class_name: str) -> None:
            lines.append(poly_to_yolo_seg(cls_out, poly, w, h))
            stats["seg_polygons"] += 1
            stats["by_class"][class_name] += 1

        left_candidates: list[np.ndarray] = list(direct_left)
        right_candidates: list[np.ndarray] = list(direct_right)

        for table_poly in table_polys:
            left, right = split_table_polygon(table_poly, split_x, w, h)
            if left is not None:
                left_candidates.append(left)
            if right is not None:
                right_candidates.append(right)
            if left is not None or right is not None:
                stats["table_splits"] += 1

        best_left = _largest_table_polygon(left_candidates, w, h)
        best_right = _largest_table_polygon(right_candidates, w, h)
        if best_left is not None:
            _emit_poly(tl_cls, best_left, "TableLeft")
            if any(best_left is poly for poly in direct_left):
                stats["native_table_polygons"] += 1
        if best_right is not None:
            _emit_poly(tr_cls, best_right, "TableRight")
            if any(best_right is poly for poly in direct_right):
                stats["native_table_polygons"] += 1

        for seg_cls, poly in deferred_seg:
            lines.append(poly_to_yolo_seg(seg_cls, poly, w, h))
            stats["seg_polygons"] += 1
            cls_name = RFV2_CLASS_NAMES[seg_cls] if use_rfv2 else SEG_CLASS_NAMES[seg_cls]
            stats["by_class"][cls_name] += 1

        if use_rfv2:
            assign_cls = functools.partial(_assign_player_rfv2_cls, players_swapped=swapped)
            player_names = RFV2_CLASS_NAMES
            p1_cls, p2_cls = RFV2_PLAYER1, RFV2_PLAYER2
        else:
            assign_cls = functools.partial(_assign_player_seg_cls, players_swapped=swapped)
            player_names = SEG_CLASS_NAMES
            p1_cls, p2_cls = SEG_PLAYER1, SEG_PLAYER2
        for seg_cls, poly in _select_players_spatial(
            raw_players,
            split_x,
            assign_cls=assign_cls,
            p1_cls=p1_cls,
            p2_cls=p2_cls,
        ):
            lines.append(poly_to_yolo_seg(seg_cls, poly, w, h))
            stats["seg_polygons"] += 1
            stats["by_class"][player_names[seg_cls]] += 1

        label_path = labels_dir / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
        stats["files"].append(
            {
                "image": str(img_path),
                "label": str(label_path),
                "lines": len(lines),
                "players_swapped": swapped,
            }
        )
        _save_prelabel_preview(img_path, lines, vis_dir, class_names, w, h)

    fmt = "roboflow_v2" if use_rfv2 else "hybrid_seg"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "model_num_classes": len(getattr(model, "names", {}) or {}),
        "native_dual_table": native_dual_table,
        "players_swapped_override": players_swapped,
        "format": fmt,
        "class_layout": class_layout,
        "format_notes": {
            "hybrid_seg": {
                "seg_classes": "0-4 TableLeft, TableRight, Net, Player1, Player2 (polygon lines)",
                "ball_class": "5 cx cy w h (detection bbox, 5 values total)",
                "roboflow_upload": "build_upload_zip --format seg excludes Ball (class 5)",
            },
            "roboflow_v2": {
                "classes": "0 Ball, 1 Net, 2 Player1, 3 Player2, 4 TableLeft, 5 TableRight (all polygons)",
                "roboflow_upload": "build_upload_zip --format seg_v2 includes Ball polygon",
            },
        }.get(fmt, {}),
        "classes": RFV2_CLASS_NAMES if use_rfv2 else SEG_CLASS_NAMES,
        "class_ball": "Ball",
        "confidence": conf,
        "stats": stats,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-label images with train12 (5 cls, split table) or train14 (6 cls, native halves)"
    )
    parser.add_argument("--run-dir", type=str, required=True, help="runs/<batch>")
    parser.add_argument("--images", type=str, default=None, help="override images dir")
    parser.add_argument("--labels", type=str, default=None, help="override labels dir")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument(
        "--class-layout",
        choices=["hybrid_seg", "roboflow_v2"],
        default="roboflow_v2",
        help="roboflow_v2: 6 polygon classes; hybrid_seg: local 0-4 seg + Ball bbox cls5",
    )
    parser.add_argument(
        "--players-swapped",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Левая сторона кадра = Player2. По умолчанию авто по имени файла и config/data_split.json",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default=None,
        help="папка превью разметки (по умолчанию <run-dir>/visualized)",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="не сохранять превью в visualized/",
    )
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_pipeline_config(root)
    run_dir = resolve_run_dir(args.run_dir, root)

    images_dir = Path(args.images) if args.images else run_dir / "images"
    labels_dir = Path(args.labels) if args.labels else run_dir / "labels"
    model_path = Path(args.model) if args.model else root / cfg["local_model"]
    if args.no_vis:
        vis_dir = None
    elif args.vis_dir:
        vis_dir = Path(args.vis_dir)
    else:
        vis_dir = run_dir / "visualized"

    report = prelabel_images(
        images_dir,
        labels_dir,
        model_path,
        args.conf,
        args.class_layout,
        players_swapped=args.players_swapped,
        vis_dir=vis_dir,
    )
    report_path = run_dir / "prelabel_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    ball_info = (
        f"{report['stats']['ball_polygons']} ball polygons"
        if args.class_layout == "roboflow_v2"
        else f"{report['stats']['ball_boxes']} ball boxes"
    )
    print(
        f"Pre-labeled {report['stats']['images']} images, "
        f"{report['stats']['seg_polygons']} seg polygons, "
        f"{ball_info}"
    )
    print(f"Report: {report_path}")
    if report["stats"].get("visualized_dir"):
        print(f"Visualized: {report['stats']['visualized_dir']}")


if __name__ == "__main__":
    main()
