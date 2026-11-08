#!/usr/bin/env python3
"""Pre-label YOLO seg + Ball bbox из train12 seg модели. Дата: 2026-05-26."""

from __future__ import annotations

import argparse
import functools
import json
import re
from datetime import datetime, timezone
from pathlib import Path

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


def split_table_polygon(
    poly_xy: np.ndarray,
    split_x: float,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Split table polygon into TableLeft / TableRight by vertical line at split_x."""
    poly = poly_xy.astype(np.float32)
    left = _clip_polygon_half_plane(poly, keep_left=True, split_x=split_x)
    right = _clip_polygon_half_plane(poly, keep_left=False, split_x=split_x)

    if left is not None and left.shape[0] >= 3:
        left[:, 0] = np.clip(left[:, 0], 0.0, float(img_w))
        left[:, 1] = np.clip(left[:, 1], 0.0, float(img_h))
    else:
        left = None

    if right is not None and right.shape[0] >= 3:
        right[:, 0] = np.clip(right[:, 0], 0.0, float(img_w))
        right[:, 1] = np.clip(right[:, 1], 0.0, float(img_h))
    else:
        right = None

    return left, right


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
) -> dict:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
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

        for poly in direct_left:
            _emit_poly(tl_cls, poly, "TableLeft")
            stats["native_table_polygons"] += 1
        for poly in direct_right:
            _emit_poly(tr_cls, poly, "TableRight")
            stats["native_table_polygons"] += 1

        for table_poly in table_polys:
            left, right = split_table_polygon(table_poly, split_x, w, h)
            if left is not None:
                _emit_poly(tl_cls, left, "TableLeft")
            if right is not None:
                _emit_poly(tr_cls, right, "TableRight")
            if left is not None or right is not None:
                stats["table_splits"] += 1

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
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_pipeline_config(root)
    run_dir = resolve_run_dir(args.run_dir, root)

    images_dir = Path(args.images) if args.images else run_dir / "images"
    labels_dir = Path(args.labels) if args.labels else run_dir / "labels"
    model_path = Path(args.model) if args.model else root / cfg["local_model"]

    report = prelabel_images(
        images_dir,
        labels_dir,
        model_path,
        args.conf,
        args.class_layout,
        players_swapped=args.players_swapped,
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


if __name__ == "__main__":
    main()
