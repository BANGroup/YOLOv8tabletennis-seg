#!/usr/bin/env python3
"""
Bounce recall analysis — independent of production thresholds for candidate finding.
Reads production formulas from scripts/serve_detector.py (SmartImpactDetector).
Outputs: ball_track.json, report.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from serve_detector import SmartImpactDetector  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_ROOT / "sources" / "IMG_0291.MOV"
MODEL_PATH = PROJECT_ROOT / "runs/segment/train14_ttseg_v3/weights/best.pt"
LIVE_IMPACTS_PATH = (
    PROJECT_ROOT / "results/serve_detector_train14_img0291_scorefix_c/live_impacts.json"
)

START_SEC = 45.0
END_SEC = 180.0
CONF = 0.1
BALL_CLASS_ID = 0
NET_CLASS_ID = 1
TABLE_CLASS_IDS = [4, 5]
NET_IMPACT_MARGIN = np.float32(25.0)

# Production thresholds (read-only reference)
Y_RANGE_MIN = 8.0
CURVATURE_MIN = 5.0
MIN_CONSECUTIVE_DETECTIONS = 3
MIN_FRAME_GAP = 12
MATCH_TIME_TOL = 0.2

TARGET_WINDOWS = {
    "ground_truth_no_right_bounce": (64.8, 68.0),
    "p1_out_74": (74.0, 75.0),
    "p1_out_129": (129.0, 130.0),
    "p1_out_138": (138.0, 139.0),
    "p1_out_146": (146.0, 147.0),
    "p1_out_164": (164.0, 165.0),
}

PRE_OUT_LOOKBACK_SEC = 3.0


def _f32(v: float) -> float:
    return float(np.float32(v))


def extract_table_and_net(result, frame: np.ndarray) -> dict[str, Any]:
    """Mirror BallTracker._maybe_update_table_polygon + _maybe_update_net (once)."""
    meta: dict[str, Any] = {
        "table_polygon": None,
        "table_left": None,
        "table_right": None,
        "table_bbox": None,
        "net_polygon": None,
        "net_bbox": None,
        "net_center_x": None,
        "table_center_x": None,
    }
    if result.masks is None or result.boxes is None:
        return meta

    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    polygons = result.masks.xy

    polys_by_cls: dict[int, list] = {}
    for idx, cls in enumerate(class_ids):
        if cls in TABLE_CLASS_IDS and idx < len(polygons):
            poly = np.array(polygons[idx], dtype=np.float32)
            if poly.shape[0] >= 4:
                center_x = np.mean(poly[:, 0])
                polys_by_cls.setdefault(cls, []).append((center_x, poly))

    left_poly = right_poly = None
    if len(TABLE_CLASS_IDS) >= 2:
        left_cls, right_cls = TABLE_CLASS_IDS[0], TABLE_CLASS_IDS[1]
        if left_cls in polys_by_cls:
            left_poly = polys_by_cls[left_cls][0][1]
        if right_cls in polys_by_cls:
            right_poly = polys_by_cls[right_cls][0][1]

    if left_poly is None and right_poly is None:
        pass
    elif left_poly is not None and right_poly is not None:
        all_points = np.vstack([left_poly, right_poly])
        hull = cv2.convexHull(all_points)
        polygon = hull.reshape(-1, 2)
        meta["table_left"] = left_poly.tolist()
        meta["table_right"] = right_poly.tolist()
        meta["table_polygon"] = polygon.tolist()
    elif left_poly is not None:
        meta["table_left"] = left_poly.tolist()
        meta["table_polygon"] = left_poly.tolist()
    else:
        meta["table_right"] = right_poly.tolist()
        meta["table_polygon"] = right_poly.tolist()

    if meta["table_polygon"] is not None:
        poly_arr = np.array(meta["table_polygon"], dtype=np.float32)
        meta["table_bbox"] = [
            float(np.min(poly_arr[:, 0])),
            float(np.min(poly_arr[:, 1])),
            float(np.max(poly_arr[:, 0])),
            float(np.max(poly_arr[:, 1])),
        ]
        meta["table_center_x"] = float((meta["table_bbox"][0] + meta["table_bbox"][2]) / 2.0)

    net_polys = []
    for idx, cls in enumerate(class_ids):
        if cls == NET_CLASS_ID and idx < len(polygons):
            poly = np.array(polygons[idx], dtype=np.float32)
            if poly.shape[0] >= 4:
                net_polys.append(poly)

    if net_polys:
        if len(net_polys) > 1:
            all_pts = np.vstack(net_polys)
            hull = cv2.convexHull(all_pts)
            net_polygon = hull.reshape(-1, 2)
        else:
            net_polygon = net_polys[0]
        meta["net_polygon"] = net_polygon.tolist()
        x_min = float(np.min(net_polygon[:, 0]))
        x_max = float(np.max(net_polygon[:, 0]))
        y_min = float(np.min(net_polygon[:, 1]))
        y_max = float(np.max(net_polygon[:, 1]))
        meta["net_bbox"] = [x_min, y_min, x_max, y_max]
        meta["net_center_x"] = float(np.median(net_polygon[:, 0]))

    return meta


def detect_best_ball(result) -> tuple[float, float, float] | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    best_conf = -1.0
    best = None
    for box, cls_id, conf in zip(boxes, class_ids, confs):
        if cls_id != BALL_CLASS_ID:
            continue
        c = float(conf)
        if c > best_conf:
            x1, y1, x2, y2 = box
            best = (_f32((x1 + x2) / 2.0), _f32((y1 + y2) / 2.0), _f32(c))
            best_conf = c
    return best


def build_ball_track() -> tuple[list[dict], dict]:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(START_SEC * fps)
    end_frame = int(END_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    model = YOLO(str(MODEL_PATH))
    detector = SmartImpactDetector(frame_height=720, fps=fps)

    scene_meta: dict[str, Any] = {}
    track: list[dict] = []
    total_frames = end_frame - start_frame

    print(f"[*] Inference {START_SEC}-{END_SEC}s ({total_frames} frames) conf={CONF}")
    t0 = time.time()

    for seg_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        abs_frame = start_frame + seg_idx
        t_sec = _f32(START_SEC + seg_idx / fps)

        results = model.predict(frame, conf=CONF, verbose=False)
        result = results[0]

        if not scene_meta.get("table_polygon") or not scene_meta.get("net_center_x"):
            extracted = extract_table_and_net(result, frame)
            for k, v in extracted.items():
                if v is not None and scene_meta.get(k) is None:
                    scene_meta[k] = v

        ball = detect_best_ball(result)
        if ball:
            x, y, conf = ball
            track.append({
                "frame": seg_idx,
                "abs_frame": abs_frame,
                "time": t_sec,
                "x": x,
                "y": y,
                "conf": conf,
            })
        else:
            track.append({
                "frame": seg_idx,
                "abs_frame": abs_frame,
                "time": t_sec,
                "x": None,
                "y": None,
                "conf": None,
            })

        if (seg_idx + 1) % 500 == 0 or seg_idx + 1 == total_frames:
            elapsed = time.time() - t0
            print(f"    {seg_idx + 1}/{total_frames} ({100*(seg_idx+1)/total_frames:.1f}%) "
                  f"{elapsed:.1f}s")

    cap.release()
    scene_meta["fps"] = float(fps)
    scene_meta["start_sec"] = START_SEC
    scene_meta["end_sec"] = END_SEC
    return track, scene_meta


def side_of(x: float, center_x: float) -> str:
    return "left" if x < center_x else "right"


def point_in_table(x: float, y: float, table_poly_cv: np.ndarray) -> bool:
    return cv2.pointPolygonTest(table_poly_cv, (float(x), float(y)), False) >= 0


def point_in_net_zone(x: float, y: float, net_poly_cv: np.ndarray | None,
                      net_bbox: list | None) -> bool:
    if net_poly_cv is not None:
        dist = cv2.pointPolygonTest(net_poly_cv, (float(x), float(y)), True)
        return dist >= -NET_IMPACT_MARGIN
    if net_bbox is None:
        return False
    x1, y1, x2, y2 = net_bbox
    margin = float(NET_IMPACT_MARGIN)
    return (x1 - margin) <= x <= (x2 + margin) and y1 <= y <= y2


def compute_impact_metrics(
    detection_history: list[tuple[int, float, float]],
    detector: SmartImpactDetector,
) -> dict | None:
    """Same maths as SmartImpactDetector.detect_impact at bounce frame."""
    if len(detection_history) < 5:
        return None

    points_list = [(p[1], p[2]) for p in detection_history[-10:]]
    first_y = points_list[0][1]
    middle_y = points_list[len(points_list) // 2][1]
    last_y = points_list[-1][1]

    goes_down = first_y < middle_y
    goes_up = middle_y > last_y
    if not (goes_down and goes_up):
        return None

    y_vals = [p[1] for p in points_list]
    y_range = max(y_vals) - min(y_vals)
    if y_range < Y_RANGE_MIN:
        return None

    curvature = detector.calculate_curvature(points_list)
    if curvature < CURVATURE_MIN:
        return None

    mid_idx = len(points_list) // 2
    bounce_point = points_list[mid_idx]
    bounce_frame = detection_history[-10 + mid_idx][0] if len(detection_history) >= 10 else detection_history[mid_idx][0]

    prev_point = points_list[max(0, mid_idx - 1)]
    next_point = points_list[min(len(points_list) - 1, mid_idx + 1)]
    incoming_vec = (bounce_point[0] - prev_point[0], bounce_point[1] - prev_point[1])
    outgoing_vec = (next_point[0] - bounce_point[0], next_point[1] - bounce_point[1])
    incoming_speed = math.sqrt(incoming_vec[0] ** 2 + incoming_vec[1] ** 2)
    outgoing_speed = math.sqrt(outgoing_vec[0] ** 2 + outgoing_vec[1] ** 2)

    return {
        "bounce_frame": bounce_frame,
        "x": _f32(bounce_point[0]),
        "y": _f32(bounce_point[1]),
        "y_range": _f32(y_range),
        "curvature": _f32(curvature),
        "incoming_speed": _f32(incoming_speed),
        "outgoing_speed": _f32(outgoing_speed),
        "goes_down": goes_down,
        "goes_up": goes_up,
    }


def find_bounce_candidates(track: list[dict], center_x: float) -> list[dict]:
    """Local max in y: >=2 frames descending then >=2 ascending."""
    candidates = []
    n = len(track)

    for i in range(2, n - 2):
        if track[i]["y"] is None:
            continue

        # descending before (y increasing = ball going down on screen)
        if not all(
            track[i - k]["y"] is not None and track[i - k + 1]["y"] is not None
            and track[i - k]["y"] < track[i - k + 1]["y"]
            for k in range(2, 0, -1)
        ):
            continue

        # ascending after (y decreasing)
        if not all(
            track[i + k]["y"] is not None and track[i + k + 1]["y"] is not None
            and track[i + k]["y"] > track[i + k + 1]["y"]
            for k in range(0, 2)
        ):
            continue

        # local maximum in y among neighbors
        yi = track[i]["y"]
        if not (yi >= track[i - 1]["y"] and yi >= track[i - 2]["y"]
                and yi >= track[i + 1]["y"] and yi >= track[i + 2]["y"]):
            continue

        # build detection history up to frame i (only detected frames)
        det_hist = []
        for j in range(max(0, i - 15), i + 1):
            if track[j]["y"] is not None:
                det_hist.append((track[j]["frame"], track[j]["x"], track[j]["y"]))

        detector = SmartImpactDetector(frame_height=720, fps=30.0)
        metrics = compute_impact_metrics(det_hist, detector)

        # Metrics from last up-to-10 detection points (same window as production deque)
        det_tail = []
        for j in range(i, -1, -1):
            if track[j]["y"] is not None:
                det_tail.insert(0, (track[j]["x"], track[j]["y"]))
                if len(det_tail) >= 10:
                    break
        y_vals = [p[1] for p in det_tail]
        y_range_raw = max(y_vals) - min(y_vals) if y_vals else 0.0
        curv_raw = detector.calculate_curvature(det_tail) if len(det_tail) >= 3 else 0.0

        # speeds from immediate neighbors
        x0, y0 = track[i - 1]["x"], track[i - 1]["y"]
        x1, y1 = track[i]["x"], track[i]["y"]
        x2, y2 = track[i + 1]["x"], track[i + 1]["y"]
        in_sp = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        out_sp = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        candidates.append({
            "frame": track[i]["frame"],
            "time": track[i]["time"],
            "x": track[i]["x"],
            "y": track[i]["y"],
            "side": side_of(track[i]["x"], center_x),
            "y_range": _f32(y_range_raw),
            "curvature": _f32(curv_raw),
            "incoming_speed": _f32(in_sp),
            "outgoing_speed": _f32(out_sp),
            "production_pattern_ok": metrics is not None,
        })

    return candidates


def consecutive_detections_at(track: list[dict], idx: int) -> int:
    count = 0
    for j in range(idx, -1, -1):
        if track[j]["y"] is None:
            break
        count += 1
    return count


def first_failing_gate(
    cand: dict,
    track: list[dict],
    table_poly_cv: np.ndarray,
    net_poly_cv: np.ndarray | None,
    net_bbox: list | None,
    registered_impacts: list[dict],
) -> str:
    """Return first production gate that would reject this candidate."""
    if cand["y_range"] < Y_RANGE_MIN:
        return "y_range < 8"
    if cand["curvature"] < CURVATURE_MIN:
        return "curvature < 5"
    if not point_in_table(cand["x"], cand["y"], table_poly_cv):
        return "outside table polygon"
    if point_in_net_zone(cand["x"], cand["y"], net_poly_cv, net_bbox):
        return "inside net zone"

    cand_frame = cand["frame"]
    prev_reg = [imp for imp in registered_impacts if imp["frame"] < cand_frame]
    if prev_reg:
        last_reg_frame = max(imp["frame"] for imp in prev_reg)
        if (cand_frame - last_reg_frame) < MIN_FRAME_GAP:
            return "min_frame_gap (< 12 frames since previous registered impact)"

    idx = cand_frame
    if consecutive_detections_at(track, idx) < MIN_CONSECUTIVE_DETECTIONS:
        return "min_consecutive_detections (< 3 consecutive ball detections)"

    return "no gate failed → unexplained"


def match_candidate(cand: dict, registered: list[dict]) -> dict | None:
    for imp in registered:
        if imp.get("table_side") != cand["side"]:
            continue
        if abs(imp["time"] - cand["time"]) <= MATCH_TIME_TOL:
            return imp
    return None


def detection_recall_stats(track: list[dict], center_x: float) -> dict:
    detected = [t for t in track if t["y"] is not None]
    left = [t for t in detected if t["x"] < center_x]
    right = [t for t in detected if t["x"] >= center_x]

    def longest_gaps(side_filter):
        gaps = []
        gap_start = None
        for i, t in enumerate(track):
            in_half = (t["y"] is not None and (
                (side_filter == "left" and t["x"] < center_x)
                or (side_filter == "right" and t["x"] >= center_x)
            ))
            if not in_half:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gaps.append({
                        "start_frame": gap_start,
                        "end_frame": i - 1,
                        "start_time": track[gap_start]["time"],
                        "end_time": track[i - 1]["time"],
                        "length_frames": i - gap_start,
                        "half": side_filter,
                    })
                    gap_start = None
        if gap_start is not None:
            gaps.append({
                "start_frame": gap_start,
                "end_frame": len(track) - 1,
                "start_time": track[gap_start]["time"],
                "end_time": track[-1]["time"],
                "length_frames": len(track) - gap_start,
                "half": side_filter,
            })
        gaps.sort(key=lambda g: g["length_frames"], reverse=True)
        return gaps

    return {
        "total_frames": len(track),
        "detected_frames": len(detected),
        "detection_rate": len(detected) / len(track) if track else 0,
        "left": {
            "detected_frames": len(left),
            "detection_rate_of_all_frames": len(left) / len(track) if track else 0,
            "mean_confidence": float(np.mean([t["conf"] for t in left])) if left else 0.0,
        },
        "right": {
            "detected_frames": len(right),
            "detection_rate_of_all_frames": len(right) / len(track) if track else 0,
            "mean_confidence": float(np.mean([t["conf"] for t in right])) if right else 0.0,
        },
        "longest_gaps_left": longest_gaps("left")[:5],
        "longest_gaps_right": longest_gaps("right")[:5],
    }


def analyze_target_windows(
    candidates: list[dict],
    track: list[dict],
    registered: list[dict],
    table_poly_cv: np.ndarray,
    net_poly_cv: np.ndarray | None,
    net_bbox: list | None,
) -> dict:
    results = {}
    for name, (t0, t1) in TARGET_WINDOWS.items():
        window_cands = [c for c in candidates if t0 <= c["time"] <= t1]
        window_track = [t for t in track if t0 <= t["time"] <= t1]
        entry = {
            "window": [t0, t1],
            "candidates": window_cands,
            "track_frames": len(window_track),
            "track_detected": sum(1 for t in window_track if t["y"] is not None),
        }
        if name.startswith("p1_out"):
            out_time = (t0 + t1) / 2.0
            before = [
                c for c in candidates
                if c["side"] == "right"
                and (out_time - PRE_OUT_LOOKBACK_SEC) <= c["time"] < out_time
            ]
            entry["right_half_candidates_before_out"] = []
            for c in before:
                matched = match_candidate(c, registered)
                item = {**c, "matched_registered": matched is not None}
                if not matched:
                    item["first_failing_gate"] = first_failing_gate(
                        c, track, table_poly_cv, net_poly_cv, net_bbox, registered
                    )
                entry["right_half_candidates_before_out"].append(item)
        results[name] = entry
    return results


def print_summary(report: dict) -> None:
    dr = report["detection_recall"]
    cm = report["candidate_summary"]
    rg = report["rejection_gates_by_half"]

    print("\n" + "=" * 72)
    print("BOUNCE RECALL ANALYSIS SUMMARY")
    print("=" * 72)
    print(f"Net center_x: {report['scene']['net_center_x']:.1f}")
    print(f"\n--- Ball detection recall ---")
    print(f"Total frames: {dr['total_frames']}")
    print(f"Detected: {dr['detected_frames']} ({100*dr['detection_rate']:.1f}%)")
    print(f"Left half:  {dr['left']['detected_frames']} frames, "
          f"mean conf={dr['left']['mean_confidence']:.3f}")
    print(f"Right half: {dr['right']['detected_frames']} frames, "
          f"mean conf={dr['right']['mean_confidence']:.3f}")

    print(f"\n--- Bounce candidates ---")
    print(f"Total: {cm['total']} | Matched: {cm['matched']} | Unmatched: {cm['unmatched']}")
    print(f"Left:  {cm['left_total']} (matched {cm['left_matched']}, unmatched {cm['left_unmatched']})")
    print(f"Right: {cm['right_total']} (matched {cm['right_matched']}, unmatched {cm['right_unmatched']})")

    print(f"\n--- Rejection gates (unmatched only) ---")
    print(f"{'Gate':<45} {'Left':>6} {'Right':>6}")
    print("-" * 60)
    all_gates = sorted(set(list(rg["left"].keys()) + list(rg["right"].keys())))
    for gate in all_gates:
        print(f"{gate:<45} {rg['left'].get(gate, 0):>6} {rg['right'].get(gate, 0):>6}")

    print(f"\n--- Target windows (P1 out) ---")
    for key in ["p1_out_74", "p1_out_129", "p1_out_138", "p1_out_146", "p1_out_164"]:
        tw = report["target_windows"][key]
        rh = tw.get("right_half_candidates_before_out", [])
        print(f"  {key} ({tw['window'][0]}-{tw['window'][1]}s): "
              f"{len(rh)} right-half candidate(s) before out")
        for c in rh:
            status = "MATCHED" if c["matched_registered"] else f"REJECTED: {c.get('first_failing_gate','?')}"
            print(f"    t={c['time']:.2f}s x={c['x']:.0f} y={c['y']:.0f} → {status}")

    gt = report["target_windows"]["ground_truth_no_right_bounce"]
    rh_gt = [c for c in gt["candidates"] if c["side"] == "right"]
    print(f"\n--- Ground truth 64.8-68.0s (no right bounce) ---")
    print(f"  Right-half candidates in window: {len(rh_gt)} "
          f"({'FAIL: detector too loose' if rh_gt else 'OK'})")

    print(f"\n--- Verdict ---")
    print(report["verdict"]["summary"])
    print(f"Recommendation: {report['verdict']['recommendation']}")
    print("=" * 72)


def main() -> None:
    ball_track_path = OUTPUT_DIR / "ball_track.json"

    if ball_track_path.exists():
        print(f"[*] Loading cached {ball_track_path}")
        with open(ball_track_path, encoding="utf-8") as f:
            cached = json.load(f)
        track = cached["track"]
        scene_meta = cached["scene"]
    else:
        track, scene_meta = build_ball_track()
        with open(ball_track_path, "w", encoding="utf-8") as f:
            json.dump({"track": track, "scene": scene_meta}, f, ensure_ascii=False, indent=2)
        print(f"[+] Saved {ball_track_path}")

    center_x = scene_meta.get("net_center_x") or scene_meta.get("table_center_x") or 600.0
    table_poly = np.array(scene_meta["table_polygon"], dtype=np.float32).reshape(-1, 1, 2)
    net_poly_cv = None
    if scene_meta.get("net_polygon"):
        net_poly_cv = np.array(scene_meta["net_polygon"], dtype=np.float32).reshape(-1, 1, 2)
    net_bbox = scene_meta.get("net_bbox")

    with open(LIVE_IMPACTS_PATH, encoding="utf-8") as f:
        registered = json.load(f)

    recall = detection_recall_stats(track, center_x)
    candidates = find_bounce_candidates(track, center_x)

    # enrich candidates with polygon checks
    for c in candidates:
        c["inside_table_polygon"] = bool(point_in_table(c["x"], c["y"], table_poly))
        c["inside_net_zone"] = bool(point_in_net_zone(c["x"], c["y"], net_poly_cv, net_bbox))

    matched_list = []
    unmatched_list = []
    rejection_gates = {"left": {}, "right": {}}

    for c in candidates:
        m = match_candidate(c, registered)
        if m:
            c["matched_impact"] = {"frame": m["frame"], "time": m["time"]}
            matched_list.append(c)
        else:
            gate = first_failing_gate(c, track, table_poly, net_poly_cv, net_bbox, registered)
            c["first_failing_gate"] = gate
            unmatched_list.append(c)
            half = c["side"]
            rejection_gates[half][gate] = rejection_gates[half].get(gate, 0) + 1

    cm = {
        "total": len(candidates),
        "matched": len(matched_list),
        "unmatched": len(unmatched_list),
        "left_total": sum(1 for c in candidates if c["side"] == "left"),
        "right_total": sum(1 for c in candidates if c["side"] == "right"),
        "left_matched": sum(1 for c in matched_list if c["side"] == "left"),
        "right_matched": sum(1 for c in matched_list if c["side"] == "right"),
        "left_unmatched": sum(1 for c in unmatched_list if c["side"] == "left"),
        "right_unmatched": sum(1 for c in unmatched_list if c["side"] == "right"),
    }

    target = analyze_target_windows(
        candidates, track, registered, table_poly, net_poly_cv, net_bbox
    )

    # Verdict logic
    right_unmatched = [c for c in unmatched_list if c["side"] == "right"]
    left_unmatched = [c for c in unmatched_list if c["side"] == "left"]
    right_rate = recall["right"]["detection_rate_of_all_frames"]
    left_rate = recall["left"]["detection_rate_of_all_frames"]

    limiting_factors = []
    if right_rate < left_rate * 0.85:
        limiting_factors.append("(a) ball detection recall lower on right")
    top_right_gates = sorted(rejection_gates["right"].items(), key=lambda x: -x[1])[:3]
    if top_right_gates:
        limiting_factors.append(f"(b) top rejection gates on right: {top_right_gates}")
    outside_right = sum(1 for c in right_unmatched if c["first_failing_gate"] == "outside table polygon")
    if outside_right > 0:
        limiting_factors.append(
            f"(c) {outside_right} right unmatched rejected by table polygon "
            f"(right poly {len(scene_meta.get('table_right', []) )} pts vs left "
            f"{len(scene_meta.get('table_left', []))} pts)"
        )

    p1_out_evidence = []
    for key in ["p1_out_74", "p1_out_129", "p1_out_138", "p1_out_146", "p1_out_164"]:
        rh = target[key].get("right_half_candidates_before_out", [])
        unmatched_before = [c for c in rh if not c["matched_registered"]]
        p1_out_evidence.append({"window": key, "right_candidates_before": len(rh),
                                "unmatched": len(unmatched_before)})

    hypothesis_supported = any(e["unmatched"] > 0 for e in p1_out_evidence)

    verdict_summary = (
        f"Registered impacts: {len(registered)} (L={cm['left_matched']} matched candidates, "
        f"R={cm['right_matched']}). "
        f"Unmatched candidates: L={cm['left_unmatched']} R={cm['right_unmatched']}. "
        f"Right detection rate {100*right_rate:.1f}% vs left {100*left_rate:.1f}%. "
        f"Hypothesis (missed right bounces → P1 out blame): "
        f"{'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'} by P1-out window analysis."
    )

    rec_parts = []
    if outside_right > 0:
        rec_parts.append("Review/refine TableRight polygon (104 pts vs 261 left)")
    if top_right_gates and top_right_gates[0][0] == "y_range < 8":
        rec_parts.append("Consider lowering y_range threshold for fast low-amplitude right-side bounces")
    elif top_right_gates and top_right_gates[0][0] == "curvature < 5":
        rec_parts.append("Consider lowering curvature threshold for shallow right-side bounces")
    elif top_right_gates and top_right_gates[0][0] == "min_frame_gap (< 12 frames since previous registered impact)":
        rec_parts.append("min_frame_gap may suppress closely-spaced right bounces")
    if right_rate < left_rate * 0.9:
        rec_parts.append("Improve ball detection recall on right half (lower conf or better model)")
    if not rec_parts:
        rec_parts.append("No single dominant factor; inspect unexplained unmatched candidates manually")

    report = {
        "trace_id": "wf:bounce-recall:2026-09-05T19:10:00Z",
        "source_run": str(LIVE_IMPACTS_PATH.relative_to(PROJECT_ROOT)),
        "scene": scene_meta,
        "thresholds": {
            "y_range_min": Y_RANGE_MIN,
            "curvature_min": CURVATURE_MIN,
            "min_consecutive_detections": MIN_CONSECUTIVE_DETECTIONS,
            "min_frame_gap": MIN_FRAME_GAP,
            "match_time_tol_sec": MATCH_TIME_TOL,
        },
        "detection_recall": recall,
        "registered_impacts_count": len(registered),
        "registered_left": sum(1 for i in registered if i.get("table_side") == "left"),
        "registered_right": sum(1 for i in registered if i.get("table_side") == "right"),
        "candidate_summary": cm,
        "rejection_gates_by_half": rejection_gates,
        "unmatched_candidates": unmatched_list,
        "matched_candidates": matched_list,
        "target_windows": target,
        "p1_out_evidence": p1_out_evidence,
        "verdict": {
            "summary": verdict_summary,
            "limiting_factors": limiting_factors,
            "hypothesis_supported": hypothesis_supported,
            "recommendation": "; ".join(rec_parts),
        },
    }

    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved {report_path}")

    print_summary(report)


if __name__ == "__main__":
    main()
