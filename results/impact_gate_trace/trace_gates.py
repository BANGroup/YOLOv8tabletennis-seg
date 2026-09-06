#!/usr/bin/env python3
"""
Impact gate trace — replay SmartImpactDetector on cached ball_track.json.
Analysis only; no production changes.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from serve_detector import SmartImpactDetector  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
BALL_TRACK_PATH = PROJECT_ROOT / "results/bounce_recall_analysis/ball_track.json"
REFERENCE_IMPACTS_PATH = (
    PROJECT_ROOT / "results/serve_detector_train14_img0291_ts/live_impacts.json"
)

MATCH_TIME_TOL = 0.1
FIDELITY_FRAME_TOL = 2
CANDIDATE_WINDOW = 12
NET_IMPACT_MARGIN = np.float32(25.0)

# 15 strongest candidates listed in task spec (for verification)
LISTED_CANDIDATES = [
    (52.03, 896, "R"),
    (60.40, 829, "R"),
    (71.00, 741, "R"),
    (73.20, 753, "R"),
    (113.23, 787, "R"),
    (123.57, 427, "L"),
    (135.70, 759, "R"),
    (143.63, 703, "R"),
    (144.00, 351, "L"),
    (145.53, 836, "R"),
    (152.63, 676, "R"),
    (156.30, 881, "R"),
    (160.07, 819, "R"),
    (167.23, 318, "L"),
    (177.63, 393, "L"),
]


def _f32(v: float) -> float:
    return float(np.float32(v))


def setup_detector(scene: dict[str, Any], fps: float = 30.0) -> SmartImpactDetector:
    """Populate geometry exactly as BallTracker after table + net detection."""
    detector = SmartImpactDetector(frame_height=720, fps=fps)

    polygon = np.array(scene["table_polygon"], dtype=np.float32)
    detector.table_polygon = polygon
    detector.table_polygon_cv = polygon.reshape(-1, 1, 2)

    if scene.get("table_left"):
        left = np.array(scene["table_left"], dtype=np.float32)
        detector.table_left_cv = left.reshape(-1, 1, 2)
    if scene.get("table_right"):
        right = np.array(scene["table_right"], dtype=np.float32)
        detector.table_right_cv = right.reshape(-1, 1, 2)

    table_y_min = float(np.min(polygon[:, 1]))
    table_y_max = float(np.max(polygon[:, 1]))
    detector.table_top_y = table_y_min
    detector.table_bottom_y = table_y_max
    detector.table_zone_start = int(np.mean(polygon[:, 1]))

    net_center_x = scene.get("net_center_x")
    if net_center_x is not None:
        detector.net_zone_x = float(net_center_x)
        detector.table_center_x = float(net_center_x)
    else:
        detector.table_center_x = float(scene.get("table_center_x", 670.0))

    if scene.get("net_polygon"):
        net_poly = np.array(scene["net_polygon"], dtype=np.float32)
        detector.net_polygon = net_poly

    return detector


def evaluate_detect_impact_gates(detector: SmartImpactDetector) -> str:
    """
    First failing gate after detect_impact returned False.
    Order mirrors SmartImpactDetector.detect_impact source.
    """
    hist_len = len(detector.trajectory_history)
    if hist_len < 5:
        return "len(history) < 5"

    points_list = list(detector.trajectory_history)
    first_y = points_list[0][1]
    middle_y = points_list[len(points_list) // 2][1]
    last_y = points_list[-1][1]

    goes_down = first_y < middle_y
    goes_up = middle_y > last_y
    if not (goes_down and goes_up):
        return "pattern (first_y < middle_y and middle_y > last_y)"

    y_min = min(detector.y_history)
    y_max = max(detector.y_history)
    y_range = y_max - y_min
    if y_range < 8:
        return "y_range < 8"

    curvature = detector.calculate_curvature(points_list)
    if curvature < 5:
        return "curvature < 5"

    mid_idx = len(points_list) // 2
    pixel_points = list(detector.pixel_history)
    bounce_pixel = pixel_points[mid_idx] if len(pixel_points) > mid_idx else points_list[mid_idx]

    if detector.table_polygon_cv is None:
        return "table_polygon_cv is None"
    if cv2.pointPolygonTest(
        detector.table_polygon_cv,
        (float(bounce_pixel[0]), float(bounce_pixel[1])),
        False,
    ) < 0:
        return "outside table polygon"

    return "all detect_impact gates passed"


def evaluate_process_gates(
    detector: SmartImpactDetector,
    impact_meta: dict,
) -> str | None:
    """Process-level rejection after detect_impact returned meta (before process_detection call)."""
    if (detector.consecutive_detections + 1) < detector.min_consecutive_detections:
        return "consecutive_detections < 3"

    impact_frame = impact_meta.get("bounce_frame")
    if impact_frame is None:
        impact_frame = impact_meta.get("frame")
    if (
        detector.last_impact_frame is not None
        and impact_frame is not None
        and (impact_frame - detector.last_impact_frame) < detector.min_frame_gap
    ):
        return "min_frame_gap < 12"

    return None


def side_of(x: float, center_x: float) -> str:
    return "left" if x < center_x else "right"


def point_in_table(x: float, y: float, table_poly_cv: np.ndarray) -> bool:
    return cv2.pointPolygonTest(table_poly_cv, (float(x), float(y)), False) >= 0


def point_in_net_zone(
    x: float,
    y: float,
    net_poly: np.ndarray | None,
    net_bbox: list | None,
) -> bool:
    if net_poly is not None:
        net_cv = net_poly.reshape(-1, 1, 2).astype(np.float32)
        dist = cv2.pointPolygonTest(net_cv, (float(x), float(y)), True)
        return dist >= -NET_IMPACT_MARGIN
    if net_bbox is None:
        return False
    x1, y1, x2, y2 = net_bbox
    margin = float(NET_IMPACT_MARGIN)
    return (x1 - margin) <= x <= (x2 + margin) and y1 <= y <= y2


def find_bounce_candidates(
    track: list[dict],
    center_x: float,
    table_poly_cv: np.ndarray,
    net_poly: np.ndarray | None,
    net_bbox: list | None,
    registered: list[dict],
) -> list[dict]:
    """Local max in y with table/net filters; unmatched vs registered ±0.1s same side."""
    raw: list[dict] = []
    n = len(track)

    for i in range(2, n - 2):
        if track[i]["y"] is None:
            continue

        if not all(
            track[i - k]["y"] is not None
            and track[i - k + 1]["y"] is not None
            and track[i - k]["y"] < track[i - k + 1]["y"]
            for k in range(2, 0, -1)
        ):
            continue

        if not all(
            track[i + k]["y"] is not None
            and track[i + k + 1]["y"] is not None
            and track[i + k]["y"] > track[i + k + 1]["y"]
            for k in range(0, 2)
        ):
            continue

        yi = track[i]["y"]
        if not (
            yi >= track[i - 1]["y"]
            and yi >= track[i - 2]["y"]
            and yi >= track[i + 1]["y"]
            and yi >= track[i + 2]["y"]
        ):
            continue

        x_i, y_i = track[i]["x"], track[i]["y"]
        if not point_in_table(x_i, y_i, table_poly_cv):
            continue
        if point_in_net_zone(x_i, y_i, net_poly, net_bbox):
            continue

        cand_side = side_of(x_i, center_x)
        matched = False
        for imp in registered:
            if imp.get("table_side") != cand_side:
                continue
            if abs(imp["time"] - track[i]["time"]) <= MATCH_TIME_TOL:
                matched = True
                break

        if matched:
            continue

        raw.append(
            {
                "frame": track[i]["frame"],
                "time": _f32(track[i]["time"]),
                "x": _f32(x_i),
                "y": _f32(y_i),
                "side": cand_side,
            }
        )

    return raw


def match_impacts(
    replayed: list[dict],
    reference: list[dict],
    frame_tol: int = FIDELITY_FRAME_TOL,
) -> dict[str, Any]:
    """Match replayed vs reference by frame within tolerance."""
    ref_used: set[int] = set()
    matched_pairs: list[dict] = []
    missed: list[dict] = []
    extra: list[dict] = []

    for rep in replayed:
        rep_frame = rep["frame"]
        best_idx = None
        best_delta = frame_tol + 1
        for idx, ref in enumerate(reference):
            if idx in ref_used:
                continue
            delta = abs(ref["frame"] - rep_frame)
            if delta <= frame_tol and delta < best_delta:
                best_delta = delta
                best_idx = idx
        if best_idx is not None:
            ref_used.add(best_idx)
            matched_pairs.append(
                {
                    "replay_frame": rep_frame,
                    "ref_frame": reference[best_idx]["frame"],
                    "delta_frames": best_delta,
                }
            )
        else:
            extra.append(rep)

    for idx, ref in enumerate(reference):
        if idx not in ref_used:
            missed.append(ref)

    return {
        "replayed_count": len(replayed),
        "reference_count": len(reference),
        "matched": len(matched_pairs),
        "missed": len(missed),
        "extra": len(extra),
        "matched_pairs": matched_pairs,
        "missed_impacts": missed,
        "extra_impacts": extra,
        "fidelity_ok": len(matched_pairs) >= 55,
    }


def replay_track(
    track: list[dict],
    scene: dict[str, Any],
) -> tuple[list[dict], list[dict], SmartImpactDetector]:
    """
    Full frame-order replay mirroring BallTracker.track_ball impact path.
    Returns (registered_impacts, per_detection_log, final_detector).
    """
    fps = float(scene.get("fps", 30.0))
    detector = setup_detector(scene, fps=fps)
    detection_log: list[dict] = []
    missed_frames = 0
    prev_live_count = 0

    for row in track:
        frame_idx = row["frame"]
        fps_time = row["time"]

        if row["y"] is None:
            missed_frames += 1
            if missed_frames > 5:
                detector.finalize_rally(frame_idx, fps_time)
            detection_log.append(
                {
                    "frame": frame_idx,
                    "time": fps_time,
                    "has_ball": False,
                    "detect_impact_passed": False,
                    "gate": "no_ball_detection",
                    "process_gate": None,
                    "registered": False,
                }
            )
            continue

        missed_frames = 0
        position = (_f32(row["x"]), _f32(row["y"]))
        detector.add_ball_position(position, frame_idx)

        impact_meta = detector.detect_impact(
            position,
            frame_idx,
            pixel_position=position,
            fps_time=fps_time,
        )

        entry: dict[str, Any] = {
            "frame": frame_idx,
            "time": fps_time,
            "has_ball": True,
            "x": position[0],
            "y": position[1],
            "detect_impact_passed": bool(impact_meta),
            "gate": None,
            "process_gate": None,
            "registered": False,
        }

        if impact_meta:
            proc_reject = evaluate_process_gates(detector, impact_meta)
            added = detector.process_detection(position, frame_idx, fps_time, impact_meta)
            if len(detector.live_impacts) > prev_live_count:
                entry["registered"] = True
                prev_live_count = len(detector.live_impacts)
            elif proc_reject:
                entry["process_gate"] = proc_reject
            elif not added:
                entry["process_gate"] = evaluate_process_gates(detector, impact_meta) or "process_detection rejected"
            entry["gate"] = "all detect_impact gates passed"
        else:
            entry["gate"] = evaluate_detect_impact_gates(detector)

        detection_log.append(entry)

    return list(detector.live_impacts), detection_log, detector


def gate_depth(gate: str) -> int:
    order = [
        "no_ball_detection",
        "len(history) < 5",
        "pattern (first_y < middle_y and middle_y > last_y)",
        "y_range < 8",
        "curvature < 5",
        "outside table polygon",
        "table_polygon_cv is None",
        "all detect_impact gates passed",
        "consecutive_detections < 3",
        "min_frame_gap < 12",
    ]
    try:
        return order.index(gate)
    except ValueError:
        return -1


def closest_to_passing(
    gates_in_window: list[str],
    any_detect_pass: bool,
    process_rejects: list[str],
) -> str:
    """Deepest failure; if detect_impact ever passed, process gate dominates."""
    if any_detect_pass:
        if process_rejects:
            return Counter(process_rejects).most_common(1)[0][0]
        return "detect_impact passed but not registered (unknown process reason)"

    detect_gates = [
        g
        for g in gates_in_window
        if g not in ("no_ball_detection", "all detect_impact gates passed")
        and "consecutive" not in g
        and "min_frame_gap" not in g
        and not g.startswith("process_detection")
        and "unknown process" not in g
    ]
    if not detect_gates:
        return "unknown"
    return max(detect_gates, key=gate_depth)


def analyze_candidate_window(
    cand: dict,
    detection_log: list[dict],
    track: list[dict],
) -> dict[str, Any]:
    cand_frame = cand["frame"]
    lo = cand_frame - CANDIDATE_WINDOW
    hi = cand_frame + CANDIDATE_WINDOW

    window_entries = [e for e in detection_log if lo <= e["frame"] <= hi]
    all_frames = [t for t in track if lo <= t["frame"] <= hi]

    gate_sequence: list[dict] = []
    any_detect_pass = False
    process_rejects: list[str] = []

    for fr in all_frames:
        fidx = fr["frame"]
        det = next((e for e in window_entries if e["frame"] == fidx), None)
        if det is None:
            gate_sequence.append({"frame": fidx, "gate": "no_ball_detection"})
        elif det["detect_impact_passed"]:
            any_detect_pass = True
            gate = "all detect_impact gates passed"
            if det.get("process_gate"):
                process_rejects.append(det["process_gate"])
                gate = det["process_gate"]
            elif det.get("registered"):
                gate = "registered"
            gate_sequence.append({"frame": fidx, "gate": gate})
        else:
            gate_sequence.append({"frame": fidx, "gate": det["gate"]})

    compact = [f"{g['frame']}:{g['gate'][:12]}" for g in gate_sequence if g["gate"] != "no_ball_detection"]
    if len(compact) > 20:
        compact_str = "; ".join(compact[:10]) + " ... " + "; ".join(compact[-5:])
    else:
        compact_str = "; ".join(compact)

    gates_only = [g["gate"] for g in gate_sequence]
    replay_registered_in_window = any(g["gate"] == "registered" for g in gate_sequence)

    if replay_registered_in_window:
        closest = "replay registered in window (reference mismatch)"
    else:
        closest = closest_to_passing(gates_only, any_detect_pass, process_rejects)

    return {
        "candidate": cand,
        "window_frames": [lo, hi],
        "gate_sequence_compact": compact_str,
        "gate_sequence": gate_sequence,
        "any_detect_impact_passed": any_detect_pass,
        "process_level_rejects": process_rejects,
        "replay_registered_in_window": replay_registered_in_window,
        "closest_to_passing_gate": closest,
    }


def resolve_listed_candidates(
    track: list[dict],
    center_x: float,
    candidates: list[dict],
) -> list[dict]:
    """Map the 15 orchestrator-listed timestamps to track rows (+ merge with candidate set)."""
    resolved: list[dict] = []
    seen: set[int] = set()

    for t, x, side_abbr in LISTED_CANDIDATES:
        side = "left" if side_abbr == "L" else "right"
        pool = [
            row
            for row in track
            if row["y"] is not None and abs(row["time"] - t) < 0.2
        ]
        if not pool:
            continue
        best = min(pool, key=lambda r: abs(r["time"] - t) + abs(r["x"] - x) * 0.001)
        cand = {
            "frame": best["frame"],
            "time": _f32(best["time"]),
            "x": _f32(best["x"]),
            "y": _f32(best["y"]),
            "side": side_of(best["x"], center_x),
            "listed_orchestrator": True,
            "listed_target": {"time": t, "x": x, "side": side_abbr},
            "in_unmatched_set": any(abs(c["time"] - best["time"]) < 0.05 for c in candidates),
        }
        if cand["frame"] not in seen:
            seen.add(cand["frame"])
            resolved.append(cand)

    for c in candidates:
        if c["frame"] not in seen:
            seen.add(c["frame"])
            resolved.append({**c, "listed_orchestrator": False, "in_unmatched_set": True})

    return resolved


def verify_listed_candidates(candidates: list[dict]) -> dict[str, Any]:
    found = []
    missing = []
    for t, x, side_abbr in LISTED_CANDIDATES:
        side = "left" if side_abbr == "L" else "right"
        match = None
        for c in candidates:
            if c["side"] != side:
                continue
            if abs(c["time"] - t) < 0.15 and abs(c["x"] - x) < 30:
                match = c
                break
        if match:
            found.append({"listed": (t, x, side_abbr), "matched": match})
        else:
            missing.append({"listed": (t, x, side_abbr)})

    return {
        "listed_count": len(LISTED_CANDIDATES),
        "found": len(found),
        "missing": missing,
        "found_pairs": found,
    }


def aggregate_blocking_gates(analyses: list[dict], *, only_unblocked: bool = False) -> dict[str, Any]:
    by_half: dict[str, Counter] = {"left": Counter(), "right": Counter()}
    for a in analyses:
        if only_unblocked and a.get("replay_registered_in_window"):
            continue
        gate = a["closest_to_passing_gate"]
        half = a["candidate"]["side"]
        by_half[half][gate] += 1

    all_gates = Counter()
    for half in by_half:
        all_gates.update(by_half[half])

    dominant = None
    if all_gates:
        top = all_gates.most_common(2)
        if len(top) == 1 or top[0][1] > top[1][1] * 1.3:
            dominant = top[0][0]

    return {
        "all": dict(all_gates),
        "left": dict(by_half["left"]),
        "right": dict(by_half["right"]),
        "dominant_gate": dominant,
        "spread": dominant is None,
    }


def recommend_change(aggregate: dict[str, Any]) -> dict[str, str]:
    dom = aggregate.get("dominant_gate")
    if not dom or aggregate.get("spread"):
        return {
            "change": "none_dominant",
            "detail": "Misses spread across multiple gates; no single threshold change clearly dominates.",
            "risk": "Any isolated threshold tweak may fix some bounces while adding false positives elsewhere.",
        }

    risks = {
        "pattern (first_y < middle_y and middle_y > last_y)": (
            "Relax pattern gate (e.g. allow shallower V)",
            "More rolling/noisy trajectories on table surface may register as bounces.",
        ),
        "y_range < 8": (
            "Lower y_range threshold (e.g. 8 → 6)",
            "Small jitter and slow rolls may register as impacts.",
        ),
        "curvature < 5": (
            "Lower curvature threshold (e.g. 5 → 3)",
            "Gradual arcs and tracking noise may produce false bounces.",
        ),
        "outside table polygon": (
            "Expand/refine table polygon (especially table_right)",
            "May register off-table contacts; polygon errors could mis-assign side.",
        ),
        "len(history) < 5": (
            "Require fewer history points or interpolate gaps",
            "Sparse detections may produce unstable bounce points.",
        ),
        "consecutive_detections < 3": (
            "Lower min_consecutive_detections (3 → 2)",
            "Single-frame noise bursts could confirm false impacts.",
        ),
        "min_frame_gap < 12": (
            "Lower min_frame_gap (12 → 8)",
            "Fast double-bounces and rolling contacts may duplicate or merge incorrectly.",
        ),
    }
    change, risk = risks.get(dom, ("Review gate: " + dom, "Unknown side effects."))
    return {"change": change, "detail": dom, "risk": risk}


def print_summary(report: dict) -> None:
    fid = report["fidelity"]
    print("\n" + "=" * 72)
    print("IMPACT GATE TRACE SUMMARY")
    print("=" * 72)
    print("FIDELITY (replay vs reference 65 impacts, ±2 frames):")
    print(
        f"  Replayed: {fid['replayed_count']} | Matched: {fid['matched']} | "
        f"Missed: {fid['missed']} | Extra: {fid['extra']}"
    )
    print(f"  Fidelity gate (>=55 matched): {'PASS' if fid['fidelity_ok'] else 'FAIL — stop gate conclusions'}")

    if not fid["fidelity_ok"]:
        print("\n  Divergence too high for gate analysis. See missed/extra in trace_report.json.")
        print("=" * 72)
        return

    agg = report["aggregate_blocking_gates"]
    print(f"\nUnregistered on-table candidates (ts ref): {report['candidate_count']}")
    print(f"  (scorefix_c ref had 34 — see candidate_set_note in report)")
    print(f"Listed 15 verification in unmatched set: found {report['listed_verification']['found']}/15")

    print("\n--- Aggregate blocking gates (closest to passing) ---")
    print(f"{'Gate':<50} {'Left':>5} {'Right':>5} {'All':>5}")
    print("-" * 68)
    all_gates = sorted(
        set(list(agg["left"].keys()) + list(agg["right"].keys()) + list(agg["all"].keys()))
    )
    for gate in all_gates:
        print(
            f"{gate:<50} {agg['left'].get(gate, 0):>5} "
            f"{agg['right'].get(gate, 0):>5} {agg['all'].get(gate, 0):>5}"
        )

    if agg.get("dominant_gate"):
        print(f"\nDominant cause: {agg['dominant_gate']}")
    else:
        print("\nNo single dominant cause — misses spread across gates.")

    rec = report["recommendation"]
    print(f"\nSuggested change (analysis only): {rec['change']}")
    print(f"Risk: {rec['risk']}")

    print("\n--- Per-candidate (listed 15 + unmatched set) ---")
    for a in report["candidate_analyses"]:
        c = a["candidate"]
        if not c.get("listed_orchestrator") and not c.get("in_unmatched_set", True):
            continue
        side = c["side"][0].upper()
        tag = "LISTED" if c.get("listed_orchestrator") else "extra"
        in_set = c.get("in_unmatched_set", True)
        print(
            f"  [{tag}{'' if in_set else ',not-in-27'}] t={c['time']:.2f}s x={c['x']:.0f} {side} → "
            f"closest={a['closest_to_passing_gate']} | "
            f"detect_pass_any={a['any_detect_impact_passed']}"
        )
    print("=" * 72)


def main() -> None:
    with open(BALL_TRACK_PATH, encoding="utf-8") as f:
        cached = json.load(f)
    track = cached["track"]
    scene = cached["scene"]

    with open(REFERENCE_IMPACTS_PATH, encoding="utf-8") as f:
        reference = json.load(f)

    center_x = float(scene.get("net_center_x") or scene.get("table_center_x") or 600.0)
    table_poly_cv = np.array(scene["table_polygon"], dtype=np.float32).reshape(-1, 1, 2)
    net_poly = np.array(scene["net_polygon"], dtype=np.float32) if scene.get("net_polygon") else None
    net_bbox = scene.get("net_bbox")

    replayed, detection_log, _ = replay_track(track, scene)
    fidelity = match_impacts(replayed, reference)

    candidates = find_bounce_candidates(
        track, center_x, table_poly_cv, net_poly, net_bbox, reference
    )
    listed_verification = verify_listed_candidates(candidates)
    analysis_targets = resolve_listed_candidates(track, center_x, candidates)

    candidate_analyses: list[dict] = []
    if fidelity["fidelity_ok"]:
        for cand in analysis_targets:
            candidate_analyses.append(analyze_candidate_window(cand, detection_log, track))
        aggregate = aggregate_blocking_gates(
            [a for a in candidate_analyses if a["candidate"].get("in_unmatched_set", True)]
        )
        aggregate_truly_blocked = aggregate_blocking_gates(
            [a for a in candidate_analyses if a["candidate"].get("in_unmatched_set", True)],
            only_unblocked=True,
        )
        aggregate_listed = aggregate_blocking_gates(
            [a for a in candidate_analyses if a["candidate"].get("listed_orchestrator")]
        )
        recommendation = recommend_change(aggregate_truly_blocked)
    else:
        aggregate = {"skipped": True, "reason": "fidelity below 55 matched"}
        aggregate_listed = aggregate
        aggregate_truly_blocked = aggregate
        recommendation = {
            "change": "n/a",
            "detail": "Replay diverged from reference; gate conclusions withheld.",
            "risk": "n/a",
        }

    candidate_set_note = {
        "unmatched_on_table_vs_ts_reference": len(candidates),
        "unmatched_on_table_vs_scorefix_c_reference": 34,
        "note": (
            "34 count from bounce_recall used scorefix_c live_impacts; "
            "with ts reference (this trace) the same definition yields "
            f"{len(candidates)} unmatched on-table candidates."
        ),
    }

    report = {
        "trace_id": "wf:impact-gate-trace:2026-09-05T20:40:00Z",
        "plan_step_id": "A1",
        "reference_run": str(REFERENCE_IMPACTS_PATH.relative_to(PROJECT_ROOT)),
        "ball_track": str(BALL_TRACK_PATH.relative_to(PROJECT_ROOT)),
        "thresholds": {
            "y_range_min": 8,
            "curvature_min": 5,
            "min_consecutive_detections": 3,
            "min_frame_gap": 12,
            "min_velocity": 2.0,
            "trajectory_deque_maxlen": 10,
        },
        "fidelity": fidelity,
        "replayed_impacts": replayed,
        "candidate_count": len(candidates),
        "candidate_set_note": candidate_set_note,
        "candidates": candidates,
        "listed_verification": listed_verification,
        "candidate_analyses": candidate_analyses,
        "aggregate_blocking_gates": aggregate,
        "aggregate_blocking_gates_truly_blocked": aggregate_truly_blocked,
        "aggregate_blocking_gates_listed_15": aggregate_listed,
        "recommendation": recommendation,
        "gate_conclusions_valid": fidelity["fidelity_ok"],
    }

    report_path = OUTPUT_DIR / "trace_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved {report_path}")

    print_summary(report)


if __name__ == "__main__":
    main()
