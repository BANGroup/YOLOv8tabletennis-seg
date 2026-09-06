#!/usr/bin/env python3
"""Diagnose out-attribution flips: compare impacts in windows around 5 events."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "results/serve_detector_train14_img0291_dbgon"
CONSECFIX_DIR = Path(__file__).resolve().parent

# From run_stdout.log (both runs identical)
TABLE_X_MIN = 258
TABLE_X_MAX = 949
TABLE_Y_MIN = 222
TABLE_Y_MAX = 495
PLAY_X_MIN = 0
PLAY_X_MAX = 1249
PLAY_Y_MIN = 222
PLAY_Y_MAX = 495
CENTER_X = 600

FRAME_MATCH_TOLERANCE = 3
WINDOW_BEFORE_SEC = 3.0

EVENTS = [
    {"time": 52.33, "label": "52.33s P1→P2 flip"},
    {"time": 65.10, "label": "65.10s off-table flip (GT: P2 point)"},
    {"time": 74.07, "label": "74.07s P1→P2 flip"},
    {"time": 138.20, "label": "138.20s P1→P2 flip"},
    {"time": 169.23, "label": "169.23s event disappeared"},
]


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def impact_row(imp: dict) -> dict:
    inc = imp.get("incoming_speed") or 0.0
    out = imp.get("outgoing_speed") or 0.0
    ratio = (out / inc) if inc > 1e-6 else None
    x, y = imp.get("x"), imp.get("y")
    geo = geom_eval(x, y) if x is not None and y is not None else {}
    return {
        "frame": imp["frame"],
        "time": round(imp["time"], 3),
        "table_side": imp.get("table_side"),
        "x": round(x, 1) if x is not None else None,
        "y": round(y, 1) if y is not None else None,
        "curvature": round(imp.get("curvature", 0), 2),
        "y_range": round(imp.get("y_range", 0), 2),
        "incoming_speed": round(inc, 2),
        "outgoing_speed": round(out, 2),
        "speed_ratio_out_in": round(ratio, 3) if ratio is not None else None,
        "is_serve": imp.get("is_serve", False),
        **geo,
    }


def geom_eval(x: float, y: float) -> dict:
    in_table_bbox = (
        TABLE_X_MIN <= x <= TABLE_X_MAX and TABLE_Y_MIN <= y <= TABLE_Y_MAX
    )
    in_play_zone = (
        PLAY_X_MIN <= x <= PLAY_X_MAX and PLAY_Y_MIN <= y <= PLAY_Y_MAX
    )
    dist_left = x - TABLE_X_MIN
    dist_right = TABLE_X_MAX - x
    dist_top = y - TABLE_Y_MIN
    dist_bottom = TABLE_Y_MAX - y
    depth_inside = min(dist_left, dist_right, dist_top, dist_bottom)
    dist_to_center = abs(x - CENTER_X)
    half = "left" if x < CENTER_X else "right"
    dist_to_half_line = abs(x - CENTER_X)
    return {
        "in_table_bbox": in_table_bbox,
        "in_play_zone": in_play_zone,
        "depth_inside_px": round(depth_inside, 1) if in_table_bbox else None,
        "dist_to_nearest_table_edge_px": round(
            min(abs(dist_left), abs(dist_right), abs(dist_top), abs(dist_bottom))
            if not in_table_bbox
            else depth_inside,
            1,
        ),
        "dist_to_center_x_px": round(dist_to_center, 1),
        "dist_to_net_line_px": round(dist_to_half_line, 1),
        "geom_half": half,
    }


def window_impacts(impacts: list[dict], event_time: float, before: float) -> list[dict]:
    t0 = event_time - before
    return sorted(
        [i for i in impacts if t0 <= i["time"] <= event_time + 0.05],
        key=lambda x: x["time"],
    )


def match_impacts(
    baseline: list[dict], new: list[dict], tol: int = FRAME_MATCH_TOLERANCE
) -> tuple[list[tuple[dict | None, dict | None]], list[dict], list[dict]]:
    """Return (pairs, only_baseline, only_new)."""
    used_new: set[int] = set()
    pairs: list[tuple[dict | None, dict | None]] = []
    for b in baseline:
        best_j = None
        best_df = tol + 1
        for j, n in enumerate(new):
            if j in used_new:
                continue
            df = abs(n["frame"] - b["frame"])
            if df <= tol and df < best_df:
                best_df = df
                best_j = j
        if best_j is not None:
            used_new.add(best_j)
            pairs.append((b, new[best_j]))
        else:
            pairs.append((b, None))
    only_new = [n for j, n in enumerate(new) if j not in used_new]
    only_baseline = [b for b, n in pairs if n is None]
    return pairs, only_baseline, only_new


def reliable_stats(baseline: list[dict], new: list[dict]) -> dict:
    pairs, _, _ = match_impacts(baseline, new)
    shared = [(b, n) for b, n in pairs if b is not None and n is not None]
    curv = [b["curvature"] for b, _ in shared]
    yr = [b["y_range"] for b, _ in shared]
    ratios = []
    for b, _ in shared:
        inc, out = b.get("incoming_speed", 0), b.get("outgoing_speed", 0)
        if inc > 1e-6:
            ratios.append(out / inc)
    def stats(vals: list[float]) -> dict:
        if not vals:
            return {}
        return {
            "n": len(vals),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "median": round(statistics.median(vals), 3),
            "mean": round(statistics.mean(vals), 3),
        }
    return {
        "shared_impacts": len(shared),
        "curvature": stats(curv),
        "y_range": stats(yr),
        "speed_ratio_out_in": stats(ratios),
    }


def out_event_at(smart: dict, event_time: float, tol: float = 0.15) -> dict | None:
    for ev in smart.get("out_events", []):
        if abs(ev["time"] - event_time) < tol:
            return ev
    return None


def point_at(smart: dict, event_time: float, tol: float = 0.15) -> dict | None:
    for p in smart.get("point_history", []):
        if abs(p["time"] - event_time) < tol:
            return p
    return None


def analyze_event(
    event: dict,
    bl_impacts: list[dict],
    nw_impacts: list[dict],
    bl_smart: dict,
    nw_smart: dict,
) -> dict:
    t = event["time"]
    bl_win = window_impacts(bl_impacts, t, WINDOW_BEFORE_SEC)
    nw_win = window_impacts(nw_impacts, t, WINDOW_BEFORE_SEC)
    pairs, only_bl, only_nw = match_impacts(bl_win, nw_win)
    rel = reliable_stats(bl_impacts, nw_impacts)

    new_rows = [impact_row(n) for n in only_nw]
    for row in new_rows:
        cur = row["curvature"]
        yr = row["y_range"]
        sr = row["speed_ratio_out_in"]
        row["vs_reliable"] = {
            "curvature_vs_median": (
                round(cur / rel["curvature"]["median"], 2)
                if rel.get("curvature", {}).get("median") and cur
                else None
            ),
            "y_range_vs_median": (
                round(yr / rel["y_range"]["median"], 2)
                if rel.get("y_range", {}).get("median") and yr
                else None
            ),
            "speed_ratio_vs_median": (
                round(sr / rel["speed_ratio_out_in"]["median"], 2)
                if rel.get("speed_ratio_out_in", {}).get("median") and sr
                else None
            ),
        }

    bl_last = impact_row(bl_win[-1]) if bl_win else None
    nw_last = impact_row(nw_win[-1]) if nw_win else None

    return {
        "label": event["label"],
        "event_time": t,
        "window_sec_before": WINDOW_BEFORE_SEC,
        "baseline": {
            "impacts_in_window": [impact_row(i) for i in bl_win],
            "last_impact_before_out": bl_last,
            "out_event": out_event_at(bl_smart, t),
            "point": point_at(bl_smart, t),
        },
        "consecfix": {
            "impacts_in_window": [impact_row(i) for i in nw_win],
            "last_impact_before_out": nw_last,
            "out_event": out_event_at(nw_smart, t),
            "point": point_at(nw_smart, t),
        },
        "diff": {
            "only_in_baseline": [impact_row(i) for i in only_bl],
            "only_in_consecfix": new_rows,
            "matched_pairs_count": sum(1 for b, n in pairs if b and n),
        },
        "reliable_reference_stats": rel,
        "attribution_note": (
            f"baseline last_side={bl_last['table_side'] if bl_last else None} "
            f"→ fault P{bl_last['table_side'] and (1 if bl_last['table_side']=='left' else 2)}; "
            f"consecfix last_side={nw_last['table_side'] if nw_last else None} "
            f"→ fault P{nw_last['table_side'] and (1 if nw_last['table_side']=='left' else 2)}"
            if bl_last or nw_last
            else None
        ),
    }


def analyze_169_disappearance(
    bl_impacts: list[dict],
    nw_impacts: list[dict],
    bl_smart: dict,
    nw_smart: dict,
) -> dict:
    t = 169.23
    wide = 8.0
    bl_w = window_impacts(bl_impacts, t + 2, wide)
    nw_w = window_impacts(nw_impacts, t + 2, wide)
    _, _, only_nw = match_impacts(bl_w, nw_w)
    # Points near 169 in both runs
    bl_points = [p for p in bl_smart.get("point_history", []) if 160 <= p["time"] <= 180]
    nw_points = [p for p in nw_smart.get("point_history", []) if 160 <= p["time"] <= 180]
    bl_outs = [o for o in bl_smart.get("out_events", []) if 160 <= o["time"] <= 180]
    nw_outs = [o for o in nw_smart.get("out_events", []) if 160 <= o["time"] <= 180]
    return {
        "baseline_points_160_180": bl_points,
        "consecfix_points_160_180": nw_points,
        "baseline_outs_160_180": bl_outs,
        "consecfix_outs_160_180": nw_outs,
        "new_impacts_in_wide_window": [impact_row(i) for i in only_nw],
        "consecfix_impacts_165_172": [
            impact_row(i)
            for i in nw_impacts
            if 165 <= i["time"] <= 172
        ],
        "baseline_impacts_165_172": [
            impact_row(i)
            for i in bl_impacts
            if 165 <= i["time"] <= 172
        ],
    }


def main() -> None:
    bl_impacts = load_json(BASELINE_DIR / "live_impacts.json")
    nw_impacts = load_json(CONSECFIX_DIR / "live_impacts.json")
    bl_smart = load_json(BASELINE_DIR / "smart_impacts_analysis.json")
    nw_smart = load_json(CONSECFIX_DIR / "smart_impacts_analysis.json")

    global_stats = reliable_stats(bl_impacts, nw_impacts)
    _, _, all_new = match_impacts(bl_impacts, nw_impacts)

    report = {
        "table_geometry": {
            "table_bbox": f"X[{TABLE_X_MIN}-{TABLE_X_MAX}], Y[{TABLE_Y_MIN}-{TABLE_Y_MAX}]",
            "play_zone": f"X[{PLAY_X_MIN}-{PLAY_X_MAX}], Y[{PLAY_Y_MIN}-{PLAY_Y_MAX}]",
            "center_x": CENTER_X,
        },
        "global_reliable_stats": global_stats,
        "all_new_impacts_vs_baseline": [impact_row(i) for i in sorted(all_new, key=lambda x: x["time"])],
        "events": [],
    }

    for ev in EVENTS:
        if ev["time"] == 169.23:
            detail = analyze_event(ev, bl_impacts, nw_impacts, bl_smart, nw_smart)
            detail["disappearance_analysis"] = analyze_169_disappearance(
                bl_impacts, nw_impacts, bl_smart, nw_smart
            )
            report["events"].append(detail)
        else:
            report["events"].append(
                analyze_event(ev, bl_impacts, nw_impacts, bl_smart, nw_smart)
            )

    out_path = CONSECFIX_DIR / "flip_events_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Global reliable stats: {json.dumps(global_stats, ensure_ascii=False)}")
    print(f"Total new impacts (full video): {len(all_new)}")
    print(f"Report: {out_path}\n")

    for ev in report["events"]:
        print("=" * 60)
        print(ev["label"], f"@ {ev['event_time']}s")
        print(f"  baseline impacts in window: {len(ev['baseline']['impacts_in_window'])}")
        print(f"  consecfix impacts in window: {len(ev['consecfix']['impacts_in_window'])}")
        for row in ev["diff"]["only_in_consecfix"]:
            print(
                f"  NEW t={row['time']} f={row['frame']} side={row['table_side']} "
                f"x={row['x']} y={row['y']} curv={row['curvature']} yr={row['y_range']} "
                f"spd_ratio={row['speed_ratio_out_in']} in_table={row['in_table_bbox']} "
                f"depth={row['depth_inside_px']} dist_net={row['dist_to_net_line_px']}"
            )
        bl_out = ev["baseline"].get("out_event")
        nw_out = ev["consecfix"].get("out_event")
        print(f"  baseline out: fault P{bl_out['player_fault'] if bl_out else '—'}")
        print(f"  consecfix out: fault P{nw_out['player_fault'] if nw_out else '—'}")
        if ev.get("attribution_note"):
            print(f"  {ev['attribution_note']}")


if __name__ == "__main__":
    main()
