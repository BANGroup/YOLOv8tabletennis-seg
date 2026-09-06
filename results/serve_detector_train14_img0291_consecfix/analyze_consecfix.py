#!/usr/bin/env python3
"""Analyze debug_trace bounce clusters for consecutive_detections fix validation."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOUNCE_CLUSTER_GAP = 5  # same rule as baseline analysis (gap=5 on bounce_frame)
MIN_CONSECUTIVE = 3


def cluster_by_bounce_frame(records: list[dict], gap: int = BOUNCE_CLUSTER_GAP) -> list[dict]:
    recs = sorted(records, key=lambda r: r["impact"]["bounce_frame"])
    clusters: list[dict] = []
    for r in recs:
        bf = r["impact"]["bounce_frame"]
        if clusters and bf - clusters[-1]["bounce_frame"] <= gap:
            clusters[-1]["records"].append(r)
            clusters[-1]["bounce_frame"] = min(clusters[-1]["bounce_frame"], bf)
        else:
            clusters.append({"bounce_frame": bf, "records": [r]})
    return clusters


def is_accepted(records: list[dict]) -> bool:
    return any(
        (r.get("routed") == "process_detection" and r.get("process_reject") is None)
        or r.get("routed") == "net_contact"
        for r in records
    )


def cluster_side(records: list[dict]) -> str:
    sides = [r["impact"].get("table_side") for r in records if r.get("impact")]
    sides = [s for s in sides if s in ("left", "right")]
    if not sides:
        x_vals = [r["impact"]["x"] for r in records if r.get("impact")]
        if x_vals:
            return "left" if sum(x_vals) / len(x_vals) < 599.5 else "right"
        return "unknown"
    return Counter(sides).most_common(1)[0][0]


def cluster_reject_reason(records: list[dict]) -> str | None:
    reasons = [r.get("process_reject") for r in records if r.get("process_reject")]
    if not reasons:
        return None
    return Counter(reasons).most_common(1)[0][0]


def consecutive_ball_frames_before_candidate(trace: list[dict], cand_frame: int) -> int:
    """Count consecutive ball-detected frames immediately before candidate frame."""
    streak = 0
    for rec in reversed(trace):
        if rec["frame"] >= cand_frame:
            continue
        if rec.get("ball") is not None:
            streak += 1
        else:
            break
    return streak


def analyze_trace(trace_path: Path) -> dict[str, Any]:
    trace = json.load(open(trace_path, encoding="utf-8"))
    candidates = [r for r in trace if r.get("impact")]
    clusters = cluster_by_bounce_frame(candidates)

    accepted = []
    lost = []
    for cl in clusters:
        if is_accepted(cl["records"]):
            accepted.append(cl)
        else:
            lost.append(cl)

    lost_reject = Counter(cluster_reject_reason(cl["records"]) for cl in lost)
    lost_side = Counter(cluster_side(cl["records"]) for cl in lost)

    # consecutive streak at first candidate in each lost cluster
    streak_hist = Counter()
    would_pass_at_2 = 0
    for cl in lost:
        first = min(cl["records"], key=lambda r: r["frame"])
        streak = consecutive_ball_frames_before_candidate(trace, first["frame"])
        bucket = streak if streak <= 4 else "5+"
        streak_hist[bucket] += 1
        if streak >= 2:
            would_pass_at_2 += 1

    return {
        "trace_path": str(trace_path),
        "trace_records": len(trace),
        "impact_candidates": len(candidates),
        "unique_bounce_frame": len({r["impact"]["bounce_frame"] for r in candidates}),
        "physical_bounces": len(clusters),
        "accepted": len(accepted),
        "lost": len(lost),
        "lost_reject": dict(lost_reject),
        "lost_side": dict(lost_side),
        "lost_streak_distribution": dict(sorted(streak_hist.items(), key=lambda x: (999 if x[0] == "5+" else x[0]))),
        "lost_would_pass_threshold_2": would_pass_at_2,
    }


def load_smart_analysis(path: Path) -> dict[str, Any]:
    data = json.load(open(path, encoding="utf-8"))
    left = sum(1 for d in data.get("impacts_details", []) if d.get("side") == "left")
    right = sum(1 for d in data.get("impacts_details", []) if d.get("side") == "right")
    # impacts_details may not have side; derive from stdout or live if needed
    if left == 0 and right == 0:
        left = data.get("smart_impacts_left")
        right = data.get("smart_impacts_right")
    return {
        "smart_impacts": data.get("smart_impacts"),
        "score": data.get("score"),
        "point_history": data.get("point_history", []),
        "net_events": len(data.get("net_events", [])),
        "out_events": len(data.get("out_events", [])),
        "double_bounce_events": len(data.get("double_bounce_events", [])),
        "left": left,
        "right": right,
        "raw": data,
    }


def parse_stdout_lr(path: Path) -> tuple[int | None, int | None]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"Left:\s+(\d+)\s*\n\s*└─ Right:\s+(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def format_point_history(points: list[dict]) -> list[str]:
    rows = []
    for i, p in enumerate(points, 1):
        t = p.get("time", p.get("src_time"))
        winner = p.get("winner", p.get("player"))
        reason = p.get("reason", p.get("cause"))
        score = p.get("score_after", p.get("score"))
        rows.append(f"#{i} t={t:.2f}s P{winner} ({reason}) score={score}")
    return rows


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    baseline_dir = PROJECT_ROOT / "results/serve_detector_train14_img0291_dbgon"

    baseline_trace = analyze_trace(baseline_dir / "debug_trace.json")
    new_trace = analyze_trace(out_dir / "debug_trace.json")

    baseline_smart = load_smart_analysis(baseline_dir / "smart_impacts_analysis.json")
    new_smart = load_smart_analysis(out_dir / "smart_impacts_analysis.json")

    bl_lr = parse_stdout_lr(baseline_dir / "run_stdout.log")
    nw_lr = parse_stdout_lr(out_dir / "run_stdout.log")
    if bl_lr[0] is not None:
        baseline_smart["left"], baseline_smart["right"] = bl_lr
    if nw_lr[0] is not None:
        new_smart["left"], new_smart["right"] = nw_lr

    report = {
        "cluster_gap_frames": BOUNCE_CLUSTER_GAP,
        "baseline": {"trace": baseline_trace, "smart": baseline_smart},
        "consecfix": {"trace": new_trace, "smart": new_smart},
        "comparison_table": {
            "physical_bounces": [baseline_trace["physical_bounces"], new_trace["physical_bounces"]],
            "accepted": [baseline_trace["accepted"], new_trace["accepted"]],
            "lost": [baseline_trace["lost"], new_trace["lost"]],
            "lost_consecutive_detections": [
                baseline_trace["lost_reject"].get("consecutive_detections", 0),
                new_trace["lost_reject"].get("consecutive_detections", 0),
            ],
            "lost_min_frame_gap": [
                baseline_trace["lost_reject"].get("min_frame_gap", 0),
                new_trace["lost_reject"].get("min_frame_gap", 0),
            ],
        },
    }

    out_json = out_dir / "consecfix_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== CLUSTER COMPARISON (gap=%d) ===" % BOUNCE_CLUSTER_GAP)
    hdr = ["metric", "baseline_dbgon", "consecfix"]
    rows = [
        ("physical_bounces", baseline_trace["physical_bounces"], new_trace["physical_bounces"]),
        ("accepted", baseline_trace["accepted"], new_trace["accepted"]),
        ("lost", baseline_trace["lost"], new_trace["lost"]),
        ("lost/consecutive_detections", baseline_trace["lost_reject"].get("consecutive_detections", 0), new_trace["lost_reject"].get("consecutive_detections", 0)),
        ("lost/min_frame_gap", baseline_trace["lost_reject"].get("min_frame_gap", 0), new_trace["lost_reject"].get("min_frame_gap", 0)),
    ]
    print(f"{'metric':<28} {'baseline':>10} {'consecfix':>10} {'delta':>10}")
    for name, b, n in rows:
        print(f"{name:<28} {b:>10} {n:>10} {n-b:>+10}")

    print("\n=== SMART IMPACTS ===")
    for label, sm in [("baseline", baseline_smart), ("consecfix", new_smart)]:
        print(f"{label}: smart_impacts={sm['smart_impacts']} L/R={sm['left']}/{sm['right']} score={sm['score']} net={sm['net_events']} out={sm['out_events']} double={sm['double_bounce_events']}")

    print("\n=== POINT HISTORY baseline ===")
    for line in format_point_history(baseline_smart["point_history"]):
        print(" ", line)
    print("\n=== POINT HISTORY consecfix ===")
    for line in format_point_history(new_smart["point_history"]):
        print(" ", line)

    if new_trace["accepted"] < baseline_trace["accepted"]:
        print("\n=== REGRESSION DIAG: lost cluster streak distribution (consecfix) ===")
        print(new_trace["lost_streak_distribution"])
        print("would_pass_at_threshold_2:", new_trace["lost_would_pass_threshold_2"], "/", new_trace["lost"])

    print(f"\nReport: {out_json}")


if __name__ == "__main__":
    main()
