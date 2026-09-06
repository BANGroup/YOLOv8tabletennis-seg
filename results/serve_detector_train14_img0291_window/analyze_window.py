#!/usr/bin/env python3
"""Compare baseline / streak(consecfix) / window bounce-filter variants."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOUNCE_CLUSTER_GAP = 5
FRAME_MATCH_TOLERANCE = 5  # for baseline smart_impact survival matching


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


def cluster_reject_reason(records: list[dict]) -> str | None:
    reasons = [r.get("process_reject") for r in records if r.get("process_reject")]
    if not reasons:
        return None
    return Counter(reasons).most_common(1)[0][0]


def analyze_trace(trace_path: Path) -> dict[str, Any]:
    trace = json.load(open(trace_path, encoding="utf-8"))
    candidates = [r for r in trace if r.get("impact")]
    clusters = cluster_by_bounce_frame(candidates)

    accepted = [cl for cl in clusters if is_accepted(cl["records"])]
    lost = [cl for cl in clusters if not is_accepted(cl["records"])]

    lost_reject = Counter(cluster_reject_reason(cl["records"]) for cl in lost)
    return {
        "physical_bounces": len(clusters),
        "accepted": len(accepted),
        "lost": len(lost),
        "lost_reject": dict(lost_reject),
        "accepted_frames": sorted(cl["bounce_frame"] for cl in accepted),
    }


def load_live_impacts(path: Path) -> list[dict]:
    return json.load(open(path, encoding="utf-8"))


def load_smart_analysis(path: Path, stdout_path: Path | None = None) -> dict[str, Any]:
    data = json.load(open(path, encoding="utf-8"))
    left = sum(1 for d in data.get("impacts_details", []) if d.get("side") == "left")
    right = sum(1 for d in data.get("impacts_details", []) if d.get("side") == "right")
    if stdout_path and stdout_path.exists():
        text = stdout_path.read_text(encoding="utf-8")
        m = re.search(r"Left:\s+(\d+)\s*\n\s*└─ Right:\s+(\d+)", text)
        if m:
            left, right = int(m.group(1)), int(m.group(2))
    score = data.get("score", {})
    if isinstance(score, dict) and "player1" in score:
        score_str = f"{score['player1']}:{score['player2']}"
    else:
        score_str = str(score)
    return {
        "smart_impacts": data.get("smart_impacts"),
        "left": left,
        "right": right,
        "score": score,
        "score_str": score_str,
        "point_history": data.get("point_history", []),
        "net_events": len(data.get("net_events", [])),
        "out_events": len(data.get("out_events", [])),
        "double_bounce_events": len(data.get("double_bounce_events", [])),
        "num_points": len(data.get("point_history", [])),
    }


def count_baseline_survived(baseline_impacts: list[dict], variant_impacts: list[dict], tol: int) -> dict[str, Any]:
    variant_frames = [i["frame"] for i in variant_impacts]
    survived = []
    lost = []
    for bi in baseline_impacts:
        bf = bi["frame"]
        match = any(abs(vf - bf) <= tol for vf in variant_frames)
        entry = {"frame": bf, "time": bi.get("time"), "side": bi.get("table_side")}
        if match:
            survived.append(entry)
        else:
            lost.append(entry)
    return {
        "baseline_smart_impacts": len(baseline_impacts),
        "survived": len(survived),
        "lost": len(lost),
        "survival_rate": f"{len(survived)}/{len(baseline_impacts)}",
        "lost_frames": lost,
    }


def added_vs_baseline(baseline_impacts: list[dict], variant_impacts: list[dict], tol: int) -> int:
    baseline_frames = [i["frame"] for i in baseline_impacts]
    added = 0
    for vi in variant_impacts:
        vf = vi["frame"]
        if not any(abs(bf - vf) <= tol for bf in baseline_frames):
            added += 1
    return added


def format_point_row(i: int, p: dict) -> str:
    t = p.get("time", 0)
    winner = p.get("winner", p.get("player"))
    reason = p.get("reason", p.get("cause", ""))
    score = p.get("score", p.get("score_after"))
    return f"#{i:2d} t={t:6.2f}s P{winner} ({reason}) score={score}"


def find_impact_near(impacts: list[dict], frame: int, tol: int = 3) -> dict | None:
    for imp in impacts:
        if abs(imp["frame"] - frame) <= tol:
            return imp
    return None


def find_point_near(points: list[dict], time: float, tol: float = 0.15) -> dict | None:
    for p in points:
        if abs(p.get("time", 0) - time) <= tol:
            return p
    return None


def main() -> None:
    dirs = {
        "baseline": PROJECT_ROOT / "results/serve_detector_train14_img0291_dbgon",
        "streak": PROJECT_ROOT / "results/serve_detector_train14_img0291_consecfix",
        "window": PROJECT_ROOT / "results/serve_detector_train14_img0291_window",
    }

    traces = {name: analyze_trace(d / "debug_trace.json") for name, d in dirs.items()}
    smarts = {
        name: load_smart_analysis(d / "smart_impacts_analysis.json", d / "run_stdout.log")
        for name, d in dirs.items()
    }
    live = {name: load_live_impacts(d / "live_impacts.json") for name, d in dirs.items()}

    baseline_live = live["baseline"]
    survival = {
        name: count_baseline_survived(baseline_live, live[name], FRAME_MATCH_TOLERANCE)
        for name in ("streak", "window")
    }
    added = {
        name: added_vs_baseline(baseline_live, live[name], FRAME_MATCH_TOLERANCE)
        for name in ("streak", "window")
    }

    # acceptance criteria
    f586 = {
        name: find_impact_near(live[name], 586, tol=3) for name in dirs
    }
    p6510 = {
        name: find_point_near(smarts[name]["point_history"], 65.10, tol=0.15) for name in dirs
    }
    acceptance = {
        "f586_in_live_impacts": f586["window"] is not None,
        "f586_side": f586["window"].get("table_side") if f586["window"] else None,
        "f586_time": f586["window"].get("time") if f586["window"] else None,
        "f586_y_range": f586["window"].get("y_range") if f586["window"] else None,
        "point_65_10_winner": p6510["window"].get("winner") if p6510["window"] else None,
        "point_65_10_reason": p6510["window"].get("reason") if p6510["window"] else None,
        "pass": f586["window"] is not None and p6510["window"] is not None and p6510["window"].get("winner") == 2,
    }

    report = {
        "cluster_gap_frames": BOUNCE_CLUSTER_GAP,
        "frame_match_tolerance": FRAME_MATCH_TOLERANCE,
        "acceptance_criteria": acceptance,
        "traces": traces,
        "smarts": {k: {kk: vv for kk, vv in v.items() if kk != "point_history"} for k, v in smarts.items()},
        "baseline_survival": survival,
        "added_vs_baseline": added,
        "point_history": {name: smarts[name]["point_history"] for name in dirs},
        "f586_by_variant": {
            name: (
                {
                    "frame": f586[name]["frame"],
                    "time": f586[name]["time"],
                    "table_side": f586[name].get("table_side"),
                    "y_range": f586[name].get("y_range"),
                }
                if f586[name]
                else None
            )
            for name in dirs
        },
        "point_65_10_by_variant": {
            name: (
                {
                    "time": p6510[name]["time"],
                    "winner": p6510[name]["winner"],
                    "reason": p6510[name]["reason"],
                    "score": p6510[name]["score"],
                }
                if p6510[name]
                else None
            )
            for name in dirs
        },
    }

    out_json = dirs["window"] / "window_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    labels = ["baseline", "streak", "window"]
    print("=" * 72)
    print("ACCEPTANCE CRITERIA")
    print("=" * 72)
    print(f"  f586 in window live_impacts: {'YES' if acceptance['f586_in_live_impacts'] else 'NO'}")
    if f586["window"]:
        w = f586["window"]
        print(f"    frame={w['frame']} time={w['time']:.2f}s side={w.get('table_side')} y_range={w.get('y_range', 0):.1f}")
    print(f"  point @65.10s winner P{acceptance['point_65_10_winner']} ({acceptance['point_65_10_reason']})")
    print(f"  OVERALL: {'PASS' if acceptance['pass'] else 'FAIL'}")

    print("\n" + "=" * 72)
    print("CLUSTER COMPARISON (gap=%d)" % BOUNCE_CLUSTER_GAP)
    print("=" * 72)
    metrics = [
        ("physical_bounces", lambda t: t["physical_bounces"]),
        ("accepted", lambda t: t["accepted"]),
        ("lost", lambda t: t["lost"]),
        ("lost/consecutive_detections", lambda t: t["lost_reject"].get("consecutive_detections", 0)),
        ("lost/min_frame_gap", lambda t: t["lost_reject"].get("min_frame_gap", 0)),
    ]
    print(f"{'metric':<30} {'baseline':>10} {'streak':>10} {'window':>10}")
    for name, fn in metrics:
        vals = [fn(traces[l]) for l in labels]
        print(f"{name:<30} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    print("\n" + "=" * 72)
    print("SMART IMPACTS & SCORING")
    print("=" * 72)
    print(f"{'metric':<30} {'baseline':>10} {'streak':>10} {'window':>10}")
    smart_metrics = [
        ("smart_impacts", lambda s: s["smart_impacts"]),
        ("left", lambda s: s["left"]),
        ("right", lambda s: s["right"]),
        ("score P1:P2", lambda s: s["score_str"]),
        ("num_points", lambda s: s["num_points"]),
        ("net_events", lambda s: s["net_events"]),
        ("out_events", lambda s: s["out_events"]),
        ("double_bounce", lambda s: s["double_bounce_events"]),
    ]
    for name, fn in smart_metrics:
        vals = [fn(smarts[l]) for l in labels]
        print(f"{name:<30} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    print("\n" + "=" * 72)
    print(f"BASELINE SMART IMPACT SURVIVAL (tol=±{FRAME_MATCH_TOLERANCE} frames)")
    print("=" * 72)
    print(f"  baseline smart_impacts: {len(baseline_live)}")
    for name in ("streak", "window"):
        s = survival[name]
        print(f"  {name}: survived {s['survival_rate']}, added vs baseline: +{added[name]}")

    if survival["window"]["lost_frames"]:
        print(f"\n  window lost baseline impacts ({len(survival['window']['lost_frames'])}):")
        for lf in survival["window"]["lost_frames"][:15]:
            print(f"    f={lf['frame']} t={lf['time']:.2f}s side={lf['side']}")
        if len(survival["window"]["lost_frames"]) > 15:
            print(f"    ... +{len(survival['window']['lost_frames']) - 15} more")

    print("\n" + "=" * 72)
    print("POINT HISTORY (all variants)")
    print("=" * 72)
    max_pts = max(len(smarts[l]["point_history"]) for l in labels)
    print(f"{'#':>3}  {'baseline':<42} {'streak':<42} {'window':<42}")
    for i in range(max_pts):
        rows = []
        for l in labels:
            ph = smarts[l]["point_history"]
            if i < len(ph):
                p = ph[i]
                t = p.get("time", 0)
                w = p.get("winner")
                r = p.get("reason", "")
                marker = " *" if abs(t - 65.10) < 0.15 else ""
                rows.append(f"t={t:5.2f}s P{w} {r[:22]}{marker}")
            else:
                rows.append("-")
        print(f"{i+1:3d}  {rows[0]:<42} {rows[1]:<42} {rows[2]:<42}")

    print(f"\nReport: {out_json}")


if __name__ == "__main__":
    main()
