#!/usr/bin/env python3
"""Валидация netparams vs baseline window (serve_detector_train14_img0291)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = OUT_DIR.parent / "serve_detector_train14_img0291_window"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_lr(live_path: Path) -> tuple[int, int]:
    live = load_json(live_path)
    left = sum(1 for i in live if i.get("table_side") == "left")
    right = sum(1 for i in live if i.get("table_side") == "right")
    return left, right


def net_contacts_in_window(trace: list[dict], t0: float, t1: float) -> list[dict]:
    out = []
    for rec in trace:
        if t0 <= rec["time"] <= t1 and rec.get("routed") == "net_contact":
            ball = rec.get("ball")
            out.append({
                "frame": rec["frame"],
                "time": round(rec["time"], 3),
                "x": round(ball["x"], 1) if ball else None,
            })
    return out


def main() -> None:
    bl = load_json(BASELINE_DIR / "smart_impacts_analysis.json")
    nw = load_json(OUT_DIR / "smart_impacts_analysis.json")
    bl_trace = load_json(BASELINE_DIR / "debug_trace.json")
    nw_trace = load_json(OUT_DIR / "debug_trace.json")

    bl_lr = count_lr(BASELINE_DIR / "live_impacts.json")
    nw_lr = count_lr(OUT_DIR / "live_impacts.json")

    c1 = not any(
        102.5 < p["time"] < 103.1 and "net" in p["reason"].lower()
        for p in nw["point_history"]
    )
    c2 = any(
        108 <= p["time"] <= 111 and "net" in p["reason"].lower()
        for p in nw["point_history"]
    ) or any(108 <= e["time"] <= 111 for e in nw["net_events"])

    report = {
        "trace_id": f"wf:net_params:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "plan_step_id": "V1",
        "acceptance": {
            "criterion_1_no_false_net_102_80": {"pass": c1},
            "criterion_2_real_net_109s": {"pass": c2},
        },
        "overall": "PASS" if (c1 and c2) else ("PARTIAL_PASS" if c1 else "FAIL"),
        "score": {"baseline": bl["score"], "new": nw["score"]},
        "counts": {
            "smart_impacts": {"baseline": bl["smart_impacts"], "new": nw["smart_impacts"]},
            "net_events": {"baseline": len(bl["net_events"]), "new": len(nw["net_events"])},
            "out_events": {"baseline": len(bl["out_events"]), "new": len(nw["out_events"])},
            "double_bounce_events": {
                "baseline": len(bl["double_bounce_events"]),
                "new": len(nw["double_bounce_events"]),
            },
            "smart_lr": {
                "baseline": f"{bl_lr[0]}/{bl_lr[1]}",
                "new": f"{nw_lr[0]}/{nw_lr[1]}",
            },
        },
        "net_events": {"baseline": bl["net_events"], "new": nw["net_events"]},
        "point_history": {"baseline": bl["point_history"], "new": nw["point_history"]},
        "debug_trace_109s": {
            "new_net_contact": net_contacts_in_window(nw_trace, 107.5, 111.0),
            "baseline_net_contact": net_contacts_in_window(bl_trace, 107.5, 111.0),
            "rally_end_point": next(
                (p for p in nw["point_history"] if 109 <= p["time"] <= 110),
                None,
            ),
        },
    }

    out_path = OUT_DIR / "validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Overall: {report['overall']}")
    print(f"C1 (no false net 102.80): {'PASS' if c1 else 'FAIL'}")
    print(f"C2 (real net ~109s): {'PASS' if c2 else 'FAIL'}")
    print(f"Score baseline {bl['score']} -> new {nw['score']}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
