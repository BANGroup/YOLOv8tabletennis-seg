#!/usr/bin/env python3
"""Анализ net_events vs промах ~109s по debug_trace.json (bbox-proxy)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

OUT_DIR = Path(__file__).resolve().parent
TRACE_PATH = OUT_DIR / "debug_trace.json"
LIVE_PATH = OUT_DIR / "live_impacts.json"
SMART_PATH = OUT_DIR / "smart_impacts_analysis.json"
OUT_JSON = OUT_DIR / "net_events_analysis.json"

# bbox-proxy из лога сегментации сетки
NET_BBOX = [570, 302, 633, 453]
NET_IMPACT_MARGIN = 25.0
NET_ZONE_MARGIN = 40.0

NET_EVENTS = [
    {"time": 102.80, "frame": 1734, "player": 2},
    {"time": 117.17, "frame": 2165, "player": 2},
    {"time": 120.60, "frame": 2268, "player": 1},
    {"time": 153.80, "frame": 3264, "player": 2},
]

MISS_WINDOW = (107.5, 111.0)
RALLY_WINDOW = (101.0, 110.0)


def bbox_proxy_dist(x: float, y: float) -> dict[str, Any]:
    x1, y1, x2, y2 = NET_BBOX
    inside = x1 <= x <= x2 and y1 <= y <= y2
    dx = 0.0 if x1 <= x <= x2 else min(abs(x - x1), abs(x - x2))
    dy = 0.0 if y1 <= y <= y2 else min(abs(y - y1), abs(y - y2))
    inside_impact = (
        (x1 - NET_IMPACT_MARGIN) <= x <= (x2 + NET_IMPACT_MARGIN)
        and (y1 - NET_IMPACT_MARGIN) <= y <= (y2 + NET_IMPACT_MARGIN)
    )
    inside_zone = (
        (x1 - NET_ZONE_MARGIN) <= x <= (x2 + NET_ZONE_MARGIN)
        and (y1 - NET_ZONE_MARGIN) <= y <= (y2 + NET_ZONE_MARGIN)
    )
    return {
        "inside_bbox": inside,
        "dx_edge": round(dx, 1),
        "dy_edge": round(dy, 1),
        "inside_impact_margin": inside_impact,
        "inside_zone_margin": inside_zone,
    }


def vx_sign_reversals(frames: list[dict], x_center: float = 600.0, x_tol: float = 80.0) -> list[dict]:
    """Развороты vx между соседними кадрами с ball != null."""
    reversals = []
    prev_x: float | None = None
    prev_vx: float | None = None
    for rec in frames:
        ball = rec.get("ball")
        if ball is None:
            prev_x = None
            prev_vx = None
            continue
        if prev_x is not None:
            vx = ball["x"] - prev_x
            if prev_vx is not None and prev_vx * vx < 0:
                near_net = abs(ball["x"] - x_center) <= x_tol
                reversals.append({
                    "frame": rec["frame"],
                    "time": round(rec["time"], 3),
                    "x": round(ball["x"], 1),
                    "y": round(ball["y"], 1),
                    "vx_before": round(prev_vx, 2),
                    "vx_after": round(vx, 2),
                    "near_net_x600": near_net,
                })
            prev_vx = vx
        else:
            prev_vx = None
        prev_x = ball["x"]
    return reversals


def last_n_ball_frames(trace: list[dict], end_frame: int | None, end_time: float | None, n: int = 15) -> list[dict]:
    subset = []
    for rec in trace:
        if end_frame is not None and rec["frame"] > end_frame:
            break
        if end_time is not None and rec["time"] > end_time:
            break
        if rec.get("ball") is not None:
            subset.append(rec)
    return subset[-n:]


def summarize_window(trace: list[dict], t0: float, t1: float) -> dict[str, Any]:
    win = [r for r in trace if t0 <= r["time"] <= t1]
    ball_frames = [r for r in win if r.get("ball") is not None]
    max_missed = max((r.get("missed_frames", 0) for r in win), default=0)
    routed = Counter(r.get("routed") for r in win)
    impacts = [r for r in win if r.get("impact")]
    net_routed = [r for r in win if r.get("routed") == "net_contact"]
    detect_called = sum(1 for r in win if r.get("detect_impact_called"))
    reject_reasons = Counter(r.get("reject_reason") for r in win if r.get("reject_reason"))
    process_rejects = Counter(r.get("process_reject") for r in win if r.get("process_reject"))

    reversals = vx_sign_reversals(win)
    near_net_reversals = [r for r in reversals if r["near_net_x600"]]

    # bbox-proxy на всех ball-кадрах
    proxy_stats = []
    for r in ball_frames:
        p = bbox_proxy_dist(r["ball"]["x"], r["ball"]["y"])
        proxy_stats.append({
            "frame": r["frame"],
            "time": round(r["time"], 3),
            "x": round(r["ball"]["x"], 1),
            "y": round(r["ball"]["y"], 1),
            **p,
        })

    min_dx = min((p["dx_edge"] for p in proxy_stats), default=None)
    closest = sorted(proxy_stats, key=lambda p: (p["dx_edge"], p["dy_edge"]))[:5]

    return {
        "time_range": [t0, t1],
        "total_frames": len(win),
        "ball_visible_frames": len(ball_frames),
        "max_missed_frames": max_missed,
        "ball_continuous": max_missed <= 5,
        "routed_counts": dict(routed),
        "detect_impact_called_count": detect_called,
        "impact_count": len(impacts),
        "impacts": [
            {
                "frame": r["frame"],
                "time": round(r["time"], 3),
                "routed": r.get("routed"),
                "reject_reason": r.get("reject_reason"),
                "process_reject": r.get("process_reject"),
                "impact": r.get("impact"),
                "bbox_proxy": bbox_proxy_dist(r["impact"]["x"], r["impact"]["y"]) if r.get("impact") else None,
            }
            for r in impacts
        ],
        "net_contact_frames": [
            {"frame": r["frame"], "time": r["time"], "impact": r.get("impact")}
            for r in net_routed
        ],
        "reject_reasons": dict(reject_reasons),
        "process_rejects": dict(process_rejects),
        "vx_reversals": reversals,
        "vx_reversals_near_net": near_net_reversals,
        "min_dx_to_net_bbox": min_dx,
        "closest_to_net_bbox": closest,
        "last_15_ball_frames": format_ball_chain(last_n_ball_frames(win, None, t1, 15)),
    }


def format_ball_chain(frames: list[dict]) -> list[dict]:
    out = []
    prev_x = None
    for r in frames:
        ball = r["ball"]
        vx = round(ball["x"] - prev_x, 2) if prev_x is not None else None
        entry = {
            "frame": r["frame"],
            "time": round(r["time"], 3),
            "x": round(ball["x"], 1),
            "y": round(ball["y"], 1),
            "vx": vx,
            "missed_frames": r.get("missed_frames"),
            "detect_impact_called": r.get("detect_impact_called"),
            "reject_reason": r.get("reject_reason"),
            "routed": r.get("routed"),
            "process_reject": r.get("process_reject"),
            "bbox_proxy": bbox_proxy_dist(ball["x"], ball["y"]),
        }
        if r.get("impact"):
            entry["impact"] = r["impact"]
        out.append(entry)
        prev_x = ball["x"]
    return out


def analyze_event(trace: list[dict], event: dict) -> dict[str, Any]:
    ef = event["frame"]
    et = event["time"]
    last15 = last_n_ball_frames(trace, ef, et, 15)
    event_rec = next((r for r in trace if r["frame"] == ef), None)
    # кадры с missed>5 около события
    around = [r for r in trace if ef - 30 <= r["frame"] <= ef + 30]

    return {
        "event": event,
        "event_frame_record": {
            "frame": event_rec["frame"] if event_rec else None,
            "time": event_rec["time"] if event_rec else None,
            "ball": event_rec.get("ball") if event_rec else None,
            "missed_frames": event_rec.get("missed_frames") if event_rec else None,
            "routed": event_rec.get("routed") if event_rec else None,
            "impact": event_rec.get("impact") if event_rec else None,
        },
        "last_15_ball_frames": format_ball_chain(last15),
        "vx_reversals_in_window": vx_sign_reversals(
            [r for r in trace if et - 3 <= r["time"] <= et + 1]
        ),
        "max_missed_around_event": max((r.get("missed_frames", 0) for r in around), default=0),
        "routed_around_event": Counter(r.get("routed") for r in around),
        "impacts_around_event": [
            {
                "frame": r["frame"],
                "time": r["time"],
                "routed": r.get("routed"),
                "impact": r.get("impact"),
                "bbox_proxy": bbox_proxy_dist(r["impact"]["x"], r["impact"]["y"]),
            }
            for r in around if r.get("impact")
        ],
    }


def routed_global_stats(trace: list[dict]) -> dict[str, Any]:
    c = Counter(r.get("routed") for r in trace)
    return {
        "total_trace_frames": len(trace),
        "routed_net_contact": c.get("net_contact", 0),
        "routed_process_detection": c.get("process_detection", 0),
        "routed_null": c.get(None, 0),
        "routed_other": {k: v for k, v in c.items() if k not in (None, "net_contact", "process_detection")},
    }


def live_impacts_in_window(live: list[dict], t0: float, t1: float) -> list[dict]:
    return [
        {
            "frame": i["frame"],
            "time": round(i["time"], 3),
            "table_side": i.get("table_side"),
            "x": round(i["x"], 1),
            "y": round(i["y"], 1),
            "y_range": round(i.get("y_range", 0), 1),
            "curvature": round(i.get("curvature", 0), 2),
            "bbox_proxy": bbox_proxy_dist(i["x"], i["y"]),
        }
        for i in live
        if t0 <= i["time"] <= t1
    ]


def main() -> None:
    trace = json.load(open(TRACE_PATH, encoding="utf-8"))
    live = json.load(open(LIVE_PATH, encoding="utf-8"))
    smart = json.load(open(SMART_PATH, encoding="utf-8"))

    report: dict[str, Any] = {
        "net_bbox_proxy": NET_BBOX,
        "net_impact_margin": NET_IMPACT_MARGIN,
        "net_zone_margin": NET_ZONE_MARGIN,
        "routed_global": routed_global_stats(trace),
        "net_events_analysis": [analyze_event(trace, ev) for ev in NET_EVENTS],
        "miss_window_107_5_111": summarize_window(trace, *MISS_WINDOW),
        "rally_window_101_110": {
            "live_impacts": live_impacts_in_window(live, *RALLY_WINDOW),
            "point_history": [
                p for p in smart.get("point_history", [])
                if RALLY_WINDOW[0] - 2 <= p.get("time", 0) <= RALLY_WINDOW[1] + 2
            ],
            "net_events": [
                e for e in smart.get("net_events", [])
                if RALLY_WINDOW[0] - 2 <= e.get("time", 0) <= RALLY_WINDOW[1] + 2
            ],
            "double_bounce_events": [
                e for e in smart.get("double_bounce_events", [])
                if RALLY_WINDOW[0] <= e.get("time", 0) <= RALLY_WINDOW[1] + 5
            ],
            "trace_summary": summarize_window(trace, *RALLY_WINDOW),
        },
        "hypothesis_check_109": {},
    }

    miss = report["miss_window_107_5_111"]
    report["hypothesis_check_109"] = {
        "ball_visible_continuously": miss["ball_continuous"],
        "max_missed_frames": miss["max_missed_frames"],
        "net_contact_routed_count": miss["routed_counts"].get("net_contact", 0),
        "detect_impact_with_impact_meta": miss["impact_count"],
        "vx_reversal_near_net": len(miss["vx_reversals_near_net"]),
        "closest_ball_to_net_bbox": miss["closest_to_net_bbox"][:3],
        "conclusion": (
            "Гипотеза ПОДТВЕРЖДЕНА частично"
            if miss["ball_continuous"]
            and miss["routed_counts"].get("net_contact", 0) == 0
            and miss["impact_count"] == 0
            else "Гипотеза ОПРОВЕРГНУТА или частично"
        ),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # stdout summary
    g = report["routed_global"]
    print("=== ROUTED GLOBAL ===")
    print(f"net_contact={g['routed_net_contact']} process_detection={g['routed_process_detection']} null={g['routed_null']}")

    print("\n=== NET EVENTS (last frame at event) ===")
    for ev_a in report["net_events_analysis"]:
        ev = ev_a["event"]
        rec = ev_a["event_frame_record"]
        print(f"\n--- t={ev['time']}s f={ev['frame']} P{ev['player']} ---")
        print(f"  event_frame: ball={rec.get('ball')} missed={rec.get('missed_frames')} routed={rec.get('routed')}")
        print(f"  max_missed_around={ev_a['max_missed_around_event']}")
        impacts = ev_a["impacts_around_event"]
        if impacts:
            for imp in impacts:
                bp = imp["bbox_proxy"]
                print(f"  impact f={imp['frame']} routed={imp['routed']} "
                      f"side={imp['impact'].get('table_side')} "
                      f"xy=({imp['impact']['x']:.0f},{imp['impact']['y']:.0f}) "
                      f"inside_impact={bp['inside_impact_margin']}")
        else:
            print("  NO impacts in ±30 frames")
        rev = ev_a["vx_reversals_in_window"]
        near = [r for r in rev if r["near_net_x600"]]
        print(f"  vx reversals: total={len(rev)} near_net={len(near)}")
        if ev_a["last_15_ball_frames"]:
            last = ev_a["last_15_ball_frames"][-1]
            print(f"  last ball: f={last['frame']} x={last['x']} missed={last['missed_frames']}")

    print("\n=== MISS WINDOW 107.5-111.0 ===")
    m = report["miss_window_107_5_111"]
    print(f"ball_frames={m['ball_visible_frames']} max_missed={m['max_missed_frames']} continuous={m['ball_continuous']}")
    print(f"routed={m['routed_counts']} impacts={m['impact_count']} net_contact={len(m['net_contact_frames'])}")
    print(f"reject_reasons={m['reject_reasons']}")
    print(f"vx_reversals_near_net={m['vx_reversals_near_net']}")
    print(f"min_dx_to_bbox={m['min_dx_to_net_bbox']}")
    for c in m["closest_to_net_bbox"][:3]:
        print(f"  closest f={c['frame']} t={c['time']} x={c['x']} dx={c['dx_edge']} inside_zone={c['inside_zone_margin']}")

    print("\n=== RALLY 101-110 live_impacts ===")
    for li in report["rally_window_101_110"]["live_impacts"]:
        bp = li["bbox_proxy"]
        print(f"  f={li['frame']} t={li['time']} side={li['table_side']} x={li['x']} "
              f"inside_impact={bp['inside_impact_margin']} y_range={li['y_range']}")

    print(f"\nReport: {OUT_JSON}")


if __name__ == "__main__":
    main()
