#!/usr/bin/env python3
"""Export dataset version из Roboflow (stub + MCP call). Дата: 2026-05-26."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from roboflow_client import find_project_root, versions_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger Roboflow version export (yolov8)")
    parser.add_argument("--run-dir", type=str, default=None, help="optional run dir for report")
    parser.add_argument("--project-id", type=str, default=None)
    parser.add_argument("--version", type=int, default=None, help="dataset version number")
    parser.add_argument("--format", type=str, default="yolov8")
    parser.add_argument("--dry-run", action="store_true", help="Только записать план без API вызова")
    args = parser.parse_args()

    root = find_project_root()
    cfg_path = root / "results" / "roboflow_pipeline" / "config" / "pipeline.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    roboflow_cfg_path = root / "scripts" / "config" / "roboflow.json"
    roboflow_cfg = json.loads(roboflow_cfg_path.read_text(encoding="utf-8"))

    project_id = args.project_id or cfg["project_id"]
    version = args.version or roboflow_cfg.get("dataset_version", 6)

    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id,
        "version_number": version,
        "export_format": args.format,
        "status": "planned",
    }

    if args.dry_run:
        plan["note"] = "dry-run: MCP versions_export не вызывался"
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    try:
        result = versions_export(project_id, version, args.format)
        plan["status"] = "requested"
        plan["export_response"] = result
    except Exception as exc:
        plan["status"] = "error"
        plan["error"] = str(exc)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = root / "results" / "roboflow_pipeline" / run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "export_version_report.json"
    else:
        out_path = root / "results" / "roboflow_pipeline" / "export_version_report.json"

    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Export report: {out_path}")
    print(f"status={plan['status']}")


if __name__ == "__main__":
    main()
