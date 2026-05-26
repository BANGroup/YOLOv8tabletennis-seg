#!/usr/bin/env python3
"""Upload batch zip в Roboflow через MCP. Дата: 2026-05-26."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from build_upload_zip import build_yolo_zip
from roboflow_client import (
    find_project_root,
    poll_upload_status,
    prepare_upload_zip,
    put_zip,
    resolve_batch_id,
)


def load_state(run_dir: Path) -> dict:
    state_path = run_dir / "state.json"
    if state_path.is_file():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {}


def save_state(run_dir: Path, state: dict) -> None:
    state_path = run_dir / "state.json"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def upload_batch(
    run_dir: Path,
    batch_name: str,
    project_id: str,
    rebuild_zip: bool = True,
) -> dict:
    run_dir = Path(run_dir)
    zip_path = run_dir / "upload.zip"
    if rebuild_zip or not zip_path.is_file():
        build_yolo_zip(run_dir / "images", run_dir / "labels", zip_path)

    prep = prepare_upload_zip(project_id=project_id, batch_name=batch_name)
    put_zip(prep["signed_url"], zip_path)
    status = poll_upload_status(prep["task_id"])
    batch_id = resolve_batch_id(project_id, batch_name, status)

    result = {
        "batch_name": batch_name,
        "project_id": project_id,
        "task_id": prep["task_id"],
        "batch_id": batch_id,
        "upload_status": status,
        "zip_path": str(zip_path),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build zip and upload to Roboflow")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--batch-name", type=str, required=True)
    parser.add_argument("--project-id", type=str, default=None)
    parser.add_argument("--no-rebuild", action="store_true")
    args = parser.parse_args()

    root = find_project_root()
    cfg_path = root / "results" / "roboflow_pipeline" / "config" / "pipeline.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    project_id = args.project_id or cfg["project_id"]

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / "results" / "roboflow_pipeline" / run_dir

    result = upload_batch(run_dir, args.batch_name, project_id, rebuild_zip=not args.no_rebuild)
    state = load_state(run_dir)
    state["upload"] = result
    save_state(run_dir, state)

    print(f"Uploaded batch_name={result['batch_name']}")
    print(f"task_id={result['task_id']}")
    print(f"batch_id={result.get('batch_id')}")
    print(f"state.json updated: {run_dir / 'state.json'}")


if __name__ == "__main__":
    main()
