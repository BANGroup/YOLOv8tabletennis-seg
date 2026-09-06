#!/usr/bin/env python3
"""Autolabel matrix M1 (Roboflow) -> M2 (SAM3). Дата: 2026-05-26."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from roboflow_client import autolabel_job_get, autolabel_start, find_project_root, load_pipeline_config, resolve_run_dir


def load_state(run_dir: Path) -> dict:
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"state.json не найден: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(run_dir: Path, state: dict) -> None:
    state_path = run_dir / "state.json"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _job_id(response: dict) -> str | None:
    for key in ("autolabelJobId", "autolabel_job_id", "jobId", "job_id", "id"):
        val = response.get(key)
        if val:
            return str(val)
    nested = response.get("job") or response.get("autolabelJob")
    if isinstance(nested, dict):
        return _job_id(nested)
    return None


def _wait_job(job_id: str, timeout_sec: float = 900.0, interval_sec: float = 10.0) -> dict:
    deadline = time.time() + timeout_sec
    last: dict = {}
    while time.time() < deadline:
        last = autolabel_job_get(job_id)
        status = str(last.get("status", last.get("state", ""))).lower()
        if status in {"complete", "completed", "success", "done", "finished"}:
            return last
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"autolabel job failed: {json.dumps(last, ensure_ascii=False)}")
        progress = last.get("progress") or last.get("percentComplete")
        if progress is not None:
            print(f"  job {job_id}: progress={progress}")
        time.sleep(interval_sec)
    raise TimeoutError(f"autolabel timeout job_id={job_id}, last={last}")


def run_matrix(run_dir: Path, cfg: dict, wait: bool = True) -> dict:
    state = load_state(run_dir)
    batch_id = (state.get("upload") or {}).get("batch_id")
    if not batch_id:
        raise RuntimeError("batch_id отсутствует в state.json — сначала upload_batch.py")

    project_id = cfg.get("seg_project_id") if cfg.get("label_mode") == "hybrid_seg" else cfg["project_id"]
    if not project_id:
        project_id = cfg["project_id"]
    matrix_cfg = cfg["autolabel_models"]
    results: dict = {"batch_id": batch_id, "jobs": {}}

    for key in ("M1", "M2"):
        model_cfg = matrix_cfg[key]
        print(f"Starting autolabel {key}: {model_cfg['model']} ({model_cfg['model_type']})")
        resp = autolabel_start(
            project_id=project_id,
            batch_id=batch_id,
            model=model_cfg["model"],
            model_type=model_cfg["model_type"],
            ontology=model_cfg.get("ontology"),
        )
        job_id = _job_id(resp)
        entry = {"start_response": resp, "job_id": job_id}
        if job_id and wait:
            entry["final_status"] = _wait_job(job_id)
        results["jobs"][key] = entry
        print(f"  {key} job_id={job_id}")

    state["autolabel_matrix"] = results
    save_state(run_dir, state)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autolabel matrix M1 then M2")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--no-wait", action="store_true", help="Не ждать завершения jobs")
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_pipeline_config(root)

    run_dir = resolve_run_dir(args.run_dir, root)

    results = run_matrix(run_dir, cfg, wait=not args.no_wait)
    report_path = run_dir / "autolabel_matrix_report.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Matrix report: {report_path}")


if __name__ == "__main__":
    main()
