#!/usr/bin/env python3
"""Upload batch zip в Roboflow через MCP. Дата: 2026-05-26."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from build_upload_zip import SEG_CLASSES, SEG_V2_CLASSES, build_yolo_zip
from roboflow_client import (
    find_project_root,
    load_pipeline_config,
    poll_upload_status,
    prepare_upload_zip,
    projects_health,
    put_zip,
    resolve_batch_id,
    resolve_run_dir,
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


def resolve_project_id(cfg: dict, override: str | None = None) -> tuple[str, dict]:
    if override:
        return override, {"source": "cli", "note": None}

    label_mode = cfg.get("label_mode", "det")
    if label_mode == "hybrid_seg" and cfg.get("seg_project_id"):
        return cfg["seg_project_id"], {
            "source": "pipeline.seg_project_id",
            "legacy_project_id": cfg.get("project_id"),
            "note": (
                "Если проект tt-seg-img0291 ещё не создан в Roboflow UI — "
                "создайте instance-segmentation project с классами "
                "TableLeft, TableRight, Net, Player1, Player2, затем повторите upload."
            ),
        }
    return cfg["project_id"], {"source": "pipeline.project_id", "note": None}


def upload_batch(
    run_dir: Path,
    batch_name: str,
    project_id: str,
    rebuild_zip: bool = True,
    label_format: str | None = None,
    split: str = "train",
) -> dict:
    run_dir = Path(run_dir)
    zip_path = run_dir / "upload.zip"
    root = find_project_root()
    cfg = load_pipeline_config(root)
    fmt = label_format or ("seg" if cfg.get("label_mode") == "hybrid_seg" else "det")
    classes = cfg.get("classes_seg") if fmt == "seg" else cfg.get("classes")
    if fmt == "seg_v2":
        classes = SEG_V2_CLASSES

    if rebuild_zip or not zip_path.is_file():
        build_yolo_zip(run_dir / "images", run_dir / "labels", zip_path, classes, fmt)

    project_note: str | None = None
    try:
        health = projects_health(project_id, regenerate=False)
        if not health:
            project_note = (
                f"projects_health вернул пустой ответ для {project_id}; "
                "убедитесь что seg-проект существует в workspace."
            )
    except RuntimeError as exc:
        project_note = (
            f"Не удалось проверить проект {project_id}: {exc}. "
            "Создайте instance-segmentation project в Roboflow UI при необходимости."
        )

    try:
        prep = prepare_upload_zip(project_id=project_id, batch_name=batch_name, split=split)
    except RuntimeError as exc:
        err = str(exc)
        missing_project = "404" in err or "does not exist" in err
        note = project_note or ""
        if missing_project:
            create_note = (
                f"Roboflow project '{project_id}' не найден. "
                "Создайте instance-segmentation project в workspace yolo-table-tennis "
                f"с slug '{project_id}' и классами: "
                f"{', '.join(cfg.get('classes_seg', SEG_CLASSES))}. "
                f"Затем повторите: python3 upload_batch.py --run-dir {run_dir} "
                f"--batch-name {batch_name} --no-rebuild --format {fmt}"
            )
            note = f"{note} {create_note}".strip()
        return {
            "batch_name": batch_name,
            "project_id": project_id,
            "label_format": fmt,
            "split": split,
            "status": "failed",
            "error": err,
            "zip_path": str(zip_path),
            "project_check_note": note or None,
            "task_id": None,
            "batch_id": None,
        }

    put_zip(prep["signed_url"], zip_path)
    status = poll_upload_status(
        prep["task_id"],
        transport=prep.get("transport"),
    )
    batch_id = resolve_batch_id(project_id, batch_name, status)

    result = {
        "batch_name": batch_name,
        "project_id": project_id,
        "label_format": fmt,
        "split": split,
        "status": "success",
        "task_id": prep["task_id"],
        "batch_id": batch_id,
        "upload_transport": prep.get("transport", "unknown"),
        "upload_status": status,
        "zip_path": str(zip_path),
        "project_check_note": project_note,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build zip and upload to Roboflow")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--batch-name", type=str, required=True)
    parser.add_argument("--project-id", type=str, default=None)
    parser.add_argument("--format", choices=["seg", "seg_v2", "det"], default=None)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--no-rebuild", action="store_true")
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_pipeline_config(root)
    project_id, project_meta = resolve_project_id(cfg, args.project_id)

    run_dir = resolve_run_dir(args.run_dir, root)

    result = upload_batch(
        run_dir,
        args.batch_name,
        project_id,
        rebuild_zip=not args.no_rebuild,
        label_format=args.format,
        split=args.split,
    )
    result["project_resolution"] = project_meta
    if project_meta.get("note"):
        print(f"NOTE: {project_meta['note']}")
    if result.get("project_check_note"):
        print(f"NOTE: {result['project_check_note']}")

    state = load_state(run_dir)
    state["upload"] = result
    save_state(run_dir, state)

    if result.get("status") == "failed":
        print(f"Upload FAILED for project_id={result['project_id']}")
        print(f"error={result.get('error')}")
        print(f"zip ready at: {result.get('zip_path')}")
        print(f"state.json updated: {run_dir / 'state.json'}")
        raise SystemExit(2)

    print(f"Uploaded batch_name={result['batch_name']}")
    print(f"split={result.get('split', 'train')}")
    print(f"project_id={result['project_id']}")
    print(f"task_id={result['task_id']}")
    print(f"batch_id={result.get('batch_id')}")
    print(f"state.json updated: {run_dir / 'state.json'}")


if __name__ == "__main__":
    main()
