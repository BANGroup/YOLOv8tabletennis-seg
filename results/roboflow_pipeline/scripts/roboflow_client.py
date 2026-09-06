#!/usr/bin/env python3
"""Roboflow MCP HTTP client (urllib, SSE JSON). Дата: 2026-05-26."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

MCP_URL = "https://mcp.roboflow.com/mcp"
REST_API_BASE = "https://api.roboflow.com"
DEFAULT_ENV_PATHS = (
    ".env",
    "../../.env",
    "../../../.env",
)


def find_project_root(start: Path | None = None) -> Path:
    """Поднимается вверх, пока не найдёт dataset/ и runs/."""
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for _ in range(8):
        if (cur / "dataset").is_dir() and (cur / "runs").is_dir():
            return cur
        cur = cur.parent
    raise FileNotFoundError("Не найден корень проекта (dataset/ + runs/)")


def load_roboflow_api_key(env_path: Path | None = None) -> str:
    if env_path is None:
        root = find_project_root()
        candidates = [root / ".env"] + [Path(__file__).resolve().parent / p for p in DEFAULT_ENV_PATHS]
    else:
        candidates = [env_path]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("ROBOFLOW_API_KEY=") and not line.endswith("PLACEHOLDER"):
                key = line.split("=", 1)[1].strip()
                if key:
                    return key
    raise EnvironmentError("ROBOFLOW_API_KEY не найден в .env")


def _parse_sse_payload(raw: str) -> dict[str, Any]:
    for line in raw.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    raise RuntimeError(f"MCP: нет data: в ответе: {raw[:500]}")


def _extract_tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("error"):
        raise RuntimeError(json.dumps(payload["error"], ensure_ascii=False))
    result = payload.get("result") or {}
    if result.get("isError"):
        text = ""
        content = result.get("content") or []
        if content:
            text = content[0].get("text", "")
        raise RuntimeError(f"MCP tool error: {text}")
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and structured:
        if "result" in structured and len(structured) == 1:
            inner = structured["result"]
            return inner if isinstance(inner, dict) else {"result": inner}
        return structured
    content = result.get("content") or []
    if content and content[0].get("text"):
        text = content[0]["text"]
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except json.JSONDecodeError:
            return {"text": text}
    return {}


def mcp_call(tool_name: str, arguments: dict[str, Any] | None = None, api_key: str | None = None) -> dict[str, Any]:
    key = api_key or load_roboflow_api_key()
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments or {}},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        MCP_URL,
        data=body,
        headers={
            "x-api-key": key,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MCP HTTP {exc.code}: {detail}") from exc
    payload = _parse_sse_payload(raw)
    return _extract_tool_result(payload)


def resolve_run_dir(run_dir: str | Path, root: Path | None = None) -> Path:
    root = root or find_project_root()
    path = Path(run_dir)
    if path.is_absolute():
        return path
    pipeline_root = root / "results" / "roboflow_pipeline"
    candidate = (pipeline_root / path).resolve()
    if candidate.exists() or path.parts[0] == "runs":
        return candidate
    return (pipeline_root / "runs" / path).resolve()


def load_pipeline_config(root: Path | None = None) -> dict[str, Any]:
    root = root or find_project_root()
    cfg_path = root / "results" / "roboflow_pipeline" / "config" / "pipeline.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _project_parts(project_id: str, workspace: str | None = None) -> tuple[str, str]:
    if "/" in project_id:
        ws, proj = project_id.split("/", 1)
        return ws, proj
    cfg = load_pipeline_config()
    ws = workspace or cfg.get("workspace", "yolo-table-tennis")
    return ws, project_id


def prepare_upload_zip_rest(
    project_id: str,
    batch_name: str | None = None,
    split: str = "train",
    workspace: str | None = None,
) -> dict[str, Any]:
    """Roboflow REST async zip upload (SSoT при сбое MCP images_prepare_upload_zip)."""
    key = load_roboflow_api_key()
    ws, proj = _project_parts(project_id, workspace)
    query: dict[str, str] = {"api_key": key, "split": split}
    if batch_name:
        query["batch"] = batch_name
    url = f"{REST_API_BASE}/{ws}/{proj}/upload/zip?{urllib.parse.urlencode(query)}"
    payload: dict[str, str] = {}
    if batch_name:
        payload = {"batch_name": batch_name, "batch": batch_name}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"REST prepare zip HTTP {exc.code}: {detail}") from exc
    if result.get("error"):
        raise RuntimeError(f"REST prepare zip error: {result['error']}")
    task_id = result.get("taskId") or result.get("task_id")
    signed_url = result.get("signedUrl") or result.get("signed_url")
    if not task_id or not signed_url:
        raise RuntimeError(f"REST prepare zip: неожиданный ответ: {result}")
    return {
        "task_id": task_id,
        "signed_url": signed_url,
        "transport": "rest",
        "raw": result,
    }


def prepare_upload_zip(
    project_id: str,
    batch_name: str | None = None,
    split: str = "train",
    tag_names: list[str] | None = None,
    workspace: str | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {"project_id": project_id, "split": split}
    if batch_name:
        args["batch_name"] = batch_name
    if tag_names:
        args["tag_names"] = tag_names
    try:
        result = mcp_call("images_prepare_upload_zip", args)
        task_id = result.get("taskId") or result.get("task_id")
        signed_url = result.get("signedUrl") or result.get("signed_url")
        if task_id and signed_url:
            return {
                "task_id": task_id,
                "signed_url": signed_url,
                "transport": "mcp",
                "raw": result,
            }
    except RuntimeError:
        pass
    return prepare_upload_zip_rest(project_id, batch_name, split, workspace)


def put_zip(signed_url: str, zip_path: Path) -> None:
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    data = zip_path.read_bytes()
    req = urllib.request.Request(
        signed_url,
        data=data,
        headers={"Content-Type": "application/zip"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            if resp.status not in (200, 201):
                raise RuntimeError(f"PUT zip status {resp.status}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"PUT zip HTTP {exc.code}: {detail}") from exc


def poll_upload_status_rest(
    task_id: str,
    workspace: str | None = None,
    timeout_sec: float = 600.0,
    interval_sec: float = 5.0,
) -> dict[str, Any]:
    key = load_roboflow_api_key()
    ws = workspace or load_pipeline_config().get("workspace", "yolo-table-tennis")
    url = f"{REST_API_BASE}/{ws}/upload/zip/{task_id}?api_key={key}"
    deadline = time.time() + timeout_sec
    last: dict[str, Any] = {}
    while time.time() < deadline:
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                last = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"REST upload status HTTP {exc.code}: {detail}") from exc
        status = str(last.get("status", last.get("state", ""))).lower()
        if status in {"complete", "completed", "success", "done", "finished"}:
            last["transport"] = "rest"
            return last
        if status in {"failed", "error"}:
            raise RuntimeError(f"upload failed: {json.dumps(last, ensure_ascii=False)}")
        time.sleep(interval_sec)
    raise TimeoutError(f"upload status timeout task_id={task_id}, last={last}")


def poll_upload_status(
    task_id: str,
    timeout_sec: float = 600.0,
    interval_sec: float = 5.0,
    transport: str | None = None,
    workspace: str | None = None,
) -> dict[str, Any]:
    if transport == "rest":
        return poll_upload_status_rest(task_id, workspace, timeout_sec, interval_sec)
    deadline = time.time() + timeout_sec
    last: dict[str, Any] = {}
    while time.time() < deadline:
        try:
            last = mcp_call("images_upload_zip_status", {"task_id": task_id})
            last["transport"] = "mcp"
        except RuntimeError:
            return poll_upload_status_rest(task_id, workspace, deadline - time.time(), interval_sec)
        status = str(last.get("status", last.get("state", ""))).lower()
        if status in {"complete", "completed", "success", "done", "finished"}:
            return last
        if status in {"failed", "error"}:
            raise RuntimeError(f"upload failed: {json.dumps(last, ensure_ascii=False)}")
        if last.get("complete") is True or last.get("done") is True:
            return last
        time.sleep(interval_sec)
    raise TimeoutError(f"upload status timeout task_id={task_id}, last={last}")


def autolabel_start(
    project_id: str,
    batch_id: str,
    model: str,
    model_type: str,
    ontology: dict[str, str] | None = None,
    num_images_to_label: int | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "project_id": project_id,
        "batch_id": batch_id,
        "model": model,
        "model_type": model_type,
    }
    if ontology:
        args["ontology"] = ontology
    if num_images_to_label is not None:
        args["num_images_to_label"] = num_images_to_label
    return mcp_call("autolabel_start", args)


def autolabel_job_get(autolabel_job_id: str) -> dict[str, Any]:
    return mcp_call("autolabel_job_get", {"autolabel_job_id": autolabel_job_id})


def annotation_batches_list(project_id: str, workspace: str | None = None) -> dict[str, Any]:
    ws, proj = _project_parts(project_id, workspace)
    key = load_roboflow_api_key()
    url = f"{REST_API_BASE}/{ws}/{proj}/batches?api_key={key}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            rest = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        rest = {"batches": []}
    mcp_batches: dict[str, Any] = {"result": []}
    try:
        mcp_batches = mcp_call("annotation_batches_list", {"project_id": project_id})
    except RuntimeError:
        pass
    return {"rest": rest, "mcp": mcp_batches, "batches": rest.get("batches") or []}


def projects_health(project_id: str, regenerate: bool = False) -> dict[str, Any]:
    return mcp_call("projects_health", {"project_id": project_id, "regenerate": regenerate})


def versions_export(project_id: str, version_number: int, export_format: str = "yolov8") -> dict[str, Any]:
    return mcp_call(
        "versions_export",
        {
            "project_id": project_id,
            "version_number": version_number,
            "export_format": export_format,
        },
    )


def resolve_batch_id(project_id: str, batch_name: str, upload_status: dict[str, Any]) -> str | None:
    for key in ("batchId", "batch_id", "annotationBatchId", "annotation_batch_id"):
        val = upload_status.get(key)
        if val:
            return str(val)
    result = upload_status.get("result")
    if isinstance(result, dict):
        for key in ("batchId", "batch_id", "batch"):
            val = result.get(key)
            if val and isinstance(val, str):
                return val
    for attempt in range(6):
        batches_payload = annotation_batches_list(project_id)
        items = batches_payload.get("batches") or []
        mcp_items = batches_payload.get("mcp", {}).get("result") or []
        if isinstance(mcp_items, list):
            items = items + mcp_items
        for batch in items:
            if not isinstance(batch, dict):
                continue
            name = batch.get("name") or batch.get("batchName") or batch.get("batch_name")
            if name == batch_name:
                return str(batch.get("id") or batch.get("batchId") or batch.get("batch_id"))
        if items:
            latest = items[-1]
            if isinstance(latest, dict):
                bid = latest.get("id") or latest.get("batchId") or latest.get("batch_id")
                if bid:
                    return str(bid)
        if attempt < 5:
            time.sleep(5)
    return None
