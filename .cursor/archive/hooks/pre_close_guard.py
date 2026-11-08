#!/usr/bin/env python3
"""preToolUse hook: блокирует --close-workflow без governance/docs handoff.

Проверка применяется только к Shell-вызовам `code/utils/agent_kg.py --close-workflow`.
Для закрытия workflow должны существовать handoff от:
- governance-compliance
- docs-curator

Дополнительно: в этих handoff должны быть непустые `trace_id` и `plan_step_id`.
"""

import json
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REQUIRED_AGENTS = ("governance-compliance", "docs-curator")
KG_RELATIVE_PATH = Path("config/agent_kg.json")


def _extract_shell_command(payload: Dict[str, object]) -> str:
    tool_input = payload.get("tool_input")
    if isinstance(tool_input, dict):
        command = tool_input.get("command")
        if isinstance(command, str):
            return command
    command = payload.get("command")
    if isinstance(command, str):
        return command
    return ""


def _is_close_workflow_command(command: str) -> bool:
    if "agent_kg.py" not in command:
        return False
    return "--close-workflow" in command


def _extract_workflow_id(command: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    for idx, token in enumerate(tokens):
        if token == "--workflow-id" and idx + 1 < len(tokens):
            return tokens[idx + 1].strip()
        if token.startswith("--workflow-id="):
            return token.split("=", 1)[1].strip()

    fallback = re.search(r"--workflow-id(?:=|\s+)([A-Za-z0-9_:-]+)", command)
    if fallback:
        return fallback.group(1).strip()
    return ""


def _load_agent_kg(repo_root: Path) -> Dict[str, object]:
    kg_path = repo_root / KG_RELATIVE_PATH
    with kg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_required_handoffs(
    kg_data: Dict[str, object], workflow_id: str
) -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    latest: Dict[str, Dict[str, object]] = {}
    handoffs = kg_data.get("handoffs", [])
    if not isinstance(handoffs, list):
        return latest, list(REQUIRED_AGENTS)

    for item in handoffs:
        if not isinstance(item, dict):
            continue
        if item.get("workflow_id") != workflow_id:
            continue
        agent = item.get("agent")
        if agent not in REQUIRED_AGENTS:
            continue
        current = latest.get(agent)
        if current is None or (item.get("created_at") or "") > (current.get("created_at") or ""):
            latest[agent] = item

    missing = [agent for agent in REQUIRED_AGENTS if agent not in latest]
    return latest, missing


def _invalid_trace_fields(handoffs_by_agent: Dict[str, Dict[str, object]]) -> Dict[str, List[str]]:
    invalid: Dict[str, List[str]] = {}
    for agent, handoff in handoffs_by_agent.items():
        missing_fields: List[str] = []
        if not str(handoff.get("trace_id") or "").strip():
            missing_fields.append("trace_id")
        if not str(handoff.get("plan_step_id") or "").strip():
            missing_fields.append("plan_step_id")
        if missing_fields:
            invalid[agent] = missing_fields
    return invalid


def _deny(reason: str) -> None:
    sys.stdout.write(json.dumps({"decision": "deny", "reason": reason}))


def _allow() -> None:
    sys.stdout.write(json.dumps({"decision": "allow"}))


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {}

    if payload.get("tool_name") != "Shell":
        _allow()
        return

    command = _extract_shell_command(payload)
    if not _is_close_workflow_command(command):
        _allow()
        return

    workflow_id = _extract_workflow_id(command)
    if not workflow_id:
        _deny(
            "Закрытие workflow заблокировано: в команде отсутствует --workflow-id. "
            "Укажите workflow-id и повторите."
        )
        return

    repo_root = Path(__file__).resolve().parents[2]
    kg_path = repo_root / KG_RELATIVE_PATH
    if not kg_path.exists():
        _deny(
            "Закрытие workflow заблокировано: не найден config/agent_kg.json. "
            "Невозможно проверить governance/docs handoff."
        )
        return

    try:
        kg_data = _load_agent_kg(repo_root)
    except (OSError, json.JSONDecodeError):
        _deny("Закрытие workflow заблокировано: не удалось прочитать config/agent_kg.json.")
        return

    handoffs_by_agent, missing = _latest_required_handoffs(kg_data, workflow_id)
    if missing:
        missing_text = ", ".join(missing)
        _deny(
            "Закрытие workflow заблокировано: отсутствуют обязательные handoff "
            f"для {workflow_id}: {missing_text}. "
            "Сначала получите и запишите handoff, затем повторите --close-workflow."
        )
        return

    invalid = _invalid_trace_fields(handoffs_by_agent)
    if invalid:
        details = "; ".join(
            f"{agent}: {', '.join(fields)}" for agent, fields in invalid.items()
        )
        _deny(
            "Закрытие workflow заблокировано: в обязательных handoff не заполнены "
            f"поля трассировки для {workflow_id}: {details}. "
            "Сначала переоформите handoff с trace_id и plan_step_id."
        )
        return

    _allow()


if __name__ == "__main__":
    main()
