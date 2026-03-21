#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {}

    flag_path = Path(__file__).resolve().parent / "connection_fail.flag"
    if not flag_path.exists():
        sys.stdout.write(json.dumps({"decision": "allow"}))
        return

    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input") or {}

    if tool_name == "Delete":
        target = tool_input.get("path")
        if target:
            try:
                target_path = Path(target).resolve()
                if target_path == flag_path.resolve():
                    sys.stdout.write(json.dumps({"decision": "allow"}))
                    return
            except Exception:
                pass

    sys.stdout.write(
        json.dumps(
            {
                "decision": "deny",
                "reason": "Остановлено: проблема подключения к сервисам. Сообщите оркестратору. Для продолжения напишите «продолжай» или «готово».",
            }
        )
    )


if __name__ == "__main__":
    main()
