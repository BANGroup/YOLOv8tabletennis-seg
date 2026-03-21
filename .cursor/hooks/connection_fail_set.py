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

    tool_name = payload.get("tool_name")
    if tool_name in {"Shell", "MCP"}:
        flag_path = Path(__file__).resolve().parent / "connection_fail.flag"
        flag_path.write_text("1\n", encoding="utf-8")

    sys.stdout.write("{}")


if __name__ == "__main__":
    main()
