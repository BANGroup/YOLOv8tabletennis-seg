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

    prompt = (payload.get("prompt") or "").lower()
    if "продолж" in prompt or "разреш" in prompt or "готов" in prompt:
        flag_path = Path(__file__).resolve().parent / "connection_fail.flag"
        try:
            if flag_path.exists():
                flag_path.unlink()
        except Exception:
            pass

    sys.stdout.write(json.dumps({"continue": True}))


if __name__ == "__main__":
    main()
