#!/usr/bin/env python3
"""afterFileEdit hook: логирует правки в зонах кодеров для audit trail.

КАСТОМИЗАЦИЯ: измените WATCHED_PREFIXES под зоны вашего проекта.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


LOG_PATH = Path(__file__).resolve().parent / "code_edit_audit.log"

# КАСТОМИЗАЦИЯ: укажите префиксы зон кодеров вашего проекта
WATCHED_PREFIXES = ("src/", "lib/", "app/", "code/", "tools/")

MAX_SNIPPET = 120  # максимум символов old/new в логе
REPO_ROOT = Path(__file__).resolve().parents[2]


def _normalize_path(raw_path: str) -> str:
    if not raw_path:
        return ""

    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        try:
            return path_obj.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path_obj.as_posix()

    return raw_path.replace("\\", "/")


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {}

    file_path = _normalize_path(payload.get("file_path", ""))
    if not any(file_path.startswith(p) for p in WATCHED_PREFIXES):
        sys.stdout.write("{}")
        return

    edits = payload.get("edits", [])
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    conv_id = payload.get("conversation_id", "?")[:8]
    gen_id = payload.get("generation_id", "?")[:8]

    lines = [f"[{ts}] conv={conv_id} gen={gen_id} file={file_path}"]
    for i, edit in enumerate(edits):
        old = (edit.get("old_string") or "")[:MAX_SNIPPET].replace("\n", "\\n")
        new = (edit.get("new_string") or "")[:MAX_SNIPPET].replace("\n", "\\n")
        lines.append(f"  edit[{i}] old='{old}' new='{new}'")

    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n\n")
    except OSError:
        pass

    sys.stdout.write("{}")


if __name__ == "__main__":
    main()
