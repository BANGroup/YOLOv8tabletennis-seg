#!/usr/bin/env python3
"""Сверка прогона serve_detector.py с эталонами tests/expected/*.yaml.

Читает smart_impacts_analysis.json (и при необходимости run_stdout.log)
и печатает [PASS] / [WARN] / [FAIL] / [SKIP] по каждой проверке.
Код возврата: 0, если нет FAIL (PASS, WARN и SKIP допустимы); 1, если есть FAIL
или не найден файл спеки / артефакта.
Выход за [min, max] даёт [WARN] вместо [FAIL], если у узла checks.* стоит
severity: warn, либо для serve_count при serve_order_errors: warning_only.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_DIR = REPO_ROOT / "tests" / "expected"
EVENT_KEYS = ("net_events", "out_events", "double_bounce_events")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Сверить smart_impacts_analysis.json с эталоном "
            "tests/expected/*.yaml"
        )
    )
    parser.add_argument(
        "--spec",
        action="append",
        nargs="+",
        default=[],
        metavar="YAML",
        help="Путь к .yaml в tests/expected/ (можно несколько)",
    )
    parser.add_argument(
        "--actual",
        default=None,
        help=(
            "Путь к smart_impacts_analysis.json. "
            "Если не задан, берётся reference_artifact из yaml"
        ),
    )
    parser.add_argument(
        "--stdout",
        default=None,
        help=(
            "Путь к run_stdout.log. "
            "Если не задан, берётся reference_stdout из yaml"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Проверить все tests/expected/*.yaml по их reference_* полям",
    )
    return parser.parse_args(argv)


def resolve_repo_path(raw):
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_cli_path(raw):
    path = Path(raw)
    if path.is_absolute():
        return path
    from_cwd = Path.cwd() / path
    if from_cwd.exists():
        return from_cwd
    return REPO_ROOT / path


def display_path(path):
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return str(path)


def fmt_num(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def expected_clause(bounds):
    lo = bounds.get("min")
    hi = bounds.get("max")
    parts = []
    if lo is not None and hi is not None:
        parts.append(f"expected {fmt_num(lo)}..{fmt_num(hi)}")
    elif lo is not None:
        parts.append(f"expected >= {fmt_num(lo)}")
    elif hi is not None:
        parts.append(f"expected <= {fmt_num(hi)}")
    if bounds.get("baseline") is not None:
        parts.append(f"baseline {fmt_num(bounds['baseline'])}")
    return ", ".join(parts)


def within_bounds(value, bounds):
    number = float(value)
    lo = bounds.get("min")
    hi = bounds.get("max")
    if lo is not None and number < float(lo):
        return False
    if hi is not None and number > float(hi):
        return False
    return True


def out_of_range_status(bounds, force_warn=False):
    """FAIL по умолчанию; WARN, если узел помечен severity: warn или force_warn."""
    if force_warn:
        return "WARN"
    if isinstance(bounds, dict):
        severity = bounds.get("severity")
        if isinstance(severity, str) and severity.strip().lower() == "warn":
            return "WARN"
    return "FAIL"


class Report:
    def __init__(self):
        self.passed = 0
        self.warned = 0
        self.failed = 0
        self.skipped = 0

    def add(self, status, label, detail=""):
        if status == "PASS":
            self.passed += 1
        elif status == "WARN":
            self.warned += 1
        elif status == "FAIL":
            self.failed += 1
        else:
            self.skipped += 1
            status = "SKIP"
        if detail:
            print(f"[{status}] {label}: {detail}")
        else:
            print(f"[{status}] {label}")

    @property
    def ok(self):
        return self.failed == 0


def collect_spec_paths(args):
    paths = []
    if args.all:
        if not EXPECTED_DIR.is_dir():
            return [], f"каталог не найден: {display_path(EXPECTED_DIR)}"
        paths.extend(sorted(EXPECTED_DIR.glob("*.yaml")))
    for group in args.spec:
        for raw in group:
            paths.append(resolve_cli_path(raw))

    unique = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique, None


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("корень YAML должен быть словарём")
    return data


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("корень JSON должен быть словарём")
    return data


def check_number(report, label, actual, key, bounds, warn_out_of_range=False):
    if not isinstance(bounds, dict):
        return
    if bounds.get("min") is None and bounds.get("max") is None:
        return
    clause = expected_clause(bounds)
    if key not in actual:
        report.add("FAIL", label, f"нет ключа {key!r} ({clause})")
        return
    value = actual[key]
    try:
        ok = within_bounds(value, bounds)
    except (TypeError, ValueError):
        report.add("FAIL", label, f"нечисловое значение {value!r} ({clause})")
        return
    status = "PASS" if ok else out_of_range_status(bounds, warn_out_of_range)
    report.add(status, label, f"{fmt_num(value)} ({clause})")


def check_score_side(report, actual, player, bounds):
    label = f"score.{player}"
    if not isinstance(bounds, dict):
        return
    if bounds.get("min") is None and bounds.get("max") is None:
        return
    clause = expected_clause(bounds)
    score = actual.get("score")
    if not isinstance(score, dict) or player not in score:
        report.add("FAIL", label, f"нет score.{player} ({clause})")
        return
    value = score[player]
    try:
        ok = within_bounds(value, bounds)
    except (TypeError, ValueError):
        report.add("FAIL", label, f"нечисловое значение {value!r} ({clause})")
        return
    status = "PASS" if ok else out_of_range_status(bounds)
    report.add(status, label, f"{fmt_num(value)} ({clause})")


def check_score_balance(report, actual, bounds):
    if not isinstance(bounds, dict):
        return
    threshold = bounds.get("min_lesser_side_share")
    if threshold is None:
        return
    clause = f"expected >= {fmt_num(threshold)}"
    score = actual.get("score")
    if not isinstance(score, dict):
        report.add("FAIL", "score_balance", f"нет score ({clause})")
        return
    try:
        player1 = float(score["player1"])
        player2 = float(score["player2"])
        limit = float(threshold)
    except (KeyError, TypeError, ValueError):
        report.add("FAIL", "score_balance", f"некорректный score ({clause})")
        return
    share = min(player1, player2) / max(1.0, player1 + player2)
    status = "PASS" if share >= limit else out_of_range_status(bounds)
    report.add(status, "score_balance", f"{share:.4f} ({clause})")


def point_time(item):
    if not isinstance(item, dict):
        return None
    raw = item.get("time", item.get("time_sec"))
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def point_winner(item):
    if not isinstance(item, dict) or "winner" not in item:
        return None
    try:
        return int(item["winner"])
    except (TypeError, ValueError):
        return None


def has_legacy_activity(actual):
    impacts = actual.get("impacts_details")
    if isinstance(impacts, list) and impacts:
        return True
    for key in EVENT_KEYS:
        events = actual.get(key)
        if isinstance(events, list) and events:
            return True
    return False


def nearest_point(history, time_sec):
    nearest = None
    for item in history:
        stamp = point_time(item)
        if stamp is None:
            continue
        delta = abs(stamp - time_sec)
        if nearest is None or delta < nearest[0]:
            nearest = (delta, stamp, point_winner(item))
    return nearest


def check_anchors(report, actual, anchors):
    if not anchors:
        return
    if not isinstance(anchors, list):
        report.add("FAIL", "human_verified_anchors", "ожидался список")
        return

    history = actual.get("point_history") if "point_history" in actual else None
    if history is None:
        if has_legacy_activity(actual):
            report.add(
                "SKIP",
                "human_verified_anchors",
                "SKIP (legacy JSON without point_history)",
            )
        else:
            report.add(
                "FAIL",
                "human_verified_anchors",
                "legacy JSON without point_history и нет активности "
                "в impacts_details/events",
            )
        return

    if not isinstance(history, list) or not history:
        report.add(
            "FAIL",
            "human_verified_anchors",
            "point_history пуст, якоря не с чем сверить",
        )
        return

    for index, anchor in enumerate(anchors, start=1):
        label = f"anchor[{index}]"
        if not isinstance(anchor, dict):
            report.add("FAIL", label, "якорь должен быть словарём")
            continue
        try:
            time_sec = float(anchor["time_sec"])
            tolerance = float(anchor["tolerance_sec"])
            winner = int(anchor["winner_player"])
        except (KeyError, TypeError, ValueError):
            report.add(
                "FAIL",
                label,
                "нужны time_sec, tolerance_sec и winner_player",
            )
            continue
        label = f"anchor t={fmt_num(time_sec)}s winner={winner}"
        matched = None
        for item in history:
            stamp = point_time(item)
            item_winner = point_winner(item)
            if stamp is None or item_winner is None:
                continue
            if abs(stamp - time_sec) <= tolerance and item_winner == winner:
                matched = (stamp, item_winner)
                break
        if matched is not None:
            report.add(
                "PASS",
                label,
                f"matched t={fmt_num(matched[0])}s winner={matched[1]} "
                f"(tol {fmt_num(tolerance)}s)",
            )
            continue
        nearest = nearest_point(history, time_sec)
        if nearest is None:
            detail = f"нет точки в пределах {fmt_num(tolerance)}s"
        else:
            detail = (
                f"нет точки в пределах {fmt_num(tolerance)}s "
                f"(nearest t={fmt_num(nearest[1])}s winner={nearest[2]})"
            )
        report.add("FAIL", label, detail)


def check_stdout(report, spec, stdout_override):
    needles = spec.get("stdout_required_substrings")
    if not needles:
        return
    if not isinstance(needles, list):
        report.add("FAIL", "stdout_required_substrings", "ожидался список")
        return

    if stdout_override:
        path = resolve_cli_path(stdout_override)
        if not path.is_file():
            report.add("FAIL", "stdout", f"файл не найден: {display_path(path)}")
            return
    else:
        raw = spec.get("reference_stdout")
        if not raw:
            report.add(
                "SKIP",
                "stdout_required_substrings",
                "reference_stdout не задан",
            )
            return
        path = resolve_repo_path(raw)
        if not path.is_file():
            report.add(
                "SKIP",
                "stdout_required_substrings",
                f"лог не найден: {display_path(path)}",
            )
            return

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        report.add("FAIL", "stdout", f"не читается: {exc}")
        return

    for needle in needles:
        needle_text = str(needle)
        if needle_text in text:
            report.add("PASS", "stdout", f"найдено {needle_text!r}")
        else:
            report.add("FAIL", "stdout", f"нет подстроки {needle_text!r}")


def actual_path_for(spec, actual_override):
    if actual_override:
        return resolve_cli_path(actual_override), None
    raw = spec.get("reference_artifact")
    if not raw:
        return None, "не задан --actual и нет reference_artifact"
    return resolve_repo_path(raw), None


def verify_spec(report, spec_path, actual_override, stdout_override):
    if not spec_path.is_file():
        print(f"\n== {spec_path.name} ==")
        report.add("FAIL", "spec", f"файл не найден: {display_path(spec_path)}")
        return

    try:
        spec = load_yaml(spec_path)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        print(f"\n== {spec_path.name} ==")
        report.add("FAIL", "spec", f"не читается: {exc}")
        return

    title = spec.get("spec_id") or spec_path.stem
    print(f"\n== {title} ==")

    actual_path, path_error = actual_path_for(spec, actual_override)
    if path_error:
        report.add("FAIL", "artifact", path_error)
        check_stdout(report, spec, stdout_override)
        return

    print(f"    artifact: {display_path(actual_path)}")
    if not actual_path.is_file():
        report.add("FAIL", "artifact", f"файл не найден: {display_path(actual_path)}")
        check_stdout(report, spec, stdout_override)
        return

    try:
        actual = load_json(actual_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        report.add("FAIL", "artifact", f"не читается: {exc}")
        check_stdout(report, spec, stdout_override)
        return

    checks = spec.get("checks") or {}
    if not isinstance(checks, dict):
        report.add("FAIL", "checks", "ожидался словарь")
        checks = {}

    check_number(
        report,
        "total_detections",
        actual,
        "total_detections",
        checks.get("total_detections"),
    )
    check_number(
        report,
        "smart_impacts",
        actual,
        "smart_impacts",
        checks.get("smart_impacts"),
    )

    score_spec = checks.get("score") or {}
    if isinstance(score_spec, dict):
        check_score_side(report, actual, "player1", score_spec.get("player1"))
        check_score_side(report, actual, "player2", score_spec.get("player2"))
    elif score_spec:
        report.add("FAIL", "score", "ожидался словарь player1/player2")

    serve_bounds = checks.get("serve_count")
    serve_warn = spec.get("serve_order_errors") == "warning_only"
    check_number(
        report,
        "serve_count",
        actual,
        "serve_count",
        serve_bounds,
        warn_out_of_range=serve_warn,
    )
    check_score_balance(report, actual, checks.get("score_balance"))
    check_anchors(report, actual, spec.get("human_verified_anchors"))
    check_stdout(report, spec, stdout_override)


def main(argv=None):
    args = parse_args(argv)
    if not args.all and not args.spec:
        print("Укажите --spec <yaml> или --all", file=sys.stderr)
        return 1

    paths, error = collect_spec_paths(args)
    if error:
        print(error, file=sys.stderr)
        return 1
    if not paths:
        print("Нет файлов tests/expected/*.yaml", file=sys.stderr)
        return 1

    report = Report()
    for spec_path in paths:
        verify_spec(report, spec_path, args.actual, args.stdout)

    print(
        f"\nИтог: PASS={report.passed} WARN={report.warned} "
        f"FAIL={report.failed} SKIP={report.skipped}"
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
