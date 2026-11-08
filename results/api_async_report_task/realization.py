#!/usr/bin/env python3
"""Клиент async API: generate_report → ожидание → get_report."""

import json
import sys
import time

import requests

BASE_URL = "https://d5dg1j9kt695d30blp03.apigw.yandexcloud.net"
HEADERS = {
    "X-API-KEY": "5f55e6c0-e9e5-4a9c-b313-63c01fc31460",
    "X-Nickname": "budnikmisch",
    "X-Cohort": "15",
}
MAX_RETRIES = 3
RETRY_DELAY_SEC = 5
GENERATE_WAIT_SEC = 120


def _is_retryable(status_code: int, body_text: str, payload: dict | None) -> bool:
    if 500 <= status_code < 600:
        return True
    if "socket hang up" in body_text.lower():
        return True
    if payload and "socket hang up" in str(payload.get("message", "")).lower():
        return True
    return False


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{BASE_URL}{path}"
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.request(method, url, headers=HEADERS, timeout=60, **kwargs)
        except requests.RequestException as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)
                continue
            raise RuntimeError(
                f"Сетевая ошибка после {MAX_RETRIES} попыток: {method} {url}"
            ) from exc

        body_text = response.text or ""
        payload = None
        try:
            payload = response.json()
        except ValueError:
            pass

        if _is_retryable(response.status_code, body_text, payload):
            last_error = RuntimeError(
                f"HTTP {response.status_code} от {url}: {body_text[:500]}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)
                continue
            raise last_error

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"HTTP {response.status_code} от {url}: {body_text[:500]}"
            ) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Ответ не JSON от {url}: {body_text[:500]}"
            ) from exc

    raise RuntimeError(f"Запрос не выполнен после {MAX_RETRIES} попыток: {method} {url}") from last_error


def _extract_task_id(payload: dict) -> str:
    task_id = payload.get("task_id")
    if not task_id and isinstance(payload.get("data"), dict):
        task_id = payload["data"].get("task_id")
    if not task_id:
        raise RuntimeError(f"Пустой task_id в ответе generate_report: {payload}")
    return str(task_id)


generate_report_response = _request("POST", "/generate_report")
task_id = _extract_task_id(generate_report_response)

time.sleep(GENERATE_WAIT_SEC)

get_report_response = _request("GET", "/get_report", params={"task_id": task_id})

status = get_report_response.get("status")
if status != "SUCCESS":
    raise RuntimeError(f"status != SUCCESS: {status!r}, ответ: {get_report_response}")

data = get_report_response.get("data")
if not isinstance(data, dict):
    raise RuntimeError(f"Отсутствует data в ответе get_report: {get_report_response}")

report_id = data.get("report_id")
if not report_id:
    raise RuntimeError(f"Пустой report_id в ответе get_report: {get_report_response}")


if __name__ == "__main__":
    try:
        print(json.dumps(get_report_response, ensure_ascii=False, indent=2))
        print(f"report_id: {report_id}")
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise
