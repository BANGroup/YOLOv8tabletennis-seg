"""Shared HTTP client; never imports or executes realization.py/settings.py."""

import json
import math
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

DEFAULT_URL = "https://de-sp8-checks.de.education-services.ru"
MAX_SOURCE = 200000


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def submit(t_code, rlz_file, full_lesson_path):
    try:
        if not re.fullmatch(r"de\d{8}", t_code):
            raise ValueError("Некорректный идентификатор задания.")
        url = os.environ.get("S8_CHECKS_URL", DEFAULT_URL).rstrip("/")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in ("https", "http")
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("S8_CHECKS_URL должен быть HTTP(S)-адресом сервиса.")
        timeout = float(os.environ.get("S8_CHECKS_TIMEOUT", "150"))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(
                "S8_CHECKS_TIMEOUT должен быть положительным числом секунд."
            )
        path = Path(full_lesson_path) / rlz_file
        with path.open("rb") as f:
            raw = f.read(MAX_SOURCE + 1)
        if len(raw) > MAX_SOURCE:
            raise ValueError("Файл решения превышает 200000 байт.")
        source = raw.decode("utf-8-sig")
        if not source.strip():
            raise ValueError("Заполните realization.py перед отправкой.")
        body = json.dumps({"source": source}, ensure_ascii=False).encode("utf-8")
        request = Request(
            url + "/api/v1/checks/" + t_code,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with build_opener(NoRedirect()).open(request, timeout=timeout) as reply:
                status, raw_reply = reply.status, reply.read(65537)
        except HTTPError as e:
            with e:
                status, raw_reply = e.code, e.read(65537)
        if len(raw_reply) > 65536:
            raise ValueError("Сервис вернул слишком большой ответ.")
        try:
            result = json.loads(raw_reply.decode("utf-8"))
        except (ValueError, UnicodeError):
            raise ValueError(f"Некорректный ответ сервиса (HTTP {status}).") from None
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("message"), str)
            or result.get("status")
            not in (
                "passed",
                "failed",
                "error",
                "busy",
                "invalid_request",
                "unknown_task",
            )
        ):
            raise ValueError(f"Некорректный ответ сервиса (HTTP {status}).")
        if result.get("task_id", t_code) != t_code:
            raise ValueError("Ответ сервиса относится к другому заданию.")
        passed = status == 200 and result["status"] == "passed"
        if passed:
            if (
                result.get("task_id") != t_code
                or not isinstance(result.get("code"), str)
                or not re.fullmatch(r"[A-Za-z0-9]{10}", result["code"])
            ):
                raise ValueError("В ответе нет корректного кода задания.")
            print("Проверка пройдена. Ваш код: " + result["code"])
            return 0
        if "code" in result:
            raise ValueError("Некорректный ответ: код получен без успешной проверки.")
        # Strip terminal control characters from service feedback.
        print("".join(c for c in result["message"] if c >= " " or c == "\n"))
        return 1
    except (TimeoutError, URLError):
        print(
            "Сервис недоступен или время ожидания истекло. Проверьте соединение и повторите отправку."
        )
    except (OSError, UnicodeError):
        print("Не удалось прочитать UTF-8 файл решения или получить ответ сервиса.")
    except ValueError as e:
        print(str(e))
    return 2
