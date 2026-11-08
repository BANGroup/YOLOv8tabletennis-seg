#!/usr/bin/env python3
"""MERGEOUT + COPY для BAD_IDEA (de06020501). Нужны env VERTICA_USER и VERTICA_PASSWORD."""

import os
import sys

import vertica_python

LESSON_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(LESSON_DIR, "bad_idea.csv")
N_ROWS = 10_000

HOST = "vertica.data-engineer.education-services.ru"
PORT = 5433
DATABASE = "dwh"


def main() -> int:
    user = os.environ.get("VERTICA_USER", "").strip()
    password = os.environ.get("VERTICA_PASSWORD", "").strip()
    if not user or not password:
        print(
            "Задайте VERTICA_USER и VERTICA_PASSWORD (логин/пароль из сниппета Практикума).",
            file=sys.stderr,
        )
        return 1

    student_schema = user.upper()
    conn_info = {
        "host": HOST,
        "port": PORT,
        "user": user,
        "password": password,
        "database": DATABASE,
        "autocommit": True,
        "tlsmode": "disable",
    }

    with vertica_python.connect(**conn_info) as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 AS a1")
        if cur.fetchone()[0] != 1:
            raise RuntimeError("SELECT 1 failed")

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {student_schema}.BAD_IDEA (
                i INT,
                v VARCHAR(16)
            )
            """
        )
        cur.execute(f"ALTER TABLE {student_schema}.BAD_IDEA SET MERGEOUT 1")

    with open(CSV_PATH, "w", encoding="utf-8") as f:
        for i in range(N_ROWS):
            f.write(f"{i}|asds\n")

    with vertica_python.connect(**conn_info) as conn:
        cur = conn.cursor()
        cur.execute(f"TRUNCATE TABLE {student_schema}.BAD_IDEA")
        copy_sql = (
            f"COPY {student_schema}.BAD_IDEA (i, v) "
            f"FROM LOCAL '{CSV_PATH}' DELIMITER '|'"
        )
        cur.execute(copy_sql)
        cur.execute(f"SELECT COUNT(*) FROM {student_schema}.BAD_IDEA")
        cnt = cur.fetchone()[0]

    if cnt != N_ROWS:
        print(f"Ожидалось {N_ROWS} строк, получено {cnt}", file=sys.stderr)
        return 1

    rlz_path = os.path.join(LESSON_DIR, "realization.py")
    with open(rlz_path, "w", encoding="utf-8") as f:
        f.write(f"vertica_user = '{student_schema}'\n")

    print(f"OK: schema={student_schema}, rows={cnt}, csv={CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
