"""One-off loader: BAD_IDEA MERGEOUT + COPY. Password only via env VERTICA_PASSWORD."""
import csv
import os
from pathlib import Path

import vertica_python

N = 10_000
SCHEMA_CANDIDATES = (
    os.environ.get("VERTICA_USER", ""),
    os.environ.get("VERTICA_USER", "").upper(),
)


def main() -> None:
    user = os.environ["VERTICA_USER"]
    password = os.environ["VERTICA_PASSWORD"]
    base = Path(__file__).resolve().parent
    csv_path = base / "bad_idea_10000.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        for i in range(N):
            writer.writerow([i, "asds"])
    print(csv_path.resolve())

    conn_info = {
        "host": os.environ.get("VERTICA_HOST", "vertica.data-engineer.education-services.ru"),
        "port": int(os.environ.get("VERTICA_PORT", "5433")),
        "user": user,
        "password": password,
        "database": os.environ.get("VERTICA_DB", "dwh"),
        "autocommit": True,
        "tlsmode": "disable",
    }

    with vertica_python.connect(**conn_info) as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1")
        schema = user
        cur.execute(
            """
            SELECT table_schema FROM v_catalog.tables
            WHERE table_name = 'BAD_IDEA'
              AND table_schema ILIKE %s
            """,
            (user,),
        )
        row = cur.fetchone()
        if row:
            schema = row[0]
        else:
            cur.execute(f"CREATE TABLE {schema}.BAD_IDEA (i INT, v VARCHAR(16))")

        cur.execute(f"ALTER TABLE {schema}.BAD_IDEA SET MERGEOUT 1")

        cur.execute(f"SELECT COUNT(*) FROM {schema}.BAD_IDEA")
        if cur.fetchone()[0] == 0:
            copy_sql = (
                f"COPY {schema}.BAD_IDEA (i, v) FROM STDIN DELIMITER '|' ABORT ON ERROR"
            )
            with csv_path.open("rb") as data:
                cur.copy(copy_sql, data)

        cur.execute(f"SELECT COUNT(*) FROM {schema}.BAD_IDEA")
        print("count:", cur.fetchone()[0])


if __name__ == "__main__":
    main()
