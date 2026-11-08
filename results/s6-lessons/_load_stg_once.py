"""One-off STG loader for homework de06030302. Password via env only."""
import os

import vertica_python

SCHEMA = "VT2609246827BE__STAGING"

LOADS = (
    ("users", "/data/users.csv", "(id, chat_name, registration_dt, country, age)"),
    (
        "groups",
        "/data/groups.csv",
        "(id, admin_id, group_name, registration_dt, is_private)",
    ),
    (
        "dialogs",
        "/data/dialogs.csv",
        "(message_id, message_ts, message_from, message_to, message, message_group)",
    ),
)


def main() -> None:
    conn_info = {
        "host": os.environ["VERTICA_HOST"],
        "port": int(os.environ["VERTICA_PORT"]),
        "user": os.environ["VERTICA_USER"],
        "password": os.environ["VERTICA_PASSWORD"],
        "database": os.environ["VERTICA_DB"],
        "autocommit": True,
        "tlsmode": "disable",
    }
    with vertica_python.connect(**conn_info) as conn:
        cur = conn.cursor()
        cur.execute("SET ROLE VT2609246827BE_A")
        for table, path, cols in LOADS:
            cur.execute(f"TRUNCATE TABLE {SCHEMA}.{table}")
            if table == "dialogs":
                insert_sql = (
                    f"INSERT INTO {SCHEMA}.dialogs "
                    "(message_id, message_ts, message_from, message_to, message, message_group) "
                    "VALUES (%s, %s, %s, %s, %s, %s)"
                )
                batch: list[tuple] = []
                batch_size = 5000
                with open(path, newline="", encoding="utf-8") as src:
                    reader = csv.reader(src)
                    next(reader)
                    for row in reader:
                        mg = row[5] if row[5] != "" else None
                        batch.append(
                            (row[0], row[1], row[2], row[3], row[4], mg)
                        )
                        if len(batch) >= batch_size:
                            cur.executemany(insert_sql, batch)
                            batch.clear()
                    if batch:
                        cur.executemany(insert_sql, batch)
            else:
                copy_sql = (
                    f"COPY {SCHEMA}.{table} {cols} FROM STDIN "
                    f"DELIMITER ',' ENCLOSED BY '\"' SKIP 1 ABORT ON ERROR"
                )
                with open(path, "rb") as data:
                    cur.copy(copy_sql, data)
            cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.{table}")
            print(f"{table}: {cur.fetchone()[0]}")


if __name__ == "__main__":
    main()
