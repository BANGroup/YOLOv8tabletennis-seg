import csv
import os

import pendulum
import vertica_python
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator

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


def _conn_info() -> dict:
    return {
        "host": os.environ.get(
            "VERTICA_HOST", "vertica.data-engineer.education-services.ru"
        ),
        "port": int(os.environ.get("VERTICA_PORT", "5433")),
        "user": os.environ["VERTICA_USER"],
        "password": os.environ["VERTICA_PASSWORD"],
        "database": os.environ.get("VERTICA_DB", "dwh"),
        "autocommit": True,
        "tlsmode": "disable",
    }


def load_table(table: str, path: str, cols: str) -> None:
    with vertica_python.connect(**_conn_info()) as conn:
        cur = conn.cursor()
        role = os.environ.get("VERTICA_ROLE", "VT2609246827BE_A")
        cur.execute(f"SET ROLE {role}")
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
                    batch.append((row[0], row[1], row[2], row[3], row[4], mg))
                    if len(batch) >= batch_size:
                        cur.executemany(insert_sql, batch)
                        batch.clear()
                if batch:
                    cur.executemany(insert_sql, batch)
        else:
            copy_sql = (
                f"COPY {SCHEMA}.{table} {cols} FROM STDIN "
                "DELIMITER ',' ENCLOSED BY '\"' SKIP 1 ABORT ON ERROR"
            )
            with open(path, "rb") as data:
                cur.copy(copy_sql, data)


default_args = {"start_date": pendulum.parse("2022-07-13")}

with DAG(
    dag_id="sprint6_dag_load_stg",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:
    start = DummyOperator(task_id="start")
    end = DummyOperator(task_id="end")

    load_users = PythonOperator(
        task_id="load_users",
        python_callable=load_table,
        op_args=LOADS[0],
    )
    load_groups = PythonOperator(
        task_id="load_groups",
        python_callable=load_table,
        op_args=LOADS[1],
    )
    load_dialogs = PythonOperator(
        task_id="load_dialogs",
        python_callable=load_table,
        op_args=LOADS[2],
    )

    start >> [load_users, load_groups, load_dialogs] >> end
