#!/usr/bin/env python3
"""One-shot generator for 583_postgresql_mart_update DAG files in vsc container paths."""
import ast
import hashlib
import textwrap
from pathlib import Path


LESSON8_PREFIX = "/lessons/5. Реализация ETL в Airflow/8. Load обновление слоя витрин"


def find_realization(n: int) -> Path:
    path = Path(f"{LESSON8_PREFIX}/Задание {n}/realization.sql")
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return path


def find_assignment3_realization() -> Path:
    path = Path(f"{LESSON8_PREFIX}/Задание 3/realization.py")
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return path


def main() -> None:
    sql1 = find_realization(1).read_text()
    sql2 = find_realization(2).read_text()

    _, after_f_deletes = sql1.split("delete from mart.f_daily_sales;", 1)
    after_f_deletes = after_f_deletes.lstrip("\n")
    cal_block, after_cal = after_f_deletes.split("--d_customer", 1)
    cal_block = cal_block.strip()
    if not cal_block.endswith(";"):
        cal_block += ";"

    cust_block, after_cust = after_cal.split("--d_item", 1)
    cust_block = cust_block.strip()
    if not cust_block.endswith(";"):
        cust_block += ";"

    item_block = after_cust.strip()
    if not item_block.endswith(";"):
        item_block += ";"

    d_statements = [
        "delete from mart.f_activity;",
        "delete from mart.f_daily_sales;",
        cal_block,
        cust_block,
        item_block,
    ]

    fa_block, after_fa = sql2.split("--f_daily_sales", 1)
    fa_block = fa_block.replace("--f_activity", "").strip()
    if not fa_block.endswith(";"):
        fa_block += ";"

    fs_lines = [
        ln
        for ln in ("--f_daily_sales" + after_fa).splitlines()
        if not ln.strip().startswith("--")
    ]
    fs_block = "\n".join(fs_lines).strip()
    if not fs_block.endswith(";"):
        fs_block += ";"

    fa_delete, fa_insert = fa_block.split("INSERT INTO", 1)
    fa_delete = fa_delete.strip()
    fa_insert = ("INSERT INTO" + fa_insert).strip()
    if not fa_delete.endswith(";"):
        fa_delete += ";"
    if not fa_insert.endswith(";"):
        fa_insert += ";"

    fs_delete, fs_insert = fs_block.split("INSERT INTO", 1)
    fs_delete = fs_delete.strip()
    fs_insert = ("INSERT INTO" + fs_insert).strip()
    if not fs_delete.endswith(";"):
        fs_delete += ";"
    if not fs_insert.endswith(";"):
        fs_insert += ";"

    f_statements = [fa_delete, fa_insert, fs_delete, fs_insert]

    d_exec_lines = []
    for stmt in d_statements:
        d_exec_lines.append(f"        cur.execute({stmt!r})")
        d_exec_lines.append("        conn.commit()")

    f_exec_lines = []
    for stmt in f_statements:
        f_exec_lines.append(f"        cur.execute({stmt!r})")
        f_exec_lines.append("        conn.commit()")

    content = textwrap.dedent(
        f"""\
from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.operators.python import PythonOperator

import datetime
import psycopg2

dag = DAG(
    dag_id='583_postgresql_mart_update',
    schedule_interval=None,
    start_date=datetime.datetime(2021, 1, 1),
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=60),
    is_paused_upon_creation=True,
)


def _pg_connect():
    conn_args = BaseHook.get_connection('pg_connection')
    return psycopg2.connect(
        dbname=conn_args.schema,
        port=conn_args.port,
        user=conn_args.login,
        host=conn_args.host,
        password=conn_args.password,
    )


def update_mart_d_tables(ti):
    conn = None
    cur = None
    try:
        conn = _pg_connect()
        cur = conn.cursor()
{chr(10).join(d_exec_lines)}
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
    return 200


def update_mart_f_tables(ti):
    conn = None
    cur = None
    try:
        conn = _pg_connect()
        cur = conn.cursor()
{chr(10).join(f_exec_lines)}
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
    return 200


t_update_mart_d_tables = PythonOperator(
    task_id='update_mart_d_tables',
    python_callable=update_mart_d_tables,
    dag=dag,
    retries=0,
)

t_update_mart_f_tables = PythonOperator(
    task_id='update_mart_f_tables',
    python_callable=update_mart_f_tables,
    dag=dag,
    retries=0,
)

t_update_mart_d_tables >> t_update_mart_f_tables
"""
    )

    dag_path = Path("/lessons/dags/583_postgresql_mart_update.py")
    if dag_path.exists():
        raise SystemExit("583_postgresql_mart_update.py already exists; stop")

    realization_path = find_assignment3_realization()
    for path in (realization_path, dag_path):
        path.write_text(content)

    for path in (realization_path, dag_path):
        ast.parse(path.read_text())

    digest = hashlib.sha256(content.encode()).hexdigest()
    print("sha256", digest)
    print("realization", realization_path)
    print("dag", dag_path)
    print("bytes", len(content.encode()))


if __name__ == "__main__":
    main()
