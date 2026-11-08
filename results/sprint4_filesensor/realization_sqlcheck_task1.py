from airflow import DAG
from datetime import datetime
from airflow.operators.dummy import DummyOperator
from airflow.operators.sql import SQLCheckOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup

default_args = {
    "start_date": datetime(2020, 1, 1),
    "owner": "airflow",
    "conn_id": "postgres_default",
}


with DAG(
    dag_id="Sprin4_Task1",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
) as dag:

    start = DummyOperator(task_id="Start")

    with TaskGroup(group_id="group1") as fg1:
        f1 = FileSensor(
            task_id="waiting_for_file_customer_research",
            fs_conn_id="fs_local",
            filepath=str(datetime.now().date()) + "customer_research.csv",
            poke_interval=5,
        )
        f2 = FileSensor(
            task_id="waiting_for_file_user_order_log",
            fs_conn_id="fs_local",
            filepath=str(datetime.now().date()) + "user_order_log.csv",
            poke_interval=5,
        )
        f3 = FileSensor(
            task_id="waiting_for_file_user_activity_log",
            fs_conn_id="fs_local",
            filepath=str(datetime.now().date()) + "user_activity_log.csv",
            poke_interval=5,
        )

    sql_check = SQLCheckOperator(
        task_id="user_order_log_isNull",
        sql="user_order_log_isNull_check.sql",
    )
    sql_check2 = SQLCheckOperator(
        task_id="user_activity_log_isNull",
        sql="user_activity_log_isNull_check.sql",
    )

    finish = DummyOperator(task_id="Finish")

    start >> fg1 >> sql_check
    sql_check >> sql_check2
    sql_check2 >> finish
