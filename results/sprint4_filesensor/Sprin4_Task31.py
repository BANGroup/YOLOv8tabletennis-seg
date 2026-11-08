from datetime import datetime

from airflow.models import DAG
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor

default_args = {
    "start_date": datetime(2020, 1, 1),
    "retries": 0,
}

with DAG(
    dag_id="Sprin4_Task31",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
    is_paused_upon_creation=True,
) as dag:
    start = DummyOperator(task_id="Start")
    waiting_for_file = FileSensor(
        task_id="waiting_for_file",
        fs_conn_id="fs_local",
        filepath="test.txt",
        poke_interval=5,
        timeout=60,
        mode="poke",
    )
    finish = DummyOperator(task_id="Finish")

    start >> waiting_for_file >> finish
