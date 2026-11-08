from airflow.models import DAG
from airflow.sensors.filesystem import FileSensor
from datetime import datetime

default_args = {
    "start_date": datetime(2020, 1, 1),
    "owner": "airflow",
}

with DAG(
    dag_id="Sprin4_Task1",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
) as dag:

    waiting_for_file = FileSensor(
        task_id="waiting_for_file",
        fs_conn_id="fs_local",
        filepath="/data/test.txt",
        poke_interval=5,
    )

    waiting_for_file
