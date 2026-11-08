import os
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.append(os.path.dirname(__file__))
from pipeline import load_day

def _load(**context):
    load_day(context["ds"])


with DAG(
    dag_id="1_data_import",
    start_date=datetime(2022, 10, 1),
    end_date=datetime(2022, 11, 1),
    schedule_interval="@daily",
    catchup=True,
    max_active_runs=1,
    tags=["final", "staging"],
) as dag:
    PythonOperator(task_id="load_staging", python_callable=_load)
