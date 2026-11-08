import os
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.append(os.path.dirname(__file__))
from pipeline import refresh_mart

def _refresh(**context):
    refresh_mart(context["ds"])


with DAG(
    dag_id="2_datamart_update",
    start_date=datetime(2022, 10, 1),
    end_date=datetime(2022, 11, 1),
    schedule_interval="@daily",
    catchup=True,
    max_active_runs=1,
    tags=["final", "dwh"],
) as dag:
    PythonOperator(task_id="update_global_metrics", python_callable=_refresh)
