from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
from airflow.utils.task_group import TaskGroup
from airflow.operators.sql import (
    SQLCheckOperator,
    SQLValueCheckOperator,
)

default_args = {
    "start_date": datetime(2020, 1, 1),
    "owner": "airflow",
    "conn_id": "postgres_default"

}

with DAG(dag_id="Sprin4_Task51", schedule_interval="@daily", default_args=default_args, catchup=False) as dag:
    sql_check = SQLCheckOperator(task_id="user_order_log_isNull", sql="user_order_log_isNull_check.sql" )
    sql_check2 = SQLCheckOperator(task_id="user_activity_log_isNull", sql="user_activity_log_isNull_check.sql")
    sql_check3 = SQLValueCheckOperator(task_id="check_row_count_user_order_log", sql="Select count(distinct(customer_id)) from user_order_log", pass_value=3)
    sql_check4 = SQLValueCheckOperator(task_id="check_row_count_user_activity_log", sql="Select count(distinct(customer_id)) from user_activity_log", pass_value=3)    

    sql_check >> sql_check2 >> sql_check3 >> sql_check4
