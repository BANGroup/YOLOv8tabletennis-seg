import os
from datetime import datetime

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

os.environ["HADOOP_CONF_DIR"] = "/etc/hadoop/conf"
os.environ["YARN_CONF_DIR"] = "/etc/hadoop/conf"
os.environ["JAVA_HOME"] = "/usr"
os.environ["SPARK_HOME"] = "/usr/lib/spark"
os.environ["PYTHONPATH"] = "/usr/local/lib/python3.8"

USER = "s27560962"
SCRIPTS = f"/lessons/dags/scripts"
EVENTS_RAW = "/user/master/data/geo/events"
GEO_CSV = "/user/master/data/geo/geo.csv"
EVENTS_ODS = f"/user/{USER}/data/geo/events"
USER_MART = f"/user/{USER}/data/analytics/user_datamart"
ZONE_MART = f"/user/{USER}/data/analytics/zone_datamart"
FRIENDS_MART = f"/user/{USER}/data/analytics/friends_datamart"

default_args = {
    "owner": USER,
    "start_date": datetime(2022, 1, 1),
}

dag = DAG(
    dag_id="geo_analytics",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
)

spark_conf = {
    "spark.driver.maxResultSize": "4g",
    "spark.sql.shuffle.partitions": "200",
    "spark.pyspark.python": "python3",
    "spark.pyspark.driver.python": "/opt/venv38/bin/python3",
}

enrich_events = SparkSubmitOperator(
    task_id="enrich_events",
    dag=dag,
    application=f"{SCRIPTS}/enrich_events.py",
    conn_id="yarn_spark",
    application_args=["{{ ds }}", EVENTS_RAW, GEO_CSV, EVENTS_ODS],
    py_files=f"{SCRIPTS}/geo_common.py",
    conf=spark_conf,
    executor_cores=2,
    executor_memory="2g",
)

user_datamart = SparkSubmitOperator(
    task_id="user_datamart",
    dag=dag,
    application=f"{SCRIPTS}/user_datamart.py",
    conn_id="yarn_spark",
    application_args=[EVENTS_ODS, USER_MART],
    conf=spark_conf,
    executor_cores=2,
    executor_memory="2g",
)

zone_datamart = SparkSubmitOperator(
    task_id="zone_datamart",
    dag=dag,
    application=f"{SCRIPTS}/zone_datamart.py",
    conn_id="yarn_spark",
    application_args=[EVENTS_ODS, ZONE_MART],
    conf=spark_conf,
    executor_cores=2,
    executor_memory="2g",
)

friends_datamart = SparkSubmitOperator(
    task_id="friends_datamart",
    dag=dag,
    application=f"{SCRIPTS}/friends_datamart.py",
    conn_id="yarn_spark",
    application_args=[EVENTS_ODS, FRIENDS_MART],
    py_files=f"{SCRIPTS}/geo_common.py",
    conf=spark_conf,
    executor_cores=2,
    executor_memory="2g",
)

enrich_events >> [user_datamart, zone_datamart, friends_datamart]
