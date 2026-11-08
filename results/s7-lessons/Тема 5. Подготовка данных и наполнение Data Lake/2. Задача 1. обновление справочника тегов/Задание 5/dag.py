import datetime
import os

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

os.environ['HADOOP_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['YARN_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['JAVA_HOME'] = '/usr'
os.environ['SPARK_HOME'] = '/usr/lib/spark'
os.environ['PYTHONPATH'] = '/usr/local/lib/python3.8'

default_args = {
    'owner': 'airflow',
    'start_date': datetime.datetime(2022, 5, 31),
}

dag = DAG(
    dag_id='sparksubmit_operator_dag',
    schedule_interval=None,
    default_args=default_args,
)

spark_submit_local = SparkSubmitOperator(
    task_id='spark_submit_task',
    dag=dag,
    application='partition.py',
    conn_id='yarn_spark',
    application_args=[
        '2022-05-31',
        '/user/master/data/events',
        '/user/s27560962/data/events',
    ],
)

verified_tags_candidates = SparkSubmitOperator(
    task_id='verified_tags_candidates',
    dag=dag,
    application='verified_tags_candidates.py',
    conn_id='yarn_spark',
    application_args=[
        '2022-05-31',
        '5',
        '300',
        '/user/s27560962/data/events',
        '/user/master/data/snapshots/tags_verified/actual',
        '/user/s27560962/data/analytics/verified_tags_candidates_d5',
    ],
)

spark_submit_local >> verified_tags_candidates
