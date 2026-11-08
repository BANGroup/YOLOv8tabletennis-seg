import datetime
import os
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

os.environ['HADOOP_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['YARN_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['JAVA_HOME']='/usr'
os.environ['SPARK_HOME'] ='/usr/lib/spark'
os.environ['PYTHONPATH'] ='/usr/local/lib/python3.8'

default_args = {
            'owner': 'airflow',
            'start_date':datetime.datetime(2020, 1, 1),
            }

dag_spark = DAG( dag_id='sparkoperator', default_args=default_args, schedule_interval=None,)

spark_submit = SparkSubmitOperator(task_id='spark_submit_task', dag=dag_spark, application='/home/username/partition.py', conn_id='yarn_spark', application_args=['2022-05-31', '/user/master/data/events', '/user/username/analitics/events'], conf={'spark.driver.maxResultSize': '20g'}, executor_cores=1, executor_memory='1g')

events_partitioned >> [verified_tags_candidates_d7, verified_tags_candidates_d84, user_interests_7, user_interests_28]

events_partitioned = SparkSubmitOperator(task_id='events_partitioned', dag=dag_spark, application='/home/username/partition_overwrite.py', conn_id='yarn_spark')
verified_tags_candidates = SparkSubmitOperator(task_id='verified_tags_candidates_d7', dag=dag_spark, application='/home/username/verified_tags_candidates.py', conn_id='yarn_spark')
verified_tags_candidates_d7 = verified_tags_candidates
verified_tags_candidates_d84 = SparkSubmitOperator(task_id='verified_tags_candidates_d84', dag=dag_spark, application='/home/username/verified_tags_candidates.py', conn_id='yarn_spark')
user_interests_7 = SparkSubmitOperator(task_id='user_interests_7', dag=dag_spark, application='/home/username/user_interests.py', conn_id='yarn_spark')
user_interests_28 = SparkSubmitOperator(task_id='user_interests_28', dag=dag_spark, application='/home/username/user_interests.py', conn_id='yarn_spark')
