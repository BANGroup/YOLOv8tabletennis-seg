import os

import pendulum
import vertica_python
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator

SCHEMA = 'VT2609246827BE__STAGING'
TABLE = 'group_log'
DATA_PATH = '/data/group_log.csv'
COPY_COLS = '(group_id, user_id, user_id_from, event, datetime)'


def _conn_info() -> dict:
    return {
        'host': os.environ.get(
            'VERTICA_HOST', 'vertica.data-engineer.education-services.ru'
        ),
        'port': int(os.environ.get('VERTICA_PORT', '5433')),
        'user': os.environ['VERTICA_USER'],
        'password': os.environ['VERTICA_PASSWORD'],
        'database': os.environ.get('VERTICA_DB', 'dwh'),
        'autocommit': True,
        'tlsmode': 'disable',
    }


def load_group_log() -> None:
    with vertica_python.connect(**_conn_info()) as conn:
        cur = conn.cursor()
        role = os.environ.get('VERTICA_ROLE', 'VT2609246827BE_A')
        cur.execute(f'SET ROLE {role}')
        cur.execute(f'TRUNCATE TABLE {SCHEMA}.{TABLE}')
        copy_sql = (
            f'COPY {SCHEMA}.{TABLE} {COPY_COLS} FROM STDIN '
            "DELIMITER ',' ENCLOSED BY '\"' SKIP 1 "
            "NULL AS '' ABORT ON ERROR"
        )
        with open(DATA_PATH, 'rb') as data:
            cur.copy(copy_sql, data)


default_args = {'start_date': pendulum.parse('2022-07-13')}

with DAG(
    dag_id='sprint6_dag_load_group_log',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:
    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')

    load_task = PythonOperator(
        task_id='load_group_log',
        python_callable=load_group_log,
    )

    start >> load_task >> end
