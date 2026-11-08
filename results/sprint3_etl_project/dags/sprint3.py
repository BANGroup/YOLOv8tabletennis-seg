import io
import time

import pandas as pd
import requests

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.hooks.http_hook import HttpHook

http_conn_id = HttpHook.get_connection('http_conn_id')
api_key = http_conn_id.extra_dejson.get('api_key')
base_url = http_conn_id.host.rstrip('/')

postgres_conn_id = 'postgresql_de'

nickname = 'budnikmisch'
cohort = '15'

headers = {
    'X-Nickname': nickname,
    'X-Cohort': cohort,
    'X-Project': 'True',
    'X-API-KEY': api_key,
    'Content-Type': 'application/x-www-form-urlencoded',
}

S3_BASE = f'https://storage.yandexcloud.net/s3-sprint3/cohort_{cohort}/{nickname}/project'

REPORT_FILES = (
    'customer_research.csv',
    'user_order_log.csv',
    'user_activity_log.csv',
    'price_log.csv',
)

INCREMENT_FILES = (
    'customer_research_inc.csv',
    'user_order_log_inc.csv',
    'user_activity_log_inc.csv',
)


def _s3_url(folder_id, filename):
    return f'{S3_BASE}/{folder_id}/{filename}'


def _download_csv(url):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def generate_report(ti):
    response = requests.post(f'{base_url}/generate_report', headers=headers, timeout=60)
    response.raise_for_status()
    payload = response.json()
    task_id = payload.get('task_id')
    if not task_id and isinstance(payload.get('data'), dict):
        task_id = payload['data'].get('task_id')
    if not task_id:
        raise ValueError('generate_report: empty task_id')
    ti.xcom_push(key='task_id', value=task_id)


def get_report(ti):
    task_id = ti.xcom_pull(key='task_id')
    report_id = None
    for _ in range(30):
        response = requests.get(
            f'{base_url}/get_report',
            params={'task_id': task_id},
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        if body.get('status') == 'SUCCESS':
            report_id = body['data']['report_id']
            break
        time.sleep(10)
    if not report_id:
        raise TimeoutError('get_report: report not ready')
    ti.xcom_push(key='report_id', value=report_id)


def get_increment(date, ti):
    report_id = ti.xcom_pull(key='report_id')
    response = requests.get(
        f'{base_url}/get_increment',
        params={'report_id': report_id, 'date': f'{date}T00:00:00'},
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    increment_id = None
    if body.get('status') == 'SUCCESS':
        increment_id = body.get('data', {}).get('increment_id')
    ti.xcom_push(key='increment_id', value=increment_id)


def _staging_has_orders():
    hook = PostgresHook(postgres_conn_id)
    row = hook.get_first('select count(*) from staging.user_order_log')
    return row and row[0] > 0


def _prepare_user_order_df(content):
    df = pd.read_csv(io.BytesIO(content))
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df = df.drop_duplicates(subset=['uniq_id'])
    if 'status' not in df.columns:
        df['status'] = 'shipped'
    else:
        df['status'] = df['status'].fillna('shipped')
    return df


def _load_price_log(content, engine):
    lines = content.decode('utf-8').splitlines()
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        item_name, price = line.rsplit(',', 1)
        rows.append({'item_name': item_name.strip(), 'price': float(price.strip())})
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    df.to_sql('price_log', engine, schema='staging', if_exists='append', index=False)
    return len(df)


def _append_dataframe(df, table, engine):
    if df.empty:
        return 0
    df.to_sql(table, engine, schema='staging', if_exists='append', index=False)
    return len(df)


def upload_full_report(ti):
    if _staging_has_orders():
        return
    report_id = ti.xcom_pull(key='report_id')
    postgres_hook = PostgresHook(postgres_conn_id)
    engine = postgres_hook.get_sqlalchemy_engine()

    for filename in REPORT_FILES:
        url = _s3_url(report_id, filename)
        content = _download_csv(url)
        if filename == 'price_log.csv':
            postgres_hook.run('truncate table staging.price_log')
            _load_price_log(content, engine)
            continue
        if filename == 'user_order_log.csv':
            df = _prepare_user_order_df(content)
            _append_dataframe(df, 'user_order_log', engine)
            continue
        df = pd.read_csv(io.BytesIO(content))
        if 'id' in df.columns and filename == 'user_activity_log.csv':
            df = df.drop(columns=['id'])
        if 'uniq_id' in df.columns:
            df = df.drop_duplicates(subset=['uniq_id'])
        table = filename.replace('.csv', '')
        _append_dataframe(df, table, engine)


def upload_increment_files(date, ti):
    increment_id = ti.xcom_pull(key='increment_id')
    if not increment_id:
        return
    postgres_hook = PostgresHook(postgres_conn_id)
    engine = postgres_hook.get_sqlalchemy_engine()
    postgres_hook.run(
        """
        delete from staging.user_order_log where date_time::date = %s::date;
        delete from staging.user_activity_log where date_time::date = %s::date;
        delete from staging.customer_research cr
        using mart.d_calendar dc
        where cr.date_id = dc.date_id and dc.date_actual = %s::date;
        """,
        parameters=(date, date, date),
    )

    for filename in INCREMENT_FILES:
        url = _s3_url(increment_id, filename)
        content = _download_csv(url)
        if filename == 'user_order_log_inc.csv':
            df = _prepare_user_order_df(content)
            _append_dataframe(df, 'user_order_log', engine)
            continue
        df = pd.read_csv(io.BytesIO(content))
        if 'id' in df.columns and filename == 'user_activity_log_inc.csv':
            df = df.drop(columns=['id'])
        if 'uniq_id' in df.columns:
            df = df.drop_duplicates(subset=['uniq_id'])
        table = filename.replace('_inc.csv', '')
        _append_dataframe(df, table, engine)


args = {
    'owner': 'student',
    'email': ['student@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

business_dt = '{{ ds }}'

with DAG(
    'sales_mart',
    default_args=args,
    description='Sprint3 project: f_sales + f_customer_retention',
    catchup=True,
    schedule_interval=None,
    start_date=datetime(2026, 8, 23),
    end_date=datetime(2026, 9, 21),
) as dag:
    migrate_schema = PostgresOperator(
        task_id='migrate_schema',
        postgres_conn_id=postgres_conn_id,
        sql='sql/00_migrate_schema.sql',
    )

    generate_report_task = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
    )

    get_report_task = PythonOperator(
        task_id='get_report',
        python_callable=get_report,
    )

    upload_full_report_task = PythonOperator(
        task_id='upload_full_report',
        python_callable=upload_full_report,
    )

    get_increment_task = PythonOperator(
        task_id='get_increment',
        python_callable=get_increment,
        op_kwargs={'date': business_dt},
    )

    upload_increment_task = PythonOperator(
        task_id='upload_increment_files',
        python_callable=upload_increment_files,
        op_kwargs={'date': business_dt},
    )

    update_d_item_table = PostgresOperator(
        task_id='update_d_item',
        postgres_conn_id=postgres_conn_id,
        sql='sql/mart.d_item.sql',
    )

    update_d_customer_table = PostgresOperator(
        task_id='update_d_customer',
        postgres_conn_id=postgres_conn_id,
        sql='sql/mart.d_customer.sql',
    )

    update_d_city_table = PostgresOperator(
        task_id='update_d_city',
        postgres_conn_id=postgres_conn_id,
        sql='sql/mart.d_city.sql',
    )

    update_f_sales = PostgresOperator(
        task_id='update_f_sales',
        postgres_conn_id=postgres_conn_id,
        sql='sql/mart.f_sales.sql',
    )

    update_f_customer_retention = PostgresOperator(
        task_id='update_f_customer_retention',
        postgres_conn_id=postgres_conn_id,
        sql='sql/mart.f_customer_retention.sql',
    )

    (
        migrate_schema
        >> generate_report_task
        >> get_report_task
        >> upload_full_report_task
        >> get_increment_task
        >> upload_increment_task
        >> [update_d_item_table, update_d_city_table, update_d_customer_table]
        >> update_f_sales
        >> update_f_customer_retention
    )
