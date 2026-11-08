import urllib.request

import pendulum
from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

bucket_files = ['group_log.csv']


def fetch_s3_file(bucket: str, key: str) -> None:
    url = f'https://storage.yandexcloud.net/{bucket}/{key}'
    dest_path = f'/data/{key}'
    urllib.request.urlretrieve(url, dest_path)


@dag(
    dag_id='sprint6_dag_get_group_log',
    schedule_interval=None,
    start_date=pendulum.parse('2022-07-13'),
    catchup=False,
)
def sprint6_dag_get_group_log():
    fetch_tasks = []
    for f in bucket_files:
        fetch_tasks.append(
            PythonOperator(
                task_id=f'fetch_{f.replace(".", "_")}',
                python_callable=fetch_s3_file,
                op_kwargs={'bucket': 'sprint6', 'key': f},
            )
        )

    bash_command_tmpl = """
{% for f in params.files %}
echo "{{ f }}"
head -n 10 {{ f }}
{% endfor %}
"""
    print_10_lines_of_each = BashOperator(
        task_id='print_10_lines_of_each',
        bash_command=bash_command_tmpl,
        params={'files': [f'/data/{f}' for f in bucket_files]},
    )

    fetch_tasks >> print_10_lines_of_each


_ = sprint6_dag_get_group_log()
