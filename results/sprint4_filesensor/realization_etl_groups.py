from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
from airflow.utils.task_group import TaskGroup
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.sql import (
    SQLCheckOperator,
    SQLValueCheckOperator,
)

def check_failure_file_customer_research (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="failure_file_customer_research",
        sql="""
            INSERT INTO dq_checks_results
            values ('customer_research', 'file_sensor' ,current_date, 1 )
          """
    )

def check_success_file_customer_research (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="success_file_customer_research",
        sql="""
            INSERT INTO dq_checks_results
            values ('customer_research', 'file_sensor' ,current_date, 0 )
          """
    )

def check_success_file_user_order_log (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="success_file_user_order_log",
        sql="""
            INSERT INTO dq_checks_results
            values ('user_order_log', 'file_sensor' ,current_date, 0 )
          """
    )


def check_failure_file_user_order_log (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="failure_file_user_order_log",
        sql="""
            INSERT INTO dq_checks_results
            values ('user_order_log', 'file_sensor' ,current_date, 1 )
          """
    )

def check_success_file_user_activity_log (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="success_file_user_activity_log",
        sql="""
            INSERT INTO dq_checks_results
            values ('user_activity_log', 'file_sensor' ,current_date, 0 )
          """
    )

def check_failure_file_user_activity_log (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="failure_file_user_activity_log",
        sql="""
            INSERT INTO dq_checks_results
            values ('user_activity_log', 'file_sensor' ,current_date, 1 )
          """
    )


# для таблицы user_order_log первая проверка
def check_success_insert_user_order_log (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="success_insert_user_order_log",
        sql="""
            INSERT INTO dq_checks_results
            values ('user_order_log', 'user_order_log_isNull' ,current_date, 0 )
          """
    )
def check_failure_insert_user_order_log (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="failure_insert_user_order_log",
        sql="""
            INSERT INTO dq_checks_results
            values ('user_order_log', 'user_order_log_isNull' ,current_date, 1 )
          """
    )    
# для таблицы user_activity_log первая проверка
def check_success_insert_user_activity_log (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="success_insert_user_activity_log",
        sql="""
            INSERT INTO dq_checks_results
            values ('user_activity_log', 'user_activity_log_isNull' ,current_date, 0 )
          """
    )
def check_failure_insert_user_activity_log (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="failure_insert_user_activity_log",
        sql="""
            INSERT INTO dq_checks_results
            values ('user_activity_log', 'user_activity_log_isNull' ,current_date, 1 )
          """
    )  
# для таблицы user_order_log вторая проверка
def check_success_insert_user_order_log2 (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="success_insert_user_order_log2",
        sql="""
            INSERT INTO dq_checks_results
            values ('user_order_log', 'check_row_count_user_order_log' ,current_date, 0 )
          """
    )
def check_failure_insert_user_order_log2 (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="failure_insert_user_user_order_log2",
        sql="""
            INSERT INTO dq_checks_results
            values ('user_order_log', 'check_row_count_user_order_log' ,current_date, 1 )
          """
    )  

# для таблицы user_activity_log вторая проверка
def check_success_insert_user_activity_log2 (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="success_insert_user_activity_log2",
        sql="""
            INSERT INTO dq_checks_results
            values ('user_activity_log', 'check_row_count_user_activity_log' ,current_date, 0 )
          """
    )
def check_failure_insert_user_activity_log2 (context):
    insert_dq_checks_results = PostgresOperator(
        task_id="failure_insert_user_user_activity_log2",
        sql="""
            INSERT INTO dq_checks_results
            values ('user_activity_log', 'check_row_count_user_activity_log' ,current_date, 1 )
          """
    )  

def load_file_to_pg(filename,pg_table,conn_args):
    ''' csv-files to pandas dataframe '''
    f = pd.read_csv(filename)
    
    ''' load data to postgres '''
    cols = list(f.columns)
    insert_stmt = f"INSERT INTO {pg_table} ({cols}) VALUES %s"
    
    pg_conn = psycopg2.connect(**conn_args)
    cur = pg_conn.cursor()
    psycopg2.extras.execute_values(cur, insert_stmt, f.values)
    pg_conn.commit()
    cur.close()
    pg_conn.close()


default_args = {
    "start_date": datetime(2020, 1, 1),
    "owner": "airflow",
    "conn_id": "postgres_default"
}



with DAG(dag_id="Sprin4_Task71", schedule_interval="@daily", default_args=default_args, catchup=False) as dag:

    begin = DummyOperator(task_id="begin")
    with TaskGroup(group_id='group1') as fg1:
        f1 = FileSensor(task_id="waiting_for_file_customer_research", fs_conn_id="fs_local", filepath=str(datetime.now().date()) + "customer_research.csv", poke_interval=5 , on_success_callback = check_success_file_customer_research, on_failure_callback =  check_failure_file_customer_research)
        f2 = FileSensor(task_id="waiting_for_file_user_order_log", fs_conn_id="fs_local", filepath=str(datetime.now().date()) + "user_order_log.csv", poke_interval=5 , on_success_callback = check_success_file_user_order_log, on_failure_callback =  check_failure_file_user_order_log)
        f3 = FileSensor(task_id="waiting_for_file_user_activity_log", fs_conn_id="fs_local", filepath=str(datetime.now().date()) + "user_activity_log.csv", poke_interval=5 , on_success_callback = check_success_file_user_activity_log, on_failure_callback =  check_failure_file_user_activity_log)
    
    with TaskGroup(group_id='group2') as fg2:
        load_customer_research = PythonOperator(task_id='load_customer_research', python_callable=load_file_to_pg, op_kwargs={'filename': str(datetime.now().date()) + "customer_research.csv",'pg_table': 'stage.customer_research'})
        load_user_order_log = PythonOperator(task_id='load_user_order_log', python_callable=load_file_to_pg,op_kwargs={'filename': str(datetime.now().date()) + "user_order_log.csv",'pg_table': 'stage.user_order_log'})
        load_user_activity_log = PythonOperator(task_id='load_user_activity_log',python_callable=load_file_to_pg,op_kwargs={'filename': str(datetime.now().date()) + "user_activity_log.csv",'pg_table': 'stage.user_activity_log'})
        load_price_log = PythonOperator(task_id='load_price_log',python_callable=load_file_to_pg,op_kwargs={'filename': str(datetime.now().date()) + "price_log.csv",'pg_table': 'stage.price_log'})
        sql_check = SQLCheckOperator(task_id="user_order_log_isNull", sql="user_order_log_isNull_check.sql" , on_success_callback = check_success_insert_user_order_log, on_failure_callback =  check_failure_insert_user_order_log )
        sql_check2 = SQLCheckOperator(task_id="user_activity_log_isNull", sql="user_activity_log_isNull_check.sql", on_success_callback = check_success_insert_user_activity_log, on_failure_callback =  check_failure_insert_user_activity_log)
        sql_check3 = SQLValueCheckOperator(task_id="check_row_count_user_order_log", sql="Select count(distinct(customer_id)) from user_order_log", pass_value=3 ,  on_success_callback = check_success_insert_user_order_log2, on_failure_callback =  check_failure_insert_user_order_log2)
        sql_check4 = SQLValueCheckOperator(task_id="check_row_count_user_activity_log", sql="Select count(distinct(customer_id)) from user_activity_log", pass_value=3 , on_success_callback = check_success_insert_user_activity_log2, on_failure_callback =  check_failure_insert_user_activity_log2)    
        load_user_order_log >> [sql_check , sql_check3 ]
        load_user_activity_log >> [ sql_check2 , sql_check4 ]
    end = DummyOperator(task_id="end")

    begin >> fg1 >> fg2 >> end