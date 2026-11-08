import logging

import pendulum
from airflow.decorators import dag, task
from delivery.stg_delivery_api_loader import DeliveryApiStgLoader
from lib import ConnectionBuilder

log = logging.getLogger(__name__)


@dag(
    dag_id="sprint5_stg_delivery_api_dag",
    schedule_interval="0/15 * * * *",
    start_date=pendulum.datetime(2022, 5, 5, tz="UTC"),
    catchup=False,
    tags=["sprint5", "stg", "delivery", "api"],
    is_paused_upon_creation=False,
)
def sprint5_stg_delivery_api_dag():
    dwh_pg_connect = ConnectionBuilder.pg_conn("PG_WAREHOUSE_CONNECTION")

    @task(task_id="couriers_load")
    def couriers_load():
        DeliveryApiStgLoader(dwh_pg_connect, log).load_couriers()

    @task(task_id="deliveries_load")
    def deliveries_load():
        DeliveryApiStgLoader(dwh_pg_connect, log).load_deliveries()

    couriers_load()
    deliveries_load()


sprint5_stg_delivery_api_dag = sprint5_stg_delivery_api_dag()
