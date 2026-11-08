import logging

import pendulum
from airflow.decorators import dag, task
from delivery.couriers_dds_loader import CouriersDdsLoader
from delivery.fct_deliveries_loader import FctDeliveriesLoader
from lib import ConnectionBuilder

log = logging.getLogger(__name__)


@dag(
    dag_id="sprint5_dds_delivery_dag",
    schedule_interval="0/15 * * * *",
    start_date=pendulum.datetime(2022, 5, 5, tz="UTC"),
    catchup=False,
    tags=["sprint5", "dds", "delivery"],
    is_paused_upon_creation=False,
)
def sprint5_dds_delivery_dag():
    dwh_pg_connect = ConnectionBuilder.pg_conn("PG_WAREHOUSE_CONNECTION")

    @task(task_id="couriers_dds_load")
    def couriers_dds_load():
        CouriersDdsLoader(dwh_pg_connect, log).load_couriers()

    @task(task_id="fct_deliveries_load")
    def fct_deliveries_load():
        FctDeliveriesLoader(dwh_pg_connect, log).load_fct_deliveries()

    couriers_dds_load() >> fct_deliveries_load()


sprint5_dds_delivery_dag = sprint5_dds_delivery_dag()
