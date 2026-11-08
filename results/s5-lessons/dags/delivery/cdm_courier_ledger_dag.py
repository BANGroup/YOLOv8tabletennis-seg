import logging

import pendulum
from airflow.decorators import dag, task
from delivery.courier_ledger_loader import CourierLedgerLoader
from lib import ConnectionBuilder

log = logging.getLogger(__name__)


@dag(
    dag_id="sprint5_cdm_courier_ledger_dag",
    schedule_interval="0/15 * * * *",
    start_date=pendulum.datetime(2022, 5, 5, tz="UTC"),
    catchup=False,
    tags=["sprint5", "cdm", "courier", "ledger"],
    is_paused_upon_creation=False,
)
def sprint5_cdm_courier_ledger_dag():
    dwh_pg_connect = ConnectionBuilder.pg_conn("PG_WAREHOUSE_CONNECTION")

    @task(task_id="courier_ledger_load")
    def courier_ledger_load():
        CourierLedgerLoader(dwh_pg_connect, log).load_courier_ledger()

    courier_ledger_load()


sprint5_cdm_courier_ledger_dag = sprint5_cdm_courier_ledger_dag()
