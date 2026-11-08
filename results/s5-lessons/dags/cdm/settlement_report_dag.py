import logging

import pendulum
from airflow.decorators import dag, task
from cdm.settlement_report_loader import SettlementReportLoader
from lib import ConnectionBuilder

log = logging.getLogger(__name__)


@dag(
    dag_id="sprint5_cdm_settlement_report_dag",
    schedule_interval="0/15 * * * *",
    start_date=pendulum.datetime(2022, 5, 5, tz="UTC"),
    catchup=False,
    tags=["sprint5", "cdm", "settlement"],
    is_paused_upon_creation=False,
)
def sprint5_cdm_settlement_report_dag():
    dwh_pg_connect = ConnectionBuilder.pg_conn("PG_WAREHOUSE_CONNECTION")

    @task(task_id="settlement_report_load")
    def settlement_report_load():
        SettlementReportLoader(dwh_pg_connect, log).load_settlement_report()

    settlement_report_load()


sprint5_cdm_settlement_report_dag = sprint5_cdm_settlement_report_dag()
