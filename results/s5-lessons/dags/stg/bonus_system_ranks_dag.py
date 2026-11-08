import logging

import pendulum
from airflow.decorators import dag, task
from lib import ConnectionBuilder
from stg.events_loader import EventLoader
from stg.ranks_loader import RankLoader
from stg.users_loader import UserLoader

log = logging.getLogger(__name__)


@dag(
    schedule_interval='0/15 * * * *',
    start_date=pendulum.datetime(2022, 5, 5, tz="UTC"),
    catchup=False,
    tags=['sprint5', 'stg', 'origin', 'ranks', 'users', 'events'],
    is_paused_upon_creation=False
)
def sprint5_stg_bonus_system_ranks_dag():
    dwh_pg_connect = ConnectionBuilder.pg_conn("PG_WAREHOUSE_CONNECTION")
    origin_pg_connect = ConnectionBuilder.pg_conn("PG_ORIGIN_BONUS_SYSTEM_CONNECTION")

    @task(task_id="ranks_load")
    def load_ranks():
        log.info("Starting full ranks load from origin to stg.bonussystem_ranks")
        RankLoader(origin_pg_connect, dwh_pg_connect).load_ranks()
        log.info("Ranks load finished")

    @task(task_id="users_load")
    def load_users():
        log.info("Starting full users load from origin to stg.bonussystem_users")
        UserLoader(origin_pg_connect, dwh_pg_connect).load_users()
        log.info("Users load finished")

    @task(task_id="events_load")
    def load_events():
        log.info("Starting incremental events load from origin outbox to stg.bonussystem_events")
        EventLoader(origin_pg_connect, dwh_pg_connect, log).load_events()
        log.info("Events load finished")

    load_ranks()
    load_users()
    load_events()


sprint5_stg_bonus_system_ranks_dag = sprint5_stg_bonus_system_ranks_dag()
