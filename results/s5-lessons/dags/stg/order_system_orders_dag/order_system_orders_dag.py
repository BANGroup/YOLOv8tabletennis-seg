import logging

import pendulum
from airflow.decorators import dag, task
from airflow.models.variable import Variable
from lib import ConnectionBuilder, MongoConnect
from stg.order_system_orders_dag.order_loader import OrderLoader
from stg.order_system_orders_dag.order_reader import OrderReader
from stg.order_system_orders_dag.pg_saver import PgSaver

log = logging.getLogger(__name__)


@dag(
    dag_id="sprint5_stg_order_system_orders",
    schedule_interval="0/15 * * * *",
    start_date=pendulum.datetime(2022, 5, 5, tz="UTC"),
    catchup=False,
    tags=["sprint5", "stg", "origin", "orders", "mongo"],
    is_paused_upon_creation=False,
)
def sprint5_stg_order_system_orders():
    dwh_pg_connect = ConnectionBuilder.pg_conn("PG_WAREHOUSE_CONNECTION")

    cert_path = Variable.get("MONGO_DB_CERTIFICATE_PATH")
    db_user = Variable.get("MONGO_DB_USER")
    db_pw = Variable.get("MONGO_DB_PASSWORD")
    rs = Variable.get("MONGO_DB_REPLICA_SET")
    db = Variable.get("MONGO_DB_DATABASE_NAME")
    host = Variable.get("MONGO_DB_HOST")

    @task()
    def load_orders():
        pg_saver = PgSaver()
        mongo_connect = MongoConnect(cert_path, db_user, db_pw, host, rs, db, db)
        collection_reader = OrderReader(mongo_connect)
        loader = OrderLoader(collection_reader, dwh_pg_connect, pg_saver, log)
        loader.run_copy()

    load_orders()


sprint5_stg_order_system_orders_dag = sprint5_stg_order_system_orders()
