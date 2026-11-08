import logging

import pendulum
from airflow.decorators import dag, task
from dds.fct_sales_loader import FctSalesLoader
from dds.orders_loader import OrderLoader
from dds.products_loader import ProductLoader
from dds.restaurants_loader import RestaurantLoader
from dds.timestamps_loader import TimestampLoader
from dds.users_loader import UserLoader
from lib import ConnectionBuilder

log = logging.getLogger(__name__)


@dag(
    dag_id="sprint5_stg_to_dds_dag",
    schedule_interval="0/15 * * * *",
    start_date=pendulum.datetime(2022, 5, 5, tz="UTC"),
    catchup=False,
    tags=["sprint5", "stg", "dds", "users", "restaurants", "timestamps", "products", "orders", "fct"],
    is_paused_upon_creation=False,
)
def sprint5_stg_to_dds_dag():
    dwh_pg_connect = ConnectionBuilder.pg_conn("PG_WAREHOUSE_CONNECTION")

    @task(task_id="users_load")
    def users_load():
        UserLoader(dwh_pg_connect, log).load_users()

    @task(task_id="restaurants_load")
    def restaurants_load():
        RestaurantLoader(dwh_pg_connect, log).load_restaurants()

    @task(task_id="timestamps_load")
    def timestamps_load():
        TimestampLoader(dwh_pg_connect, log).load_timestamps()

    @task(task_id="products_load")
    def products_load():
        ProductLoader(dwh_pg_connect, log).load_products()

    @task(task_id="orders_load")
    def orders_load():
        OrderLoader(dwh_pg_connect, log).load_orders()

    @task(task_id="fct_sales_load")
    def fct_sales_load():
        FctSalesLoader(dwh_pg_connect, log).load_fct_sales()

    users = users_load()
    timestamps = timestamps_load()
    restaurants = restaurants_load()
    products = products_load()
    orders = orders_load()

    restaurants >> products
    users >> orders
    restaurants >> orders
    timestamps >> orders

    fct_sales = fct_sales_load()
    orders >> fct_sales
    products >> fct_sales


sprint5_stg_to_dds_dag = sprint5_stg_to_dds_dag()
