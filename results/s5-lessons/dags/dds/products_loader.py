from datetime import datetime
from logging import Logger
from typing import List, Optional

from dds.dds_settings_repository import DdsEtlSettingsRepository, EtlSetting
from lib import PgConnect
from lib.dict_util import json2str, str2json
from psycopg import Connection
from psycopg.rows import class_row
from pydantic import BaseModel

ACTIVE_TO_SENTINEL = datetime(2099, 12, 31)


class RestaurantStgObj(BaseModel):
    id: int
    object_id: str
    object_value: str
    update_ts: datetime


class ProductDdsObj(BaseModel):
    product_id: str
    product_name: str
    product_price: float
    active_from: datetime
    active_to: datetime
    restaurant_id: int


class RestaurantsStgRepository:
    def __init__(self, pg: PgConnect) -> None:
        self._db = pg

    def list_restaurants(self, threshold: int, limit: int) -> List[RestaurantStgObj]:
        with self._db.client().cursor(row_factory=class_row(RestaurantStgObj)) as cur:
            cur.execute(
                """
                    SELECT id, object_id, object_value, update_ts
                    FROM stg.ordersystem_restaurants
                    WHERE id > %(threshold)s
                    ORDER BY id ASC
                    LIMIT %(limit)s;
                """,
                {
                    "threshold": threshold,
                    "limit": limit,
                },
            )
            return cur.fetchall()


class ProductDdsRepository:
    def find_restaurant_surrogate_id(self, conn: Connection, restaurant_id: str) -> Optional[int]:
        with conn.cursor() as cur:
            cur.execute(
                """
                    SELECT id
                    FROM dds.dm_restaurants
                    WHERE restaurant_id = %(rid)s
                      AND date_trunc('day', active_to) = DATE '2099-12-31';
                """,
                {"rid": restaurant_id},
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]

    def insert_product(self, conn: Connection, product: ProductDdsObj) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO dds.dm_products(
                        product_id,
                        product_name,
                        product_price,
                        active_from,
                        active_to,
                        restaurant_id
                    )
                    VALUES (
                        %(product_id)s,
                        %(product_name)s,
                        %(product_price)s,
                        %(active_from)s,
                        %(active_to)s,
                        %(restaurant_id)s
                    );
                """,
                {
                    "product_id": product.product_id,
                    "product_name": product.product_name,
                    "product_price": product.product_price,
                    "active_from": product.active_from,
                    "active_to": product.active_to,
                    "restaurant_id": product.restaurant_id,
                },
            )


class ProductLoader:
    WF_KEY = "stg_to_dds_products_workflow"
    LAST_LOADED_ID_KEY = "last_loaded_id"
    BATCH_LIMIT = 100

    def __init__(self, dwh: PgConnect, log: Logger) -> None:
        self.dwh = dwh
        self.stg = RestaurantsStgRepository(dwh)
        self.dds = ProductDdsRepository()
        self.settings_repository = DdsEtlSettingsRepository()
        self.log = log

    def load_products(self) -> None:
        while True:
            with self.dwh.connection() as conn:
                wf_setting = self.settings_repository.get_setting(conn, self.WF_KEY)
                if not wf_setting:
                    wf_setting = EtlSetting(
                        id=0,
                        workflow_key=self.WF_KEY,
                        workflow_settings={self.LAST_LOADED_ID_KEY: -1},
                    )

                last_loaded = wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY]
                load_queue = self.stg.list_restaurants(last_loaded, self.BATCH_LIMIT)
                self.log.info(
                    f"Found {len(load_queue)} restaurants to scan for products "
                    f"(after id={last_loaded})."
                )
                if not load_queue:
                    self.log.info("No more restaurants to load products from. Quitting.")
                    return

                inserted = 0
                for row in load_queue:
                    data = str2json(row.object_value)
                    business_restaurant_id = str(data["_id"])
                    restaurant_surrogate_id = self.dds.find_restaurant_surrogate_id(
                        conn, business_restaurant_id
                    )
                    if restaurant_surrogate_id is None:
                        self.log.warning(
                            f"Skip restaurant stg.id={row.id}: no DDS row for "
                            f"restaurant_id={business_restaurant_id}"
                        )
                        continue

                    menu = data.get("menu") or []
                    for item in menu:
                        product = ProductDdsObj(
                            product_id=str(item["_id"]),
                            product_name=item["name"],
                            product_price=item["price"],
                            active_from=row.update_ts,
                            active_to=ACTIVE_TO_SENTINEL,
                            restaurant_id=restaurant_surrogate_id,
                        )
                        self.dds.insert_product(conn, product)
                        inserted += 1

                wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY] = max(
                    r.id for r in load_queue
                )
                wf_setting_json = json2str(wf_setting.workflow_settings)
                self.settings_repository.save_setting(
                    conn, wf_setting.workflow_key, wf_setting_json
                )

                self.log.info(
                    f"Batch finished: inserted={inserted}, last_loaded_id="
                    f"{wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY]}"
                )
