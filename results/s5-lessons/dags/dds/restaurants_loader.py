from datetime import datetime
from logging import Logger
from typing import List

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


class RestaurantDdsObj(BaseModel):
    restaurant_id: str
    restaurant_name: str
    active_from: datetime
    active_to: datetime


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


class RestaurantDdsRepository:
    def insert_restaurant(self, conn: Connection, restaurant: RestaurantDdsObj) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO dds.dm_restaurants(
                        restaurant_id, restaurant_name, active_from, active_to
                    )
                    VALUES (
                        %(restaurant_id)s,
                        %(restaurant_name)s,
                        %(active_from)s,
                        %(active_to)s
                    );
                """,
                {
                    "restaurant_id": restaurant.restaurant_id,
                    "restaurant_name": restaurant.restaurant_name,
                    "active_from": restaurant.active_from,
                    "active_to": restaurant.active_to,
                },
            )


class RestaurantLoader:
    WF_KEY = "stg_to_dds_restaurants_workflow"
    LAST_LOADED_ID_KEY = "last_loaded_id"
    BATCH_LIMIT = 100

    def __init__(self, dwh: PgConnect, log: Logger) -> None:
        self.dwh = dwh
        self.stg = RestaurantsStgRepository(dwh)
        self.dds = RestaurantDdsRepository()
        self.settings_repository = DdsEtlSettingsRepository()
        self.log = log

    def load_restaurants(self) -> None:
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
                    f"Found {len(load_queue)} restaurants to load into dds "
                    f"(after id={last_loaded})."
                )
                if not load_queue:
                    self.log.info("No more restaurants to load. Quitting.")
                    return

                for row in load_queue:
                    data = str2json(row.object_value)
                    restaurant = RestaurantDdsObj(
                        restaurant_id=str(data["_id"]),
                        restaurant_name=data["name"],
                        active_from=row.update_ts,
                        active_to=ACTIVE_TO_SENTINEL,
                    )
                    self.dds.insert_restaurant(conn, restaurant)

                wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY] = max(
                    r.id for r in load_queue
                )
                wf_setting_json = json2str(wf_setting.workflow_settings)
                self.settings_repository.save_setting(
                    conn, wf_setting.workflow_key, wf_setting_json
                )

                self.log.info(
                    f"Batch finished at last_loaded_id="
                    f"{wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY]}"
                )
