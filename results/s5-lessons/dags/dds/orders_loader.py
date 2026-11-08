from datetime import datetime
from logging import Logger
from typing import List, Optional

from dds.dds_settings_repository import DdsEtlSettingsRepository, EtlSetting
from lib import PgConnect
from lib.dict_util import json2str, str2json
from psycopg import Connection
from psycopg.rows import class_row
from pydantic import BaseModel

FINAL_STATUSES = ("CLOSED", "CANCELLED")
DATE_FMT = "%Y-%m-%d %H:%M:%S"


class OrderStgObj(BaseModel):
    id: int
    object_id: str
    object_value: str
    update_ts: datetime


class OrderDdsObj(BaseModel):
    order_key: str
    order_status: str
    restaurant_id: int
    timestamp_id: int
    user_id: int


class OrdersStgRepository:
    def __init__(self, pg: PgConnect) -> None:
        self._db = pg

    def list_orders(self, threshold: int, limit: int) -> List[OrderStgObj]:
        with self._db.client().cursor(row_factory=class_row(OrderStgObj)) as cur:
            cur.execute(
                """
                    SELECT id, object_id, object_value, update_ts
                    FROM stg.ordersystem_orders
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


class OrderDdsRepository:
    def find_timestamp_id(self, conn: Connection, ts: datetime) -> Optional[int]:
        with conn.cursor() as cur:
            cur.execute(
                """
                    SELECT id
                    FROM dds.dm_timestamps
                    WHERE ts = %(ts)s;
                """,
                {"ts": ts},
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]

    def find_user_id(self, conn: Connection, uid: str) -> Optional[int]:
        with conn.cursor() as cur:
            cur.execute(
                """
                    SELECT id
                    FROM dds.dm_users
                    WHERE user_id = %(uid)s;
                """,
                {"uid": uid},
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]

    def find_restaurant_id(self, conn: Connection, rid: str, ts: datetime) -> Optional[int]:
        with conn.cursor() as cur:
            cur.execute(
                """
                    SELECT id
                    FROM dds.dm_restaurants
                    WHERE restaurant_id = %(rid)s
                      AND %(ts)s >= active_from
                      AND %(ts)s <= active_to;
                """,
                {"rid": rid, "ts": ts},
            )
            row = cur.fetchone()
            if row is not None:
                return row[0]
            cur.execute(
                """
                    SELECT id
                    FROM dds.dm_restaurants
                    WHERE restaurant_id = %(rid)s
                      AND date_trunc('day', active_to) = DATE '2099-12-31';
                """,
                {"rid": rid},
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]

    def insert_order(self, conn: Connection, order: OrderDdsObj) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO dds.dm_orders(
                        order_key,
                        order_status,
                        restaurant_id,
                        timestamp_id,
                        user_id
                    )
                    VALUES (
                        %(order_key)s,
                        %(order_status)s,
                        %(restaurant_id)s,
                        %(timestamp_id)s,
                        %(user_id)s
                    );
                """,
                {
                    "order_key": order.order_key,
                    "order_status": order.order_status,
                    "restaurant_id": order.restaurant_id,
                    "timestamp_id": order.timestamp_id,
                    "user_id": order.user_id,
                },
            )


class OrderLoader:
    WF_KEY = "stg_to_dds_orders_workflow"
    LAST_LOADED_ID_KEY = "last_loaded_id"
    BATCH_LIMIT = 1000

    def __init__(self, dwh: PgConnect, log: Logger) -> None:
        self.dwh = dwh
        self.stg = OrdersStgRepository(dwh)
        self.dds = OrderDdsRepository()
        self.settings_repository = DdsEtlSettingsRepository()
        self.log = log

    def _try_build_order(
        self, conn: Connection, data: dict, stg_id: int
    ) -> Optional[OrderDdsObj]:
        if data.get("final_status") not in FINAL_STATUSES:
            return None

        date_str = data.get("date")
        if not date_str:
            self.log.warning(f"Skip order stg.id={stg_id}: missing date")
            return None

        order_ts = datetime.strptime(date_str, DATE_FMT)
        restaurant = data.get("restaurant") or {}
        user = data.get("user") or {}
        restaurant_mongo_id = restaurant.get("id")
        user_mongo_id = user.get("id")
        if restaurant_mongo_id is None or user_mongo_id is None:
            self.log.warning(
                f"Skip order stg.id={stg_id}: missing restaurant.id or user.id"
            )
            return None

        rid = str(restaurant_mongo_id)
        uid = str(user_mongo_id)

        timestamp_id = self.dds.find_timestamp_id(conn, order_ts)
        if timestamp_id is None:
            self.log.warning(
                f"Skip order stg.id={stg_id}: no timestamp for ts={order_ts}"
            )
            return None

        user_id = self.dds.find_user_id(conn, uid)
        if user_id is None:
            self.log.warning(
                f"Skip order stg.id={stg_id}: no user for user_id={uid}"
            )
            return None

        restaurant_id = self.dds.find_restaurant_id(conn, rid, order_ts)
        if restaurant_id is None:
            self.log.warning(
                f"Skip order stg.id={stg_id}: no restaurant for restaurant_id={rid}"
            )
            return None

        return OrderDdsObj(
            order_key=str(data["_id"]),
            order_status=data["final_status"],
            restaurant_id=restaurant_id,
            timestamp_id=timestamp_id,
            user_id=user_id,
        )

    def load_orders(self) -> None:
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
                load_queue = self.stg.list_orders(last_loaded, self.BATCH_LIMIT)
                self.log.info(
                    f"Found {len(load_queue)} orders to load into dds "
                    f"(after id={last_loaded})."
                )
                if not load_queue:
                    self.log.info("No more orders to load. Quitting.")
                    return

                inserted = 0
                for row in load_queue:
                    data = str2json(row.object_value)
                    order = self._try_build_order(conn, data, row.id)
                    if order is not None:
                        self.dds.insert_order(conn, order)
                        inserted += 1

                wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY] = max(
                    o.id for o in load_queue
                )
                wf_setting_json = json2str(wf_setting.workflow_settings)
                self.settings_repository.save_setting(
                    conn, wf_setting.workflow_key, wf_setting_json
                )

                self.log.info(
                    f"Batch finished: inserted={inserted}, last_loaded_id="
                    f"{wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY]}"
                )
