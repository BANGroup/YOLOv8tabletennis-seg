from datetime import datetime
from logging import Logger
from typing import List, Optional

from dds.dds_settings_repository import DdsEtlSettingsRepository, EtlSetting
from delivery.couriers_dds_loader import CourierDdsRepository
from lib import PgConnect
from lib.dict_util import json2str, str2json
from psycopg import Connection
from psycopg.rows import class_row
from pydantic import BaseModel

DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


class DeliveryStgObj(BaseModel):
    id: int
    object_id: str
    object_value: str


class DeliveriesStgRepository:
    def __init__(self, pg: PgConnect) -> None:
        self._db = pg

    def list_deliveries(self, threshold: int, limit: int) -> List[DeliveryStgObj]:
        with self._db.client().cursor(row_factory=class_row(DeliveryStgObj)) as cur:
            cur.execute(
                """
                    SELECT id, object_id, object_value
                    FROM stg.deliverysystem_deliveries
                    WHERE id > %(threshold)s
                    ORDER BY id ASC
                    LIMIT %(limit)s;
                """,
                {"threshold": threshold, "limit": limit},
            )
            return cur.fetchall()


class FctDeliveriesDdsRepository:
    def __init__(self) -> None:
        self.couriers = CourierDdsRepository()

    def find_order_pk(self, conn: Connection, order_key: str) -> Optional[int]:
        with conn.cursor() as cur:
            cur.execute(
                """
                    SELECT id FROM dds.dm_orders WHERE order_key = %(order_key)s;
                """,
                {"order_key": order_key},
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]

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
            if row is not None:
                return row[0]
            cur.execute(
                """
                    SELECT id
                    FROM dds.dm_timestamps
                    WHERE date_trunc('second', ts) = date_trunc('second', %(ts)s);
                """,
                {"ts": ts},
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]

    def upsert_delivery(
        self,
        conn: Connection,
        delivery_id: str,
        order_pk: Optional[int],
        courier_pk: int,
        timestamp_id: Optional[int],
        rate: int,
        tip_sum: float,
        delivery_ts: datetime,
        address: Optional[str],
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO dds.fct_deliveries(
                        delivery_id,
                        order_id,
                        courier_id,
                        timestamp_id,
                        rate,
                        tip_sum,
                        delivery_ts,
                        address
                    )
                    VALUES (
                        %(delivery_id)s,
                        %(order_id)s,
                        %(courier_id)s,
                        %(timestamp_id)s,
                        %(rate)s,
                        %(tip_sum)s,
                        %(delivery_ts)s,
                        %(address)s
                    )
                    ON CONFLICT (delivery_id) DO UPDATE
                    SET order_id = EXCLUDED.order_id,
                        courier_id = EXCLUDED.courier_id,
                        timestamp_id = EXCLUDED.timestamp_id,
                        rate = EXCLUDED.rate,
                        tip_sum = EXCLUDED.tip_sum,
                        delivery_ts = EXCLUDED.delivery_ts,
                        address = EXCLUDED.address;
                """,
                {
                    "delivery_id": delivery_id,
                    "order_id": order_pk,
                    "courier_id": courier_pk,
                    "timestamp_id": timestamp_id,
                    "rate": rate,
                    "tip_sum": tip_sum,
                    "delivery_ts": delivery_ts,
                    "address": address,
                },
            )

    def sync_orders_courier_id(self, conn: Connection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                    UPDATE dds.dm_orders AS o
                    SET courier_id = f.courier_id
                    FROM dds.fct_deliveries AS f
                    WHERE f.order_id = o.id
                      AND o.courier_id IS NULL;
                """
            )


class FctDeliveriesLoader:
    WF_KEY = "stg_to_dds_fct_deliveries_workflow"
    LAST_LOADED_ID_KEY = "last_loaded_id"
    BATCH_LIMIT = 500

    def __init__(self, dwh: PgConnect, log: Logger) -> None:
        self.dwh = dwh
        self.stg = DeliveriesStgRepository(dwh)
        self.dds = FctDeliveriesDdsRepository()
        self.settings_repository = DdsEtlSettingsRepository()
        self.log = log

    def _parse_ts(self, raw: str) -> datetime:
        normalized = raw.replace("T", " ")
        if "." in normalized:
            base, frac = normalized.split(".", 1)
            frac = frac[:6]
            normalized = f"{base}.{frac}"
            return datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S.%f")
        return datetime.strptime(normalized, DATETIME_FMT)

    def load_fct_deliveries(self) -> None:
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
                load_queue = self.stg.list_deliveries(last_loaded, self.BATCH_LIMIT)
                self.log.info(
                    f"Found {len(load_queue)} deliveries for DDS (after id={last_loaded})"
                )
                if not load_queue:
                    self.log.info("No more deliveries to load into dds.")
                    self.dds.sync_orders_courier_id(conn)
                    return

                loaded = 0
                skipped = 0
                successful_stg_ids: List[int] = []
                for row in load_queue:
                    data = str2json(row.object_value)
                    courier_key = str(data.get("courier_id", ""))
                    courier_pk = self.dds.couriers.find_courier_pk(conn, courier_key)
                    if courier_pk is None:
                        self.log.warning(
                            f"Skip delivery stg.id={row.id}: unknown courier {courier_key}"
                        )
                        skipped += 1
                        continue

                    order_key = str(data.get("order_id", ""))
                    order_pk = self.dds.find_order_pk(conn, order_key)

                    order_ts_raw = data.get("order_ts")
                    timestamp_id = None
                    if order_ts_raw:
                        order_ts = self._parse_ts(str(order_ts_raw))
                        timestamp_id = self.dds.find_timestamp_id(conn, order_ts)

                    delivery_ts = self._parse_ts(str(data["delivery_ts"]))
                    tip_sum = float(data.get("tip_sum") or 0)
                    rate = int(data.get("rate") or 0)
                    address = data.get("address")

                    self.dds.upsert_delivery(
                        conn,
                        str(data["delivery_id"]),
                        order_pk,
                        courier_pk,
                        timestamp_id,
                        rate,
                        tip_sum,
                        delivery_ts,
                        address,
                    )
                    loaded += 1
                    successful_stg_ids.append(row.id)

                self.dds.sync_orders_courier_id(conn)

                if loaded > 0:
                    wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY] = max(
                        successful_stg_ids
                    )
                    wf_setting_json = json2str(wf_setting.workflow_settings)
                    self.settings_repository.save_setting(
                        conn, wf_setting.workflow_key, wf_setting_json
                    )
                    self.log.info(
                        f"Batch finished: loaded={loaded}, skipped={skipped}, "
                        f"last_loaded_id="
                        f"{wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY]}"
                    )
                else:
                    self.log.warning(
                        f"Batch fully skipped: skipped={skipped} "
                        f"(reason: unknown courier); watermark unchanged "
                        f"last_loaded_id={last_loaded}"
                    )
                    return
