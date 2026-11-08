from datetime import date, datetime, time
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
YEAR_MIN = 2022
YEAR_MAX = 2499


class OrderStgObj(BaseModel):
    id: int
    object_id: str
    object_value: str
    update_ts: datetime


class TimestampDdsObj(BaseModel):
    ts: datetime
    year: int
    month: int
    day: int
    time: time
    date: date


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


class TimestampDdsRepository:
    def insert_timestamp(self, conn: Connection, row: TimestampDdsObj) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO dds.dm_timestamps(ts, year, month, day, time, date)
                    VALUES (%(ts)s, %(year)s, %(month)s, %(day)s, %(time)s, %(date)s);
                """,
                {
                    "ts": row.ts,
                    "year": row.year,
                    "month": row.month,
                    "day": row.day,
                    "time": row.time,
                    "date": row.date,
                },
            )


class TimestampLoader:
    WF_KEY = "stg_to_dds_timestamps_workflow"
    LAST_LOADED_ID_KEY = "last_loaded_id"
    BATCH_LIMIT = 1000

    def __init__(self, dwh: PgConnect, log: Logger) -> None:
        self.dwh = dwh
        self.stg = OrdersStgRepository(dwh)
        self.dds = TimestampDdsRepository()
        self.settings_repository = DdsEtlSettingsRepository()
        self.log = log

    def _build_dds_row(self, data: dict) -> Optional[TimestampDdsObj]:
        if data.get("final_status") not in FINAL_STATUSES:
            return None
        date_str = data.get("date")
        if not date_str:
            return None
        ts = datetime.strptime(date_str, DATE_FMT)
        if ts.year < YEAR_MIN or ts.year > YEAR_MAX:
            return None
        return TimestampDdsObj(
            ts=ts,
            year=ts.year,
            month=ts.month,
            day=ts.day,
            time=ts.time(),
            date=ts.date(),
        )

    def load_timestamps(self) -> None:
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
                    f"Found {len(load_queue)} orders to scan for timestamps "
                    f"(after id={last_loaded})."
                )
                if not load_queue:
                    self.log.info("No more orders to load. Quitting.")
                    return

                inserted = 0
                for row in load_queue:
                    data = str2json(row.object_value)
                    dds_row = self._build_dds_row(data)
                    if dds_row is not None:
                        self.dds.insert_timestamp(conn, dds_row)
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
