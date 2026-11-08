from datetime import datetime
from logging import Logger
from typing import List, Optional

from dds.dds_settings_repository import DdsEtlSettingsRepository, EtlSetting
from lib import PgConnect
from lib.dict_util import json2str, str2json
from psycopg import Connection
from psycopg.rows import class_row
from pydantic import BaseModel


class BonusEventStgObj(BaseModel):
    id: int
    event_ts: datetime
    event_type: str
    event_value: str


class FctProductSaleDdsObj(BaseModel):
    product_id: int
    order_id: int
    count: int
    price: float
    total_sum: float
    bonus_payment: float
    bonus_grant: float


class BonusEventsStgRepository:
    def __init__(self, pg: PgConnect) -> None:
        self._db = pg

    def list_bonus_transactions(self, threshold: int, limit: int) -> List[BonusEventStgObj]:
        with self._db.client().cursor(row_factory=class_row(BonusEventStgObj)) as cur:
            cur.execute(
                """
                    SELECT id, event_ts, event_type, event_value
                    FROM stg.bonussystem_events
                    WHERE id > %(threshold)s
                      AND event_type = 'bonus_transaction'
                    ORDER BY id ASC
                    LIMIT %(limit)s;
                """,
                {
                    "threshold": threshold,
                    "limit": limit,
                },
            )
            return cur.fetchall()


class FctProductSalesDdsRepository:
    def find_order_id(self, conn: Connection, order_key: str) -> Optional[int]:
        with conn.cursor() as cur:
            cur.execute(
                """
                    SELECT id
                    FROM dds.dm_orders
                    WHERE order_key = %(ok)s;
                """,
                {"ok": order_key},
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]

    def find_product_id(self, conn: Connection, product_id: str) -> Optional[int]:
        with conn.cursor() as cur:
            cur.execute(
                """
                    SELECT id
                    FROM dds.dm_products
                    WHERE product_id = %(pid)s
                    LIMIT 1;
                """,
                {"pid": product_id},
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]

    def insert_sale(self, conn: Connection, sale: FctProductSaleDdsObj) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO dds.fct_product_sales(
                        product_id,
                        order_id,
                        count,
                        price,
                        total_sum,
                        bonus_payment,
                        bonus_grant
                    )
                    VALUES (
                        %(product_id)s,
                        %(order_id)s,
                        %(count)s,
                        %(price)s,
                        %(total_sum)s,
                        %(bonus_payment)s,
                        %(bonus_grant)s
                    );
                """,
                {
                    "product_id": sale.product_id,
                    "order_id": sale.order_id,
                    "count": sale.count,
                    "price": sale.price,
                    "total_sum": sale.total_sum,
                    "bonus_payment": sale.bonus_payment,
                    "bonus_grant": sale.bonus_grant,
                },
            )


class FctSalesLoader:
    WF_KEY = "stg_to_dds_fct_sales_workflow"
    LAST_LOADED_ID_KEY = "last_loaded_id"
    BATCH_LIMIT = 20000

    def __init__(self, dwh: PgConnect, log: Logger) -> None:
        self.dwh = dwh
        self.stg = BonusEventsStgRepository(dwh)
        self.dds = FctProductSalesDdsRepository()
        self.settings_repository = DdsEtlSettingsRepository()
        self.log = log

    def _try_build_sales(
        self, conn: Connection, data: dict, stg_id: int
    ) -> List[FctProductSaleDdsObj]:
        order_key_raw = data.get("order_id")
        if order_key_raw is None:
            self.log.warning(f"Skip event stg.id={stg_id}: missing order_id in event_value")
            return []

        order_key = str(order_key_raw)
        order_surrogate_id = self.dds.find_order_id(conn, order_key)
        if order_surrogate_id is None:
            self.log.warning(
                f"Skip event stg.id={stg_id}: no order for order_key={order_key}"
            )
            return []

        product_payments = data.get("product_payments")
        if not product_payments:
            self.log.warning(
                f"Skip event stg.id={stg_id}: missing or empty product_payments"
            )
            return []

        sales: List[FctProductSaleDdsObj] = []
        for item in product_payments:
            product_business_id = item.get("product_id")
            if product_business_id is None:
                self.log.warning(
                    f"Skip line stg.id={stg_id}: missing product_id in product_payments"
                )
                continue

            pid = str(product_business_id)
            product_surrogate_id = self.dds.find_product_id(conn, pid)
            if product_surrogate_id is None:
                self.log.warning(
                    f"Skip line stg.id={stg_id}: no product for product_id={pid}"
                )
                continue

            sales.append(
                FctProductSaleDdsObj(
                    product_id=product_surrogate_id,
                    order_id=order_surrogate_id,
                    count=item["quantity"],
                    price=item["price"],
                    total_sum=item["product_cost"],
                    bonus_payment=item["bonus_payment"],
                    bonus_grant=item["bonus_grant"],
                )
            )

        return sales

    def load_fct_sales(self) -> None:
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
                load_queue = self.stg.list_bonus_transactions(
                    last_loaded, self.BATCH_LIMIT
                )
                self.log.info(
                    f"Found {len(load_queue)} bonus_transaction events to load into dds "
                    f"(after id={last_loaded})."
                )
                if not load_queue:
                    self.log.info("No more fct sales events to load. Quitting.")
                    return

                inserted = 0
                for row in load_queue:
                    data = str2json(row.event_value)
                    sales = self._try_build_sales(conn, data, row.id)
                    for sale in sales:
                        self.dds.insert_sale(conn, sale)
                        inserted += 1

                wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY] = max(
                    e.id for e in load_queue
                )
                wf_setting_json = json2str(wf_setting.workflow_settings)
                self.settings_repository.save_setting(
                    conn, wf_setting.workflow_key, wf_setting_json
                )

                self.log.info(
                    f"Batch finished: inserted={inserted}, last_loaded_id="
                    f"{wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY]}"
                )
