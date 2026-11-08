from datetime import date, datetime, timedelta, timezone
from logging import Logger

from dds.dds_settings_repository import DdsEtlSettingsRepository, EtlSetting
from lib import PgConnect
from lib.dict_util import json2str
from psycopg import Connection


SETTLEMENT_REPORT_UPSERT_SQL = """
;WITH order_sums AS (
    SELECT
        r.restaurant_id           AS restaurant_id,
        r.restaurant_name         AS restaurant_name,
        tss.date                  AS settlement_date,
        COUNT(DISTINCT orders.id) AS orders_count,
        SUM(fct.total_sum)        AS orders_total_sum,
        SUM(fct.bonus_payment)    AS orders_bonus_payment_sum,
        SUM(fct.bonus_grant)      AS orders_bonus_granted_sum
    FROM dds.fct_product_sales AS fct
        INNER JOIN dds.dm_orders AS orders
            ON fct.order_id = orders.id
        INNER JOIN dds.dm_timestamps AS tss
            ON tss.id = orders.timestamp_id
        INNER JOIN dds.dm_restaurants AS r
            ON r.id = orders.restaurant_id
    WHERE orders.order_status = 'CLOSED'
      AND tss.date BETWEEN %(start_date)s AND %(end_date)s
    GROUP BY
        r.restaurant_id,
        r.restaurant_name,
        tss.date
)
INSERT INTO cdm.dm_settlement_report(
    restaurant_id,
    restaurant_name,
    settlement_date,
    orders_count,
    orders_total_sum,
    orders_bonus_payment_sum,
    orders_bonus_granted_sum,
    order_processing_fee,
    restaurant_reward_sum
)
SELECT
    restaurant_id,
    restaurant_name,
    settlement_date,
    orders_count,
    orders_total_sum,
    orders_bonus_payment_sum,
    orders_bonus_granted_sum,
    s.orders_total_sum * 0.25 AS order_processing_fee,
    s.orders_total_sum - s.orders_total_sum * 0.25 - s.orders_bonus_payment_sum AS restaurant_reward_sum
FROM order_sums AS s
ON CONFLICT (restaurant_id, settlement_date) DO UPDATE
SET
    orders_count = EXCLUDED.orders_count,
    orders_total_sum = EXCLUDED.orders_total_sum,
    orders_bonus_payment_sum = EXCLUDED.orders_bonus_payment_sum,
    orders_bonus_granted_sum = EXCLUDED.orders_bonus_granted_sum,
    order_processing_fee = EXCLUDED.order_processing_fee,
    restaurant_reward_sum = EXCLUDED.restaurant_reward_sum;
"""


class SettlementReportLoader:
    WF_KEY = "dds_to_cdm_settlement_report_workflow"
    LAST_LOADED_DATE_KEY = "last_loaded_date"
    DEFAULT_LAST_LOADED_DATE = "2022-01-01"
    OVERLAP_DAYS = 3

    def __init__(self, dwh: PgConnect, log: Logger) -> None:
        self.dwh = dwh
        self.settings_repository = DdsEtlSettingsRepository()
        self.log = log

    def _resolve_last_loaded_date(self, wf_setting: EtlSetting) -> date:
        raw = wf_setting.workflow_settings.get(
            self.LAST_LOADED_DATE_KEY, self.DEFAULT_LAST_LOADED_DATE
        )
        if isinstance(raw, date):
            return raw
        return date.fromisoformat(str(raw))

    def _upsert_settlement_report(
        self, conn: Connection, start_date: date, end_date: date
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                SETTLEMENT_REPORT_UPSERT_SQL,
                {
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )

    def load_settlement_report(self) -> None:
        with self.dwh.connection() as conn:
            wf_setting = self.settings_repository.get_setting(conn, self.WF_KEY)
            if not wf_setting:
                wf_setting = EtlSetting(
                    id=0,
                    workflow_key=self.WF_KEY,
                    workflow_settings={
                        self.LAST_LOADED_DATE_KEY: self.DEFAULT_LAST_LOADED_DATE
                    },
                )

            last_loaded = self._resolve_last_loaded_date(wf_setting)
            start_date = last_loaded - timedelta(days=self.OVERLAP_DAYS)
            end_date = datetime.now(timezone.utc).date()

            self.log.info(
                f"Loading cdm.dm_settlement_report for "
                f"start_date={start_date}, end_date={end_date} "
                f"(last_loaded_date={last_loaded})."
            )

            self._upsert_settlement_report(conn, start_date, end_date)

            wf_setting.workflow_settings[self.LAST_LOADED_DATE_KEY] = (
                end_date.isoformat()
            )
            wf_setting_json = json2str(wf_setting.workflow_settings)
            self.settings_repository.save_setting(
                conn, wf_setting.workflow_key, wf_setting_json
            )

            self.log.info(
                f"Settlement report load finished; last_loaded_date={end_date.isoformat()}"
            )
