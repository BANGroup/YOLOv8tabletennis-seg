from datetime import date, datetime, timedelta, timezone
from logging import Logger

from dds.dds_settings_repository import DdsEtlSettingsRepository, EtlSetting
from lib import PgConnect
from lib.dict_util import json2str
from psycopg import Connection

COURIER_LEDGER_UPSERT_SQL = """
;WITH delivery_orders AS (
    SELECT
        c.courier_id,
        c.courier_name,
        t.year AS settlement_year,
        t.month AS settlement_month,
        o.id AS order_id,
        fd.tip_sum,
        fd.rate::numeric AS rate
    FROM dds.fct_deliveries AS fd
        INNER JOIN dds.dm_couriers AS c ON c.id = fd.courier_id
        INNER JOIN dds.dm_orders AS o ON o.id = fd.order_id
        INNER JOIN dds.dm_timestamps AS t ON t.id = o.timestamp_id
    WHERE o.order_status = 'CLOSED'
      AND t.date BETWEEN %(start_date)s AND %(end_date)s
),
order_sums AS (
    SELECT
        d.courier_id,
        d.courier_name,
        d.settlement_year,
        d.settlement_month,
        d.order_id,
        MAX(d.tip_sum) AS tip_sum,
        SUM(fps.total_sum) AS order_sum
    FROM delivery_orders AS d
        INNER JOIN dds.fct_product_sales AS fps ON fps.order_id = d.order_id
    GROUP BY
        d.courier_id,
        d.courier_name,
        d.settlement_year,
        d.settlement_month,
        d.order_id
),
courier_month_rates AS (
    SELECT
        courier_id,
        courier_name,
        settlement_year,
        settlement_month,
        AVG(rate) AS rate_avg
    FROM delivery_orders
    GROUP BY courier_id, courier_name, settlement_year, settlement_month
),
order_payouts AS (
    SELECT
        os.courier_id,
        os.courier_name,
        os.settlement_year,
        os.settlement_month,
        os.order_id,
        os.order_sum,
        os.tip_sum,
        cmr.rate_avg,
        CASE
            WHEN cmr.rate_avg < 4 THEN GREATEST(os.order_sum * 0.05, 100)
            WHEN cmr.rate_avg < 4.5 THEN GREATEST(os.order_sum * 0.07, 150)
            WHEN cmr.rate_avg < 4.9 THEN GREATEST(os.order_sum * 0.08, 175)
            ELSE GREATEST(os.order_sum * 0.10, 200)
        END AS order_payout
    FROM order_sums AS os
        INNER JOIN courier_month_rates AS cmr
            ON cmr.courier_id = os.courier_id
           AND cmr.settlement_year = os.settlement_year
           AND cmr.settlement_month = os.settlement_month
),
aggregated AS (
    SELECT
        courier_id,
        courier_name,
        settlement_year,
        settlement_month,
        COUNT(DISTINCT order_id) AS orders_count,
        SUM(order_sum) AS orders_total_sum,
        MAX(rate_avg) AS rate_avg,
        SUM(order_payout) AS courier_order_sum,
        SUM(tip_sum) AS courier_tips_sum
    FROM order_payouts
    GROUP BY courier_id, courier_name, settlement_year, settlement_month
)
INSERT INTO cdm.dm_courier_ledger(
    courier_id,
    courier_name,
    settlement_year,
    settlement_month,
    orders_count,
    orders_total_sum,
    rate_avg,
    order_processing_fee,
    courier_order_sum,
    courier_tips_sum,
    courier_reward_sum
)
SELECT
    courier_id,
    courier_name,
    settlement_year,
    settlement_month,
    orders_count,
    orders_total_sum,
    rate_avg,
    orders_total_sum * 0.25 AS order_processing_fee,
    courier_order_sum,
    courier_tips_sum,
    courier_order_sum + courier_tips_sum * 0.95 AS courier_reward_sum
FROM aggregated
ON CONFLICT (courier_id, settlement_year, settlement_month) DO UPDATE
SET
    courier_name = EXCLUDED.courier_name,
    orders_count = EXCLUDED.orders_count,
    orders_total_sum = EXCLUDED.orders_total_sum,
    rate_avg = EXCLUDED.rate_avg,
    order_processing_fee = EXCLUDED.order_processing_fee,
    courier_order_sum = EXCLUDED.courier_order_sum,
    courier_tips_sum = EXCLUDED.courier_tips_sum,
    courier_reward_sum = EXCLUDED.courier_reward_sum;
"""


class CourierLedgerLoader:
    WF_KEY = "dds_to_cdm_courier_ledger_workflow"
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

    def _upsert_ledger(
        self, conn: Connection, start_date: date, end_date: date
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                COURIER_LEDGER_UPSERT_SQL,
                {"start_date": start_date, "end_date": end_date},
            )

    def load_courier_ledger(self) -> None:
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
                f"Loading cdm.dm_courier_ledger for "
                f"start_date={start_date}, end_date={end_date} "
                f"(last_loaded_date={last_loaded})."
            )

            self._upsert_ledger(conn, start_date, end_date)

            wf_setting.workflow_settings[self.LAST_LOADED_DATE_KEY] = (
                end_date.isoformat()
            )
            wf_setting_json = json2str(wf_setting.workflow_settings)
            self.settings_repository.save_setting(
                conn, wf_setting.workflow_key, wf_setting_json
            )

            self.log.info(
                f"Courier ledger load finished; last_loaded_date={end_date.isoformat()}"
            )
