import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List

import requests
from airflow.models import Variable
from lib import PgConnect
from lib.dict_util import json2str
from psycopg import Connection

PAGE_LIMIT = 50
LOOKBACK_DAYS = 7
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


class DeliveryApiStgLoader:
    def __init__(self, dwh: PgConnect, log: logging.Logger) -> None:
        self.dwh = dwh
        self.log = log

    def _api_host(self) -> str:
        return Variable.get("DELIVERY_API_HOST").rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "X-API-KEY": Variable.get("DELIVERY_API_KEY"),
            "X-Nickname": Variable.get("DELIVERY_API_NICKNAME"),
            "X-Cohort": Variable.get("DELIVERY_API_COHORT"),
        }

    def _fetch_pages(
        self,
        path: str,
        extra_params: Dict[str, Any],
        sort_field: str,
    ) -> Iterable[List[dict]]:
        offset = 0
        while True:
            params = {
                **extra_params,
                "limit": PAGE_LIMIT,
                "offset": offset,
                "sort_field": sort_field,
                "sort_direction": "asc",
            }
            url = f"{self._api_host()}{path}"
            response = requests.get(
                url, headers=self._headers(), params=params, timeout=120
            )
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list):
                raise RuntimeError(f"Unexpected API payload from {url}: {page!r}")
            if not page:
                break
            self.log.info(f"Fetched {len(page)} rows from {path} offset={offset}")
            yield page
            if len(page) < PAGE_LIMIT:
                break
            offset += PAGE_LIMIT

    def _upsert_courier(self, conn: Connection, item: dict) -> None:
        object_id = str(item["_id"])
        object_value = json2str(item)
        with conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO stg.deliverysystem_couriers(object_id, object_value)
                    VALUES (%(object_id)s, %(object_value)s)
                    ON CONFLICT (object_id) DO UPDATE
                    SET object_value = EXCLUDED.object_value,
                        load_ts = now();
                """,
                {"object_id": object_id, "object_value": object_value},
            )

    def _parse_ts(self, raw: str) -> datetime:
        if raw is None:
            raise ValueError("timestamp is null")
        normalized = raw.replace("T", " ")
        if "." in normalized:
            base, frac = normalized.split(".", 1)
            frac = frac[:6]
            normalized = f"{base}.{frac}"
            return datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S.%f")
        return datetime.strptime(normalized, DATETIME_FMT)

    def _upsert_delivery(self, conn: Connection, item: dict) -> None:
        object_id = str(item["delivery_id"])
        order_id = str(item["order_id"])
        order_ts = self._parse_ts(item["order_ts"])
        object_value = json2str(item)
        with conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO stg.deliverysystem_deliveries(
                        object_id, object_value, order_id, order_ts
                    )
                    VALUES (
                        %(object_id)s,
                        %(object_value)s,
                        %(order_id)s,
                        %(order_ts)s
                    )
                    ON CONFLICT (object_id) DO UPDATE
                    SET object_value = EXCLUDED.object_value,
                        order_id = EXCLUDED.order_id,
                        order_ts = EXCLUDED.order_ts,
                        load_ts = now();
                """,
                {
                    "object_id": object_id,
                    "object_value": object_value,
                    "order_id": order_id,
                    "order_ts": order_ts,
                },
            )

    def load_couriers(self) -> None:
        total = 0
        with self.dwh.connection() as conn:
            for page in self._fetch_pages("/couriers", {}, sort_field="_id"):
                for item in page:
                    self._upsert_courier(conn, item)
                    total += 1
        self.log.info(f"STG couriers upserted: {total}")

    def load_deliveries(self) -> None:
        total = self._load_deliveries_for_lookback(LOOKBACK_DAYS)
        self.log.info(
            f"STG deliveries upserted: {total} (lookback={LOOKBACK_DAYS} days)"
        )

    def _load_deliveries_for_lookback(self, lookback_days: int) -> int:
        now = datetime.utcnow()
        from_ts = (now - timedelta(days=lookback_days)).strftime(DATETIME_FMT)
        to_ts = now.strftime(DATETIME_FMT)
        total = 0
        with self.dwh.connection() as conn:
            for page in self._fetch_pages(
                "/deliveries",
                {"from": from_ts, "to": to_ts},
                sort_field="_id",
            ):
                for item in page:
                    self._upsert_delivery(conn, item)
                    total += 1
        return total
