import uuid
from typing import Any, Dict, List, Tuple

from lib.pg import PgConnect

_NS = uuid.UUID("8f14e45f-ceea-467a-9575-6e1d6e0c1a11")


def stable_uuid(value: str) -> uuid.UUID:
    return uuid.uuid5(_NS, value)


class CdmRepository:
    def __init__(self, db: PgConnect) -> None:
        self._db = db

    def refresh_user(self, user_id: str) -> None:
        user_uuid = stable_uuid(f"user|{user_id}")
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT hp.product_id,
                           sp.name,
                           COUNT(DISTINCT ho.h_order_pk)
                    FROM dds.h_user hu
                    JOIN dds.l_order_user lou ON lou.h_user_pk = hu.h_user_pk
                    JOIN dds.h_order ho ON ho.h_order_pk = lou.h_order_pk
                    JOIN dds.l_order_product lop ON lop.h_order_pk = ho.h_order_pk
                    JOIN dds.h_product hp ON hp.h_product_pk = lop.h_product_pk
                    JOIN dds.s_product_names sp ON sp.h_product_pk = hp.h_product_pk
                    WHERE hu.user_id = %(user_id)s
                      AND sp.load_dt = (
                          SELECT MAX(sp2.load_dt)
                          FROM dds.s_product_names sp2
                          WHERE sp2.h_product_pk = hp.h_product_pk
                      )
                    GROUP BY hp.product_id, sp.name
                    """,
                    {"user_id": user_id},
                )
                products: List[Tuple[Any, ...]] = cur.fetchall()

                cur.execute(
                    """
                    SELECT hc.category_name,
                           COUNT(DISTINCT ho.h_order_pk)
                    FROM dds.h_user hu
                    JOIN dds.l_order_user lou ON lou.h_user_pk = hu.h_user_pk
                    JOIN dds.h_order ho ON ho.h_order_pk = lou.h_order_pk
                    JOIN dds.l_order_product lop ON lop.h_order_pk = ho.h_order_pk
                    JOIN dds.l_product_category lpc ON lpc.h_product_pk = lop.h_product_pk
                    JOIN dds.h_category hc ON hc.h_category_pk = lpc.h_category_pk
                    WHERE hu.user_id = %(user_id)s
                    GROUP BY hc.category_name
                    """,
                    {"user_id": user_id},
                )
                categories: List[Tuple[Any, ...]] = cur.fetchall()

                cur.execute(
                    "DELETE FROM cdm.user_product_counters WHERE user_id = %(user_id)s",
                    {"user_id": user_uuid},
                )
                cur.execute(
                    "DELETE FROM cdm.user_category_counters WHERE user_id = %(user_id)s",
                    {"user_id": user_uuid},
                )
                for product_id, product_name, order_cnt in products:
                    cur.execute(
                        """
                        INSERT INTO cdm.user_product_counters
                            (user_id, product_id, product_name, order_cnt)
                        VALUES (%(user_id)s, %(product_id)s, %(product_name)s, %(order_cnt)s)
                        """,
                        {
                            "user_id": user_uuid,
                            "product_id": stable_uuid(f"product|{product_id}"),
                            "product_name": product_name,
                            "order_cnt": int(order_cnt),
                        },
                    )
                for category_name, order_cnt in categories:
                    cur.execute(
                        """
                        INSERT INTO cdm.user_category_counters
                            (user_id, category_id, category_name, order_cnt)
                        VALUES (%(user_id)s, %(category_id)s, %(category_name)s, %(order_cnt)s)
                        """,
                        {
                            "user_id": user_uuid,
                            "category_id": stable_uuid(f"category|{category_name}"),
                            "category_name": category_name,
                            "order_cnt": int(order_cnt),
                        },
                    )
