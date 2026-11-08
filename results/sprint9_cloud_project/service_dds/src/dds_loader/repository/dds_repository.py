import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List

from lib.pg import PgConnect

_NS = uuid.UUID("8f14e45f-ceea-467a-9575-6e1d6e0c1a11")
LOAD_SRC = "orders-system-kafka"


def stable_uuid(value: str) -> uuid.UUID:
    return uuid.uuid5(_NS, value)


class DdsRepository:
    def __init__(self, db: PgConnect) -> None:
        self._db = db

    def save_order(
        self,
        order_id: int,
        order_dt: datetime,
        cost: Decimal,
        payment: Decimal,
        status: str,
        user_id: str,
        username: str,
        userlogin: str,
        restaurant_id: str,
        restaurant_name: str,
        products: List[Dict[str, Any]],
    ) -> None:
        load_dt = order_dt
        h_order_pk = stable_uuid(f"order|{order_id}")
        h_user_pk = stable_uuid(f"user|{user_id}")
        h_restaurant_pk = stable_uuid(f"restaurant|{restaurant_id}")

        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO dds.h_order (h_order_pk, order_id, order_dt, load_dt, load_src)
                    VALUES (%(pk)s, %(order_id)s, %(order_dt)s, %(load_dt)s, %(src)s)
                    ON CONFLICT (h_order_pk) DO NOTHING
                    """,
                    {
                        "pk": h_order_pk,
                        "order_id": order_id,
                        "order_dt": order_dt,
                        "load_dt": load_dt,
                        "src": LOAD_SRC,
                    },
                )
                cur.execute(
                    """
                    INSERT INTO dds.h_user (h_user_pk, user_id, load_dt, load_src)
                    VALUES (%(pk)s, %(user_id)s, %(load_dt)s, %(src)s)
                    ON CONFLICT (h_user_pk) DO NOTHING
                    """,
                    {"pk": h_user_pk, "user_id": user_id, "load_dt": load_dt, "src": LOAD_SRC},
                )
                cur.execute(
                    """
                    INSERT INTO dds.h_restaurant (h_restaurant_pk, restaurant_id, load_dt, load_src)
                    VALUES (%(pk)s, %(restaurant_id)s, %(load_dt)s, %(src)s)
                    ON CONFLICT (h_restaurant_pk) DO NOTHING
                    """,
                    {
                        "pk": h_restaurant_pk,
                        "restaurant_id": restaurant_id,
                        "load_dt": load_dt,
                        "src": LOAD_SRC,
                    },
                )
                cur.execute(
                    """
                    INSERT INTO dds.s_user_names
                        (h_user_pk, username, userlogin, load_dt, load_src, hk_user_names_hashdiff)
                    VALUES (%(pk)s, %(username)s, %(userlogin)s, %(load_dt)s, %(src)s, %(hashdiff)s)
                    ON CONFLICT (h_user_pk, load_dt) DO NOTHING
                    """,
                    {
                        "pk": h_user_pk,
                        "username": username,
                        "userlogin": userlogin,
                        "load_dt": load_dt,
                        "src": LOAD_SRC,
                        "hashdiff": stable_uuid(f"user_names|{username}|{userlogin}"),
                    },
                )
                cur.execute(
                    """
                    INSERT INTO dds.s_restaurant_names
                        (h_restaurant_pk, name, load_dt, load_src, hk_restaurant_names_hashdiff)
                    VALUES (%(pk)s, %(name)s, %(load_dt)s, %(src)s, %(hashdiff)s)
                    ON CONFLICT (h_restaurant_pk, load_dt) DO NOTHING
                    """,
                    {
                        "pk": h_restaurant_pk,
                        "name": restaurant_name,
                        "load_dt": load_dt,
                        "src": LOAD_SRC,
                        "hashdiff": stable_uuid(f"restaurant_name|{restaurant_name}"),
                    },
                )
                cur.execute(
                    """
                    INSERT INTO dds.s_order_cost
                        (h_order_pk, cost, payment, load_dt, load_src, hk_order_cost_hashdiff)
                    VALUES (%(pk)s, %(cost)s, %(payment)s, %(load_dt)s, %(src)s, %(hashdiff)s)
                    ON CONFLICT (h_order_pk, load_dt) DO NOTHING
                    """,
                    {
                        "pk": h_order_pk,
                        "cost": cost,
                        "payment": payment,
                        "load_dt": load_dt,
                        "src": LOAD_SRC,
                        "hashdiff": stable_uuid(f"order_cost|{cost}|{payment}"),
                    },
                )
                cur.execute(
                    """
                    INSERT INTO dds.s_order_status
                        (h_order_pk, status, load_dt, load_src, hk_order_status_hashdiff)
                    VALUES (%(pk)s, %(status)s, %(load_dt)s, %(src)s, %(hashdiff)s)
                    ON CONFLICT (h_order_pk, load_dt) DO NOTHING
                    """,
                    {
                        "pk": h_order_pk,
                        "status": status,
                        "load_dt": load_dt,
                        "src": LOAD_SRC,
                        "hashdiff": stable_uuid(f"order_status|{status}"),
                    },
                )
                cur.execute(
                    """
                    INSERT INTO dds.l_order_user
                        (hk_order_user_pk, h_order_pk, h_user_pk, load_dt, load_src)
                    VALUES (%(pk)s, %(h_order_pk)s, %(h_user_pk)s, %(load_dt)s, %(src)s)
                    ON CONFLICT (hk_order_user_pk) DO NOTHING
                    """,
                    {
                        "pk": stable_uuid(f"l_order_user|{h_order_pk}|{h_user_pk}"),
                        "h_order_pk": h_order_pk,
                        "h_user_pk": h_user_pk,
                        "load_dt": load_dt,
                        "src": LOAD_SRC,
                    },
                )

                for product in products:
                    product_id = str(product["id"])
                    product_name = str(product["name"])
                    category_name = str(product["category"])
                    h_product_pk = stable_uuid(f"product|{product_id}")
                    h_category_pk = stable_uuid(f"category|{category_name}")

                    cur.execute(
                        """
                        INSERT INTO dds.h_product (h_product_pk, product_id, load_dt, load_src)
                        VALUES (%(pk)s, %(product_id)s, %(load_dt)s, %(src)s)
                        ON CONFLICT (h_product_pk) DO NOTHING
                        """,
                        {
                            "pk": h_product_pk,
                            "product_id": product_id,
                            "load_dt": load_dt,
                            "src": LOAD_SRC,
                        },
                    )
                    cur.execute(
                        """
                        INSERT INTO dds.h_category (h_category_pk, category_name, load_dt, load_src)
                        VALUES (%(pk)s, %(category_name)s, %(load_dt)s, %(src)s)
                        ON CONFLICT (h_category_pk) DO NOTHING
                        """,
                        {
                            "pk": h_category_pk,
                            "category_name": category_name,
                            "load_dt": load_dt,
                            "src": LOAD_SRC,
                        },
                    )
                    cur.execute(
                        """
                        INSERT INTO dds.s_product_names
                            (h_product_pk, name, load_dt, load_src, hk_product_names_hashdiff)
                        VALUES (%(pk)s, %(name)s, %(load_dt)s, %(src)s, %(hashdiff)s)
                        ON CONFLICT (h_product_pk, load_dt) DO NOTHING
                        """,
                        {
                            "pk": h_product_pk,
                            "name": product_name,
                            "load_dt": load_dt,
                            "src": LOAD_SRC,
                            "hashdiff": stable_uuid(f"product_name|{product_name}"),
                        },
                    )
                    cur.execute(
                        """
                        INSERT INTO dds.l_order_product
                            (hk_order_product_pk, h_order_pk, h_product_pk, load_dt, load_src)
                        VALUES (%(pk)s, %(h_order_pk)s, %(h_product_pk)s, %(load_dt)s, %(src)s)
                        ON CONFLICT (hk_order_product_pk) DO NOTHING
                        """,
                        {
                            "pk": stable_uuid(f"l_order_product|{h_order_pk}|{h_product_pk}"),
                            "h_order_pk": h_order_pk,
                            "h_product_pk": h_product_pk,
                            "load_dt": load_dt,
                            "src": LOAD_SRC,
                        },
                    )
                    cur.execute(
                        """
                        INSERT INTO dds.l_product_restaurant
                            (hk_product_restaurant_pk, h_product_pk, h_restaurant_pk, load_dt, load_src)
                        VALUES (%(pk)s, %(h_product_pk)s, %(h_restaurant_pk)s, %(load_dt)s, %(src)s)
                        ON CONFLICT (hk_product_restaurant_pk) DO NOTHING
                        """,
                        {
                            "pk": stable_uuid(f"l_product_restaurant|{h_product_pk}|{h_restaurant_pk}"),
                            "h_product_pk": h_product_pk,
                            "h_restaurant_pk": h_restaurant_pk,
                            "load_dt": load_dt,
                            "src": LOAD_SRC,
                        },
                    )
                    cur.execute(
                        """
                        INSERT INTO dds.l_product_category
                            (hk_product_category_pk, h_product_pk, h_category_pk, load_dt, load_src)
                        VALUES (%(pk)s, %(h_product_pk)s, %(h_category_pk)s, %(load_dt)s, %(src)s)
                        ON CONFLICT (hk_product_category_pk) DO NOTHING
                        """,
                        {
                            "pk": stable_uuid(f"l_product_category|{h_product_pk}|{h_category_pk}"),
                            "h_product_pk": h_product_pk,
                            "h_category_pk": h_category_pk,
                            "load_dt": load_dt,
                            "src": LOAD_SRC,
                        },
                    )
