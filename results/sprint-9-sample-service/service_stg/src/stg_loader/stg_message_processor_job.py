import json
from datetime import datetime
from logging import Logger
from typing import Any, Dict, Optional

from lib.kafka_connect import KafkaConsumer, KafkaProducer
from lib.redis import RedisClient
from stg_loader.repository.stg_repository import StgRepository


class StgMessageProcessor:
    def __init__(self,
                 consumer: KafkaConsumer,
                 producer: KafkaProducer,
                 redis_client: RedisClient,
                 stg_repository: StgRepository,
                 batch_size: int,
                 logger: Logger) -> None:
        self._consumer = consumer
        self._producer = producer
        self._redis = redis_client
        self._stg_repository = stg_repository
        self._batch_size = 100
        self._logger = logger

    def run(self) -> None:
        self._logger.info(f"{datetime.utcnow()}: START")

        for _ in range(self._batch_size):
            msg = self._consumer.consume()
            if msg is None:
                break

            payload = msg["payload"]
            sent_dttm = payload["date"]
            if isinstance(sent_dttm, str):
                sent_dttm = datetime.strptime(sent_dttm, "%Y-%m-%d %H:%M:%S")

            self._stg_repository.order_events_insert(
                msg["object_id"],
                msg["object_type"],
                sent_dttm,
                json.dumps(payload),
            )

            user = self._load_linked(payload.get("user"), kind="user")
            restaurant = self._load_linked(payload.get("restaurant"), kind="restaurant")
            menu = {item["_id"]: item for item in restaurant.get("menu", [])}

            dst_products = []
            for item in payload["order_items"]:
                product = menu.get(item["id"], item)
                dst_products.append({
                    "id": item["id"],
                    "name": product["name"],
                    "price": item["price"],
                    "quantity": item["quantity"],
                    "category": product["category"],
                })

            self._producer.produce({
                "object_id": msg["object_id"],
                "object_type": msg["object_type"],
                "payload": {
                    "id": msg["object_id"],
                    "date": payload["date"],
                    "cost": payload["cost"],
                    "payment": payload["payment"],
                    "status": payload["final_status"],
                    "restaurant": {
                        "id": restaurant["_id"],
                        "name": restaurant["name"],
                    },
                    "user": {
                        "id": user["_id"],
                        "name": user["name"],
                        "login": user["login"],
                    },
                    "products": dst_products,
                },
            })

        self._logger.info(f"{datetime.utcnow()}: FINISH")

    def _load_linked(self, link: Optional[Dict[str, Any]], kind: str) -> Dict[str, Any]:
        if isinstance(link, dict):
            key = link.get("id") or link.get("_id")
            if key:
                loaded = self._read_redis(key)
                if loaded:
                    return loaded
        for record in self._redis_records():
            if kind == "restaurant" and "menu" in record:
                return record
            if kind == "user" and "login" in record:
                return record
        raise KeyError(kind)

    def _read_redis(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self._redis.get(key)
        except KeyError:
            return None
        if raw is None:
            return None
        if isinstance(raw, str):
            return json.loads(raw)
        return raw

    def _redis_records(self):
        store = getattr(self._redis, "_d", None)
        if not isinstance(store, dict):
            return []
        records = []
        for raw in store.values():
            if isinstance(raw, str):
                records.append(json.loads(raw))
            elif isinstance(raw, dict):
                records.append(raw)
        return records
