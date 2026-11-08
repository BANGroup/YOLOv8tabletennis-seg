from datetime import datetime
from decimal import Decimal
from logging import Logger

from lib.kafka_connect import KafkaConsumer, KafkaProducer
from dds_loader.repository.dds_repository import DdsRepository


class DdsMessageProcessor:
    def __init__(
        self,
        consumer: KafkaConsumer,
        producer: KafkaProducer,
        dds_repository: DdsRepository,
        batch_size: int,
        logger: Logger,
    ) -> None:
        self._consumer = consumer
        self._producer = producer
        self._dds_repository = dds_repository
        self._batch_size = batch_size
        self._logger = logger

    def run(self) -> None:
        self._logger.info(f"{datetime.utcnow()}: START")
        processed = 0

        for _ in range(self._batch_size):
            msg = self._consumer.consume()
            if msg is None:
                break

            payload = msg["payload"]
            order_dt = datetime.strptime(payload["date"], "%Y-%m-%d %H:%M:%S")
            self._dds_repository.save_order(
                order_id=int(msg["object_id"]),
                order_dt=order_dt,
                cost=Decimal(str(payload["cost"])),
                payment=Decimal(str(payload["payment"])),
                status=str(payload["status"]),
                user_id=str(payload["user"]["id"]),
                username=str(payload["user"]["name"]),
                userlogin=str(payload["user"]["login"]),
                restaurant_id=str(payload["restaurant"]["id"]),
                restaurant_name=str(payload["restaurant"]["name"]),
                products=payload["products"],
            )
            self._producer.produce(
                {
                    "object_id": msg["object_id"],
                    "object_type": "dds_order",
                    "payload": {
                        "user_id": payload["user"]["id"],
                        "order_id": msg["object_id"],
                    },
                }
            )
            processed += 1

        if processed:
            self._consumer.commit()
        self._logger.info(f"{datetime.utcnow()}: FINISH processed={processed}")
