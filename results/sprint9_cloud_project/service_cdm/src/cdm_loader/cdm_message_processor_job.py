from datetime import datetime
from logging import Logger

from lib.kafka_connect import KafkaConsumer
from cdm_loader.repository.cdm_repository import CdmRepository


class CdmMessageProcessor:
    def __init__(
        self,
        consumer: KafkaConsumer,
        cdm_repository: CdmRepository,
        batch_size: int,
        logger: Logger,
    ) -> None:
        self._consumer = consumer
        self._cdm_repository = cdm_repository
        self._batch_size = batch_size
        self._logger = logger

    def run(self) -> None:
        self._logger.info(f"{datetime.utcnow()}: START")
        processed = 0
        seen_users = set()

        for _ in range(self._batch_size):
            msg = self._consumer.consume()
            if msg is None:
                break
            user_id = str(msg["payload"]["user_id"])
            if user_id not in seen_users:
                self._cdm_repository.refresh_user(user_id)
                seen_users.add(user_id)
            processed += 1

        if processed:
            self._consumer.commit()
        self._logger.info(f"{datetime.utcnow()}: FINISH processed={processed}")
