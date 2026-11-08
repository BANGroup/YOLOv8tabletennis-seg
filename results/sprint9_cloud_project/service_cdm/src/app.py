import logging

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask

from app_config import AppConfig
from cdm_loader.cdm_message_processor_job import CdmMessageProcessor
from cdm_loader.repository.cdm_repository import CdmRepository

app = Flask(__name__)


@app.get("/health")
def health():
    return "healthy"


if __name__ == "__main__":
    app.logger.setLevel(logging.DEBUG)
    config = AppConfig()
    proc = CdmMessageProcessor(
        config.kafka_consumer(),
        CdmRepository(config.pg_warehouse_db()),
        100,
        app.logger,
    )
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=proc.run, trigger="interval", seconds=config.DEFAULT_JOB_INTERVAL)
    scheduler.start()
    app.run(debug=True, host="0.0.0.0", use_reloader=False)
