from logging import Logger
from typing import List, Optional

from dds.dds_settings_repository import DdsEtlSettingsRepository, EtlSetting
from lib import PgConnect
from lib.dict_util import json2str, str2json
from psycopg import Connection
from psycopg.rows import class_row
from pydantic import BaseModel


class CourierStgObj(BaseModel):
    id: int
    object_id: str
    object_value: str


class CouriersStgRepository:
    def __init__(self, pg: PgConnect) -> None:
        self._db = pg

    def list_couriers(self, threshold: int, limit: int) -> List[CourierStgObj]:
        with self._db.client().cursor(row_factory=class_row(CourierStgObj)) as cur:
            cur.execute(
                """
                    SELECT id, object_id, object_value
                    FROM stg.deliverysystem_couriers
                    WHERE id > %(threshold)s
                    ORDER BY id ASC
                    LIMIT %(limit)s;
                """,
                {"threshold": threshold, "limit": limit},
            )
            return cur.fetchall()


class CourierDdsRepository:
    def upsert_courier(
        self, conn: Connection, courier_id: str, courier_name: str
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO dds.dm_couriers(courier_id, courier_name)
                    VALUES (%(courier_id)s, %(courier_name)s)
                    ON CONFLICT (courier_id) DO UPDATE
                    SET courier_name = EXCLUDED.courier_name;
                """,
                {
                    "courier_id": courier_id,
                    "courier_name": courier_name,
                },
            )

    def find_courier_pk(self, conn: Connection, courier_id: str) -> Optional[int]:
        with conn.cursor() as cur:
            cur.execute(
                """
                    SELECT id FROM dds.dm_couriers WHERE courier_id = %(courier_id)s;
                """,
                {"courier_id": courier_id},
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]


class CouriersDdsLoader:
    WF_KEY = "stg_to_dds_couriers_workflow"
    LAST_LOADED_ID_KEY = "last_loaded_id"
    BATCH_LIMIT = 500

    def __init__(self, dwh: PgConnect, log: Logger) -> None:
        self.dwh = dwh
        self.stg = CouriersStgRepository(dwh)
        self.dds = CourierDdsRepository()
        self.settings_repository = DdsEtlSettingsRepository()
        self.log = log

    def load_couriers(self) -> None:
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
                load_queue = self.stg.list_couriers(last_loaded, self.BATCH_LIMIT)
                self.log.info(
                    f"Found {len(load_queue)} couriers for DDS (after id={last_loaded})"
                )
                if not load_queue:
                    self.log.info("No more couriers to load into dds.")
                    return

                loaded = 0
                skipped = 0
                successful_stg_ids: List[int] = []
                for row in load_queue:
                    data = str2json(row.object_value)
                    name = data.get("name")
                    if not name:
                        self.log.warning(
                            f"Skip courier stg.id={row.id}: missing name"
                        )
                        skipped += 1
                        continue
                    self.dds.upsert_courier(conn, str(data["_id"]), str(name))
                    loaded += 1
                    successful_stg_ids.append(row.id)

                if loaded > 0:
                    wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY] = max(
                        successful_stg_ids
                    )
                    wf_setting_json = json2str(wf_setting.workflow_settings)
                    self.settings_repository.save_setting(
                        conn, wf_setting.workflow_key, wf_setting_json
                    )
                    self.log.info(
                        f"Batch finished: loaded={loaded}, skipped={skipped}, "
                        f"last_loaded_id="
                        f"{wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY]}"
                    )
                else:
                    self.log.warning(
                        f"Batch fully skipped: skipped={skipped} "
                        f"(reason: missing name); watermark unchanged "
                        f"last_loaded_id={last_loaded}"
                    )
                    return
