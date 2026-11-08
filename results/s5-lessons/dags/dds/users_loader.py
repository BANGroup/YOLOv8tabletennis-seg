from datetime import datetime
from logging import Logger
from typing import List

from dds.dds_settings_repository import DdsEtlSettingsRepository, EtlSetting
from lib import PgConnect
from lib.dict_util import json2str, str2json
from psycopg import Connection
from psycopg.rows import class_row
from pydantic import BaseModel


class UserStgObj(BaseModel):
    id: int
    object_id: str
    object_value: str
    update_ts: datetime


class UserDdsObj(BaseModel):
    user_id: str
    user_name: str
    user_login: str


class UsersStgRepository:
    def __init__(self, pg: PgConnect) -> None:
        self._db = pg

    def list_users(self, threshold: int, limit: int) -> List[UserStgObj]:
        with self._db.client().cursor(row_factory=class_row(UserStgObj)) as cur:
            cur.execute(
                """
                    SELECT id, object_id, object_value, update_ts
                    FROM stg.ordersystem_users
                    WHERE id > %(threshold)s
                    ORDER BY id ASC
                    LIMIT %(limit)s;
                """,
                {
                    "threshold": threshold,
                    "limit": limit,
                },
            )
            return cur.fetchall()


class UserDdsRepository:
    def insert_user(self, conn: Connection, user: UserDdsObj) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                    INSERT INTO dds.dm_users(user_id, user_name, user_login)
                    VALUES (%(user_id)s, %(user_name)s, %(user_login)s);
                """,
                {
                    "user_id": user.user_id,
                    "user_name": user.user_name,
                    "user_login": user.user_login,
                },
            )


class UserLoader:
    WF_KEY = "stg_to_dds_users_workflow"
    LAST_LOADED_ID_KEY = "last_loaded_id"
    BATCH_LIMIT = 100

    def __init__(self, dwh: PgConnect, log: Logger) -> None:
        self.dwh = dwh
        self.stg = UsersStgRepository(dwh)
        self.dds = UserDdsRepository()
        self.settings_repository = DdsEtlSettingsRepository()
        self.log = log

    def load_users(self) -> None:
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
                load_queue = self.stg.list_users(last_loaded, self.BATCH_LIMIT)
                self.log.info(
                    f"Found {len(load_queue)} users to load into dds (after id={last_loaded})."
                )
                if not load_queue:
                    self.log.info("No more users to load. Quitting.")
                    return

                for row in load_queue:
                    data = str2json(row.object_value)
                    user = UserDdsObj(
                        user_id=str(data["_id"]),
                        user_name=data["name"],
                        user_login=data["login"],
                    )
                    self.dds.insert_user(conn, user)

                wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY] = max(
                    u.id for u in load_queue
                )
                wf_setting_json = json2str(wf_setting.workflow_settings)
                self.settings_repository.save_setting(
                    conn, wf_setting.workflow_key, wf_setting_json
                )

                self.log.info(
                    f"Batch finished at last_loaded_id="
                    f"{wf_setting.workflow_settings[self.LAST_LOADED_ID_KEY]}"
                )
