"""Загрузка link-таблиц DWH (de06030604). Пароль только из окружения."""
import os

import vertica_python

vertica_user = 'vt2609246827be'

_USER = vertica_user.upper()
_DWH = f'{_USER}__DWH'
_STG = f'{_USER}__STAGING'
_ROLE = f'{_USER}_A'

SQL_LOADS = (
    f"""
INSERT INTO {_DWH}.l_admins(hk_l_admin_id, hk_group_id, hk_user_id, load_dt, load_src)
SELECT
    hash(hg.hk_group_id, hu.hk_user_id),
    hg.hk_group_id,
    hu.hk_user_id,
    now() AS load_dt,
    's3' AS load_src
FROM {_STG}.groups AS g
LEFT JOIN {_DWH}.h_users AS hu ON g.admin_id = hu.user_id
LEFT JOIN {_DWH}.h_groups AS hg ON g.id = hg.group_id
WHERE hash(hg.hk_group_id, hu.hk_user_id) NOT IN (
    SELECT hk_l_admin_id FROM {_DWH}.l_admins
);
""",
    f"""
INSERT INTO {_DWH}.l_groups_dialogs(
    hk_l_groups_dialogs, hk_message_id, hk_group_id, load_dt, load_src
)
SELECT
    hash(hd.hk_message_id, hg.hk_group_id),
    hd.hk_message_id,
    hg.hk_group_id,
    now() AS load_dt,
    's3' AS load_src
FROM {_STG}.dialogs AS d
LEFT JOIN {_DWH}.h_groups AS hg ON d.message_group = hg.group_id
LEFT JOIN {_DWH}.h_dialogs AS hd ON d.message_id = hd.message_id
WHERE d.message_group IS NOT NULL
  AND hash(hd.hk_message_id, hg.hk_group_id) NOT IN (
    SELECT hk_l_groups_dialogs FROM {_DWH}.l_groups_dialogs
);
""",
    f"""
INSERT INTO {_DWH}.l_user_message(hk_l_user_message, hk_user_id, hk_message_id, load_dt, load_src)
SELECT
    hash(hd.hk_message_id, hu.hk_user_id),
    hd.hk_message_id,
    hu.hk_user_id,
    now() AS load_dt,
    's3' AS load_src
FROM {_STG}.dialogs AS d
LEFT JOIN {_DWH}.h_users AS hu ON d.message_from = hu.user_id
LEFT JOIN {_DWH}.h_dialogs AS hd ON d.message_id = hd.message_id
WHERE d.message_group IS NOT NULL
  AND hash(hd.hk_message_id, hu.hk_user_id) NOT IN (
    SELECT hk_l_user_message FROM {_DWH}.l_user_message
)
  AND hu.hk_user_id IS NOT NULL;
""",
)

_COUNT_QUERIES = (
    ('l_admins', f'SELECT COUNT(*) FROM {_DWH}.l_admins'),
    ('l_groups_dialogs', f'SELECT COUNT(*) FROM {_DWH}.l_groups_dialogs'),
    ('l_user_message', f'SELECT COUNT(*) FROM {_DWH}.l_user_message'),
)


def _conn_info() -> dict:
    return {
        'host': os.environ.get('VERTICA_HOST', 'vertica.data-engineer.education-services.ru'),
        'port': int(os.environ.get('VERTICA_PORT', '5433')),
        'user': os.environ.get('VERTICA_USER', vertica_user),
        'password': os.environ['VERTICA_PASSWORD'],
        'database': os.environ.get('VERTICA_DB', 'dwh'),
        'autocommit': True,
        'tlsmode': os.environ.get('VERTICA_TLSMODE', 'disable'),
    }


def main() -> None:
    with vertica_python.connect(**_conn_info()) as conn:
        cur = conn.cursor()
        cur.execute(f'SET ROLE {_ROLE}')
        for sql in SQL_LOADS:
            cur.execute(sql)
        for name, q in _COUNT_QUERIES:
            cur.execute(q)
            print(f'{name}: {cur.fetchone()[0]}')


if __name__ == '__main__':
    main()
