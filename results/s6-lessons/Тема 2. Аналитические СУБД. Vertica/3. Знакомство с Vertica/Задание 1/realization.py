import os

import vertica_python

conn_info = {
    'host': 'vertica.data-engineer.education-services.ru',
    'port': 5433,
    'user': 'vt2609246827be',
    'password': os.environ.get('VERTICA_PASSWORD', ''),
    'database': 'dwh',
    'autocommit': True,
}

def try_select(conn_info=conn_info):
	# И рекомендуем использовать соединение вот так
	with vertica_python.connect(**conn_info) as conn:
		cur = conn.cursor()
		cur.execute('SELECT 1 as a1')
		res = cur.fetchall()
		return res
