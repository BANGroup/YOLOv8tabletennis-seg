import os

import requests


def submit(t_code, rlz_file=''):
    user_code = ''
    if rlz_file:
        full_lesson_path = os.path.dirname(os.path.abspath(__file__))
        user_file = f'{full_lesson_path}/{rlz_file}'

        with open(user_file, 'r', encoding="utf8") as u_file:
            user_code = u_file.read()

    r = requests.post(
        'http://localhost:3002',
        json={
            "code": user_code,
            "test": t_code
            })

    print(r.json()['stderr'].replace('__test', rlz_file[:-3]))
    print(r.json()['stdout'].replace('__test', rlz_file[:-3]))


if __name__ == '__main__':
    submit(
        'de06020901',
        'realization.py'
    )
