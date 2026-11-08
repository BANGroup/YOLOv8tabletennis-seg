# Спринт 5

* Скопируйте проект в директорию:
```shell script
git clone https://github.com/practicum-de/s5-lessons.git
```
* Перейдите в директорию c проектом:
```shell script
cd s5-lessons
```
* Создайте [виртуальное окружение](https://docs.python.org/3/library/venv.html) и активируйте его:
```shell script
python3 -m venv venv
```
или для Windows
```shell script
python -m venv venv
```
Проверить, что виртуальное окружение создано можно командой ls - в списке с файлов и директорий вы увидите директорию venv

* Активируйте его:
```shell script
source venv/bin/activate
```
или в Windows
```shell script
source venv/Scripts/activate
```
или альтернативный способ для Windows
```shell script
.\venv\Scripts\activate.bat
```

* Обновите pip до последней версии:
```shell script
pip install --upgrade pip
```
* Установите зависимости:
```shell script
pip install -r requirements.txt
```

Для выполнения заданий выполните:

`docker compose up -d`

Перед запуском тестов убедитесь что запущен контейнер.

Если у Вас не установлен python 3.8 то самое время сделать это. 

Поключние к БД:
```
  "host": "localhost",
  "user": "jovyan",
  "password": "jovyan"
  "port": 15432,
  "ssl": false,
  "database": "de"
```

Metabase доступен по адресу http://localhost:3333/
Для подключения с помощью Metabase к БД в контейнере укажите следующие параметры 
```
  "host": "de-pg-cr-af",
  "user": "jovyan",
  "password": "jovyan"
  "port": 5432,
  "ssl": false,
  "database": "de"
```

Airflow доступен по адресу http://localhost:3000/airflow
```
login: AirflowAdmin
password: airflow_pass
```

Скачайте Mongo DB Compass c [официального сайта](https://www.mongodb.com/products/compass) или по ссылкам ниже:

* [macOS arm64 (M1) (11.0+)](https://storage.yandexcloud.net/mongo-db-compass-1.3.3.1/mongodb-compass-1.33.1-darwin-arm64.dmg)

* [macOS 64-bit (10.14+)](https://storage.yandexcloud.net/mongo-db-compass-1.3.3.1/mongodb-compass-1.33.1-darwin-x64.dmg)

* [RedHat 64-bit (7+)](https://storage.yandexcloud.net/mongo-db-compass-1.3.3.1/mongodb-compass-1.33.1.x86_64.rpm)

* [Ubuntu 64-bit (16.04+)](https://storage.yandexcloud.net/mongo-db-compass-1.3.3.1/mongodb-compass_1.33.1_amd64.deb)

* [Windows 64-bit (7+)](https://storage.yandexcloud.net/mongo-db-compass-1.3.3.1/mongodb-compass-1.33.1-win32-x64.exe)

* [Windows 64-bit (7+) (MSI)](https://storage.yandexcloud.net/mongo-db-compass-1.3.3.1/mongodb-compass-1.33.1-win32-x64.msi)

## Примечание
При запуске контейнеров командой `docker compose up -d` создается волюм для сохранения состояния базы данных между запусками. При необходимости сброса состояния к исходному выполните слеующие шаги:
- Остановите контейнеры: `docker compose down`.
- Выведите список волюмов: `docker volume ls`. Должен быть волюм для 5 спринта под именем `s5-lessons_s5t4postgresql`.
- Удалите волюм: `docker volume rm s5-lessons_s5t4postgresql`
- Запустите контейнеры: `docker compose up -d`.
