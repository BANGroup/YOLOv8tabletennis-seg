# Витрина выплат курьерам (финальный проект sprint-5)

## Поля витрины `cdm.dm_courier_ledger`

| Поле | Описание |
|------|----------|
| id | Идентификатор записи (serial PK) |
| courier_id | Бизнес-ключ курьера из API (`_id`) |
| courier_name | Имя курьера |
| settlement_year | Год периода расчёта (по дате заказа) |
| settlement_month | Месяц периода расчёта (по дате заказа) |
| orders_count | Число уникальных заказов с доставкой за период |
| orders_total_sum | Сумма `total_sum` по позициям заказа (`fct_product_sales`) |
| rate_avg | Средняя оценка доставки (`fct_deliveries.rate`) за период |
| order_processing_fee | `orders_total_sum * 0.25` |
| courier_order_sum | Сумма выплат за заказы по тарифной сетке от `rate_avg` |
| courier_tips_sum | Сумма чаевых (`tip_sum`) |
| courier_reward_sum | `courier_order_sum + courier_tips_sum * 0.95` |

## DDS: существующие и новые объекты

**Уже в DWH (используются в расчёте):** `dds.dm_orders`, `dds.dm_timestamps`, `dds.dm_users`, `dds.dm_restaurants`, `dds.fct_product_sales`, `dds.dm_products`, `dds.srv_wf_settings`.

**Новые:** `dds.dm_couriers`, `dds.fct_deliveries`.

**Изменение существующей таблицы:** в `dds.dm_orders` добавлен nullable столбец `courier_id` → `dds.dm_couriers(id)` (без NOT NULL, чтобы сохранить работу `orders_loader`).

## STG

- `stg.deliverysystem_couriers` — сырой JSON курьеров.
- `stg.deliverysystem_deliveries` — сырой JSON доставок + `order_id`, `order_ts` для индексации.

Рестораны из API не загружаются: справочник уже в `stg.ordersystem_restaurants` / `dds.dm_restaurants`.

## Сущности Delivery API для загрузки

| Эндпоинт | Назначение |
|----------|------------|
| `GET /couriers` | Справочник курьеров (`_id`, `name`) |
| `GET /deliveries` | Факты доставок (`delivery_id`, `order_id`, `order_ts`, `courier_id`, `rate`, `tip_sum`, `delivery_ts`, `address`, …) |

## Связи и grain отчёта

- Связь заказа: `deliveries.order_id` (Mongo `_id` заказа) = `dds.dm_orders.order_key`.
- Период расчёта и grain витрины: **год и месяц даты заказа** из `dds.dm_timestamps` (через `dm_orders.timestamp_id`), **не** `delivery_ts`.
- В расчёт попадают заказы со статусом `CLOSED`, для которых есть запись в `dds.fct_deliveries`.
- Сумма заказа: агрегат `SUM(fct_product_sales.total_sum)` по `order_id`.
