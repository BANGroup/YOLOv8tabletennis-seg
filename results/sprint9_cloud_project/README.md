# Загрузка заказов в DWH

Сервисы читают поток заказов и заполняют слои PostgreSQL `sprint9dwh`.

## Образы

- stg: `cr.yandex/crpj52tm6180eu1tlbg6/stg_service:v2026-09-27-r2`
- dds: `cr.yandex/crpj52tm6180eu1tlbg6/dds_service:v2026-09-27-r1`
- cdm: `cr.yandex/crpj52tm6180eu1tlbg6/cdm_service:v2026-09-27-r1`

## Поток

1. `order-service_orders` → `service` stg пишет `stg.order_events` и публикует обогащённый заказ в `stg-service-orders`.
2. `service_dds` читает `stg-service-orders`, идемпотентно заполняет хабы, линки и сателлиты `dds` и публикует `dds-service-orders`.
3. `service_cdm` читает `dds-service-orders` и пересчитывает витрины `cdm.user_product_counters` и `cdm.user_category_counters` для пользователя из сообщения.

Ключи хабов и поля `user_id` / `product_id` / `category_id` в витринах — один и тот же UUID версии 5 от бизнес-ключа. Повторная обработка того же сообщения не создаёт дублей: в DDS стоит `ON CONFLICT DO NOTHING`, в CDM счётчики пересчитываются из DDS.

`load_src` в DDS: `orders-system-kafka`.

## Выкат

Из каталога сервиса:

```
helm upgrade --install --atomic dds-service app -n <namespace>
helm upgrade --install --atomic cdm-service app -n <namespace>
```

## Дашборд DataLens

Шаг с дашбордом в курсе необязательный, ссылка на DataLens здесь не приводится: дашборд создаётся в аккаунте Yandex ID. Датасет строится по витринам. Чарты:

- доля пользователей по категориям: `COUNT(user_id)` из `cdm.user_category_counters`, группировка `category_name`;
- доля заказов по категориям: `SUM(order_cnt)` из `cdm.user_category_counters`, группировка `category_name`;
- пользователи по блюдам: `COUNT(user_id)` из `cdm.user_product_counters`, группировка `product_name`;
- заказы по блюдам: `SUM(order_cnt)` из `cdm.user_product_counters`, группировка `product_name`.
