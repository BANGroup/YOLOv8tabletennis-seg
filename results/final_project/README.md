# Выпускной проект

Источник: бакет S3 `final-project` (`transactions_batch_*.csv`, `currencies_history.csv`).

Загрузка за октябрь 2022 идёт в `VT2609246827BE__STAGING.transactions` и `VT2609246827BE__STAGING.currencies`.
Витрина: `VT2609246827BE__DWH.global_metrics` на `vertica.data-engineer.education-services.ru:5433`, база `dwh`.

Поле `amount_total` — сумма успешных транзакций (`status = done`) в долларах. Код доллара в этих данных — `420`. Сумма в минимальных единицах умножается на курс к `420` и делится на 100. Строки с номером счёта меньше 0 не попадают в витрину. Уникальный счёт в `cnt_accounts_make_transactions` — `account_number_to`: в выгрузке это клиентский счёт, а `account_number_from` у успешных операций — служебные счета `903810` и `914810`.

Порядок: DAG `1_data_import`, затем DAG `2_datamart_update`. Логическая дата запуска — день, за который пишется партиция.
