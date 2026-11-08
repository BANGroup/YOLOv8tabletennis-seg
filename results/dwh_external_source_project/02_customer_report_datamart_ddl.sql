/* DDL витрины заказчиков и таблицы watermark для инкрементальной загрузки */

BEGIN;

DROP TABLE IF EXISTS dwh.load_dates_customer_report_datamart;
DROP TABLE IF EXISTS dwh.customer_report_datamart;

CREATE TABLE dwh.customer_report_datamart (
    id BIGINT GENERATED ALWAYS AS IDENTITY NOT NULL,
    customer_id BIGINT NOT NULL,
    customer_name VARCHAR NOT NULL,
    customer_address VARCHAR NOT NULL,
    customer_birthday DATE NOT NULL,
    customer_email VARCHAR NOT NULL,
    customer_money NUMERIC(15,2) NOT NULL,
    platform_money BIGINT NOT NULL,
    count_order BIGINT NOT NULL,
    avg_price_order NUMERIC(10,2) NOT NULL,
    median_time_order_completed NUMERIC(10,1),
    top_product_category VARCHAR NOT NULL,
    top_craftsman_id BIGINT NOT NULL,
    count_order_created BIGINT NOT NULL,
    count_order_in_progress BIGINT NOT NULL,
    count_order_delivery BIGINT NOT NULL,
    count_order_done BIGINT NOT NULL,
    count_order_not_done BIGINT NOT NULL,
    report_period VARCHAR NOT NULL,
    CONSTRAINT customer_report_datamart_pk PRIMARY KEY (id),
    CONSTRAINT customer_report_datamart_customer_id_report_period_uq UNIQUE (customer_id, report_period)
);

COMMENT ON TABLE dwh.customer_report_datamart IS 'Инкрементальная витрина отчётности по заказчикам маркетплейса; зерно: customer_id + report_period';
COMMENT ON COLUMN dwh.customer_report_datamart.id IS 'Сурrogate-ключ записи витрины';
COMMENT ON COLUMN dwh.customer_report_datamart.customer_id IS 'Идентификатор заказчика';
COMMENT ON COLUMN dwh.customer_report_datamart.customer_name IS 'Ф. И. О. заказчика';
COMMENT ON COLUMN dwh.customer_report_datamart.customer_address IS 'Адрес заказчика';
COMMENT ON COLUMN dwh.customer_report_datamart.customer_birthday IS 'Дата рождения заказчика';
COMMENT ON COLUMN dwh.customer_report_datamart.customer_email IS 'Электронная почта заказчика';
COMMENT ON COLUMN dwh.customer_report_datamart.customer_money IS 'Сумма, которую потратил заказчик за отчётный период';
COMMENT ON COLUMN dwh.customer_report_datamart.platform_money IS 'Сумма, которую заработала платформа от покупок заказчика за период (10% от customer_money)';
COMMENT ON COLUMN dwh.customer_report_datamart.count_order IS 'Количество заказов у заказчика за отчётный период';
COMMENT ON COLUMN dwh.customer_report_datamart.avg_price_order IS 'Средняя стоимость одного заказа у заказчика за отчётный период';
COMMENT ON COLUMN dwh.customer_report_datamart.median_time_order_completed IS 'Медианное время в днях от создания заказа до завершения за отчётный период';
COMMENT ON COLUMN dwh.customer_report_datamart.top_product_category IS 'Самая популярная категория товаров у заказчика за отчётный период';
COMMENT ON COLUMN dwh.customer_report_datamart.top_craftsman_id IS 'Идентификатор самого популярного мастера у заказчика за отчётный период';
COMMENT ON COLUMN dwh.customer_report_datamart.count_order_created IS 'Количество заказов в статусе created за отчётный период';
COMMENT ON COLUMN dwh.customer_report_datamart.count_order_in_progress IS 'Количество заказов в статусе in progress за отчётный период';
COMMENT ON COLUMN dwh.customer_report_datamart.count_order_delivery IS 'Количество заказов в статусе delivery за отчётный период';
COMMENT ON COLUMN dwh.customer_report_datamart.count_order_done IS 'Количество заказов в статусе done за отчётный период';
COMMENT ON COLUMN dwh.customer_report_datamart.count_order_not_done IS 'Количество незавершённых заказов за отчётный период';
COMMENT ON COLUMN dwh.customer_report_datamart.report_period IS 'Отчётный период в формате yyyy-mm';

CREATE TABLE dwh.load_dates_customer_report_datamart (
    id BIGINT GENERATED ALWAYS AS IDENTITY NOT NULL,
    load_dttm DATE NOT NULL,
    CONSTRAINT load_dates_customer_report_datamart_pk PRIMARY KEY (id)
);

COMMENT ON TABLE dwh.load_dates_customer_report_datamart IS 'Watermark-таблица дат инкрементальных загрузок витрины dwh.customer_report_datamart';
COMMENT ON COLUMN dwh.load_dates_customer_report_datamart.id IS 'Сурrogate-ключ записи watermark';
COMMENT ON COLUMN dwh.load_dates_customer_report_datamart.load_dttm IS 'Дата загрузки данных в DWH, зафиксированная при обновлении витрины';

COMMIT;
