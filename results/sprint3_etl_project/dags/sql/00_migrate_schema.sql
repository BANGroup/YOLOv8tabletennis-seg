-- Миграция схемы staging и витрин (этапы 1–2 проекта sprint3)

CREATE TABLE IF NOT EXISTS staging.customer_research (
    date_id integer,
    category_id integer,
    geo_id integer,
    sales_qty bigint,
    sales_amt numeric(12, 2)
);

CREATE TABLE IF NOT EXISTS staging.user_activity_log (
    id integer,
    uniq_id character varying(32) NOT NULL,
    date_time timestamp without time zone,
    action_id integer,
    customer_id integer,
    quantity bigint,
    CONSTRAINT user_activity_log_pk PRIMARY KEY (uniq_id)
);

CREATE TABLE IF NOT EXISTS staging.price_log (
    item_name character varying(200) NOT NULL,
    price numeric(10, 2) NOT NULL
);

ALTER TABLE staging.user_order_log
    ADD COLUMN IF NOT EXISTS status character varying(20);

ALTER TABLE mart.f_sales
    ADD COLUMN IF NOT EXISTS status character varying(20);

ALTER TABLE mart.f_sales
    ADD COLUMN IF NOT EXISTS uniq_id character varying(32);

UPDATE mart.f_sales
SET status = 'shipped'
WHERE status IS NULL;

ALTER TABLE mart.f_sales
    ALTER COLUMN status SET DEFAULT 'shipped';

CREATE UNIQUE INDEX IF NOT EXISTS f_sales_uniq_id_uq
    ON mart.f_sales (uniq_id)
    WHERE uniq_id IS NOT NULL;

CREATE SEQUENCE IF NOT EXISTS mart.f_customer_retention_id_seq;

CREATE TABLE IF NOT EXISTS mart.f_customer_retention (
    id integer NOT NULL DEFAULT nextval('mart.f_customer_retention_id_seq'::regclass),
    period_name character varying(20) NOT NULL,
    period_id integer NOT NULL,
    item_id integer NOT NULL,
    new_customers_count bigint,
    returning_customers_count bigint,
    refunded_customer_count bigint,
    new_customers_revenue numeric(14, 2),
    returning_customers_revenue numeric(14, 2),
    customers_refunded bigint,
    CONSTRAINT f_customer_retention_pkey PRIMARY KEY (id),
    CONSTRAINT f_customer_retention_period_item_uq UNIQUE (period_name, period_id, item_id)
);
