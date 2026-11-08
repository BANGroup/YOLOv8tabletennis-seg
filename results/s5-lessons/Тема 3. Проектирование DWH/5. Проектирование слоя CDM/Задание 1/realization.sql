DROP TABLE IF EXISTS cdm.dm_settlement_report;
CREATE TABLE cdm.dm_settlement_report (
    id serial not null,
    restaurant_id varchar not null,
    restaurant_name varchar not null,
    settlement_date date not null,
    orders_count integer not null,
    orders_total_sum numeric(14, 2) not null,
    orders_bonus_payment_sum numeric(14, 2) not null,
    orders_bonus_granted_sum numeric(14, 2) not null,
    order_processing_fee numeric(14, 2) not null,
    restaurant_reward_sum numeric(14, 2) not null
);
