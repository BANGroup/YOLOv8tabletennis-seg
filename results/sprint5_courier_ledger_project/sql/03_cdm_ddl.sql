CREATE TABLE IF NOT EXISTS cdm.dm_courier_ledger (
  id serial PRIMARY KEY,
  courier_id varchar NOT NULL,
  courier_name varchar NOT NULL,
  settlement_year integer NOT NULL,
  settlement_month integer NOT NULL,
  orders_count integer NOT NULL DEFAULT 0,
  orders_total_sum numeric(14,2) NOT NULL DEFAULT 0,
  rate_avg numeric(14,2) NOT NULL DEFAULT 0,
  order_processing_fee numeric(14,2) NOT NULL DEFAULT 0,
  courier_order_sum numeric(14,2) NOT NULL DEFAULT 0,
  courier_tips_sum numeric(14,2) NOT NULL DEFAULT 0,
  courier_reward_sum numeric(14,2) NOT NULL DEFAULT 0,
  CONSTRAINT dm_courier_ledger_period_uq UNIQUE (courier_id, settlement_year, settlement_month),
  CONSTRAINT dm_courier_ledger_month_chk CHECK (settlement_month BETWEEN 1 AND 12),
  CONSTRAINT dm_courier_ledger_year_chk CHECK (settlement_year >= 2022 AND settlement_year < 2500)
);
