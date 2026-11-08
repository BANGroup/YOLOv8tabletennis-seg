CREATE TABLE IF NOT EXISTS dds.dm_couriers (
  id serial PRIMARY KEY,
  courier_id varchar NOT NULL UNIQUE,
  courier_name varchar NOT NULL
);

CREATE TABLE IF NOT EXISTS dds.fct_deliveries (
  id serial PRIMARY KEY,
  delivery_id varchar NOT NULL UNIQUE,
  order_id integer REFERENCES dds.dm_orders(id),
  courier_id integer NOT NULL REFERENCES dds.dm_couriers(id),
  timestamp_id integer REFERENCES dds.dm_timestamps(id),
  rate smallint NOT NULL,
  tip_sum numeric(14,2) NOT NULL DEFAULT 0,
  delivery_ts timestamp NOT NULL,
  address varchar
);

ALTER TABLE dds.dm_orders ADD COLUMN IF NOT EXISTS courier_id integer REFERENCES dds.dm_couriers(id);
