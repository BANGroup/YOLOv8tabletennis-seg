CREATE TABLE IF NOT EXISTS stg.deliverysystem_couriers (
  id serial PRIMARY KEY,
  object_id varchar NOT NULL UNIQUE,
  object_value text NOT NULL,
  load_ts timestamp NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS stg.deliverysystem_deliveries (
  id serial PRIMARY KEY,
  object_id varchar NOT NULL UNIQUE,
  object_value text NOT NULL,
  order_id varchar NOT NULL,
  order_ts timestamp NOT NULL,
  load_ts timestamp NOT NULL DEFAULT now()
);
