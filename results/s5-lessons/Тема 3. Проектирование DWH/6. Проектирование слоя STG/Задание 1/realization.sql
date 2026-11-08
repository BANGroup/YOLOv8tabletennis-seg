CREATE SCHEMA IF NOT EXISTS stg;

CREATE TABLE stg.bonussystem_events
(
    id          INTEGER   NOT NULL PRIMARY KEY,
    event_ts    TIMESTAMP NOT NULL,
    event_type  VARCHAR   NOT NULL,
    event_value TEXT      NOT NULL
);

CREATE INDEX idx_bonussystem_events__event_ts
    ON stg.bonussystem_events (event_ts);


CREATE TABLE stg.bonussystem_users
(
    id            INTEGER NOT NULL PRIMARY KEY,
    order_user_id TEXT    NOT NULL
);

CREATE TABLE stg.bonussystem_ranks
(
    id                    INTEGER                  NOT NULL PRIMARY KEY,
    name                  VARCHAR(2048)            NOT NULL,
    bonus_percent         NUMERIC(19, 5) DEFAULT 0 NOT NULL
        CONSTRAINT bonussystem_ranks_bonus_percent_check CHECK (bonus_percent >= (0)::NUMERIC),
    min_payment_threshold NUMERIC(19, 5) DEFAULT 0 NOT NULL
        CONSTRAINT bonussystem_ranks_min_payment_threshold_check CHECK (min_payment_threshold >= (0)::NUMERIC)
);

-- for test
-- CREATE SCHEMA stg_test;

-- CREATE TABLE stg_test.bonussystem_events
-- (
--     id          INTEGER   NOT NULL PRIMARY KEY,
--     event_ts    TIMESTAMP NOT NULL,
--     event_type  VARCHAR   NOT NULL,
--     event_value TEXT      NOT NULL
-- );

-- CREATE INDEX idx_bonussystem_events__event_ts
--     ON stg_test.bonussystem_events (event_ts);


-- CREATE TABLE stg_test.bonussystem_users
-- (
--     id            INTEGER NOT NULL PRIMARY KEY,
--     order_user_id TEXT    NOT NULL
-- );

-- CREATE TABLE stg_test.bonussystem_ranks
-- (
--     id                    INTEGER                  NOT NULL PRIMARY KEY,
--     name                  VARCHAR(2048)            NOT NULL,
--     bonus_percent         NUMERIC(19, 5) DEFAULT 0 NOT NULL
--         CONSTRAINT bonussystem_ranks_bonus_percent_check CHECK (bonus_percent >= (0)::NUMERIC),
--     min_payment_threshold NUMERIC(19, 5) DEFAULT 0 NOT NULL
--         CONSTRAINT bonussystem_ranks_min_payment_threshold_check CHECK (min_payment_threshold >= (0)::NUMERIC)
-- );
