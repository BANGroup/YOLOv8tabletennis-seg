DROP TABLE IF EXISTS VT2609246827BE__STAGING.transactions CASCADE;
DROP TABLE IF EXISTS VT2609246827BE__STAGING.currencies CASCADE;

CREATE TABLE VT2609246827BE__STAGING.transactions
(
    operation_id varchar(64) NOT NULL,
    account_number_from int NOT NULL,
    account_number_to int NOT NULL,
    currency_code int NOT NULL,
    country varchar(64),
    status varchar(32),
    transaction_type varchar(64),
    amount int NOT NULL,
    transaction_dt timestamp NOT NULL
)
PARTITION BY ((transactions.transaction_dt)::date)
GROUP BY calendar_hierarchy_day((transactions.transaction_dt)::date, 3, 2);

CREATE PROJECTION VT2609246827BE__STAGING.transactions_by_date
(
    operation_id,
    account_number_from,
    account_number_to,
    currency_code,
    country,
    status,
    transaction_type,
    amount,
    transaction_dt
)
AS
SELECT
    transactions.operation_id,
    transactions.account_number_from,
    transactions.account_number_to,
    transactions.currency_code,
    transactions.country,
    transactions.status,
    transactions.transaction_type,
    transactions.amount,
    transactions.transaction_dt
FROM VT2609246827BE__STAGING.transactions
ORDER BY transactions.transaction_dt, transactions.operation_id
SEGMENTED BY hash(transactions.transaction_dt, transactions.operation_id) ALL NODES;

CREATE TABLE VT2609246827BE__STAGING.currencies
(
    date_update date NOT NULL,
    currency_code int NOT NULL,
    currency_code_with int NOT NULL,
    currency_with_div numeric(12, 4) NOT NULL
)
PARTITION BY (currencies.date_update)
GROUP BY calendar_hierarchy_day(currencies.date_update, 3, 2);

CREATE PROJECTION VT2609246827BE__STAGING.currencies_by_date
(
    date_update,
    currency_code,
    currency_code_with,
    currency_with_div
)
AS
SELECT
    currencies.date_update,
    currencies.currency_code,
    currencies.currency_code_with,
    currencies.currency_with_div
FROM VT2609246827BE__STAGING.currencies
ORDER BY currencies.date_update, currencies.currency_code
SEGMENTED BY hash(currencies.date_update, currencies.currency_code) ALL NODES;
