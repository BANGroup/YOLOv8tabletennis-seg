DELETE FROM VT2609246827BE__DWH.global_metrics
WHERE date_update = DATE '__LOAD_DATE__';

INSERT INTO VT2609246827BE__DWH.global_metrics (
    date_update,
    currency_from,
    amount_total,
    cnt_transactions,
    avg_transactions_per_account,
    cnt_accounts_make_transactions
)
WITH done_tx AS (
    SELECT
        operation_id,
        account_number_to,
        currency_code,
        amount,
        transaction_dt::date AS date_update
    FROM (
        SELECT
            operation_id,
            account_number_to,
            currency_code,
            amount,
            transaction_dt,
            ROW_NUMBER() OVER (
                PARTITION BY operation_id
                ORDER BY transaction_dt DESC
            ) AS rn
        FROM VT2609246827BE__STAGING.transactions
        WHERE transaction_dt::date = DATE '__LOAD_DATE__'
          AND status = 'done'
          AND account_number_from >= 0
          AND account_number_to >= 0
    ) ranked
    WHERE rn = 1
),
usd_rate AS (
    SELECT
        date_update,
        currency_code,
        MAX(currency_with_div) AS currency_with_div
    FROM VT2609246827BE__STAGING.currencies
    WHERE currency_code_with = 420
      AND date_update = DATE '__LOAD_DATE__'
    GROUP BY date_update, currency_code
)
SELECT
    done_tx.date_update,
    done_tx.currency_code AS currency_from,
    SUM(
        done_tx.amount * COALESCE(usd_rate.currency_with_div, 1) / 100
    )::numeric(18, 2) AS amount_total,
    COUNT(done_tx.operation_id) AS cnt_transactions,
    (
        COUNT(done_tx.operation_id)::numeric
        / COUNT(DISTINCT done_tx.account_number_to)
    )::numeric(18, 2) AS avg_transactions_per_account,
    COUNT(DISTINCT done_tx.account_number_to) AS cnt_accounts_make_transactions
FROM done_tx
LEFT JOIN usd_rate
    ON usd_rate.currency_code = done_tx.currency_code
   AND usd_rate.date_update = done_tx.date_update
GROUP BY done_tx.date_update, done_tx.currency_code;
