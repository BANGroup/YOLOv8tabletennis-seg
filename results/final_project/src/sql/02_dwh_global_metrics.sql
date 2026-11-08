DROP TABLE IF EXISTS VT2609246827BE__DWH.global_metrics CASCADE;

CREATE TABLE VT2609246827BE__DWH.global_metrics
(
    date_update date NOT NULL,
    currency_from int NOT NULL,
    amount_total numeric(18, 2) NOT NULL,
    cnt_transactions int NOT NULL,
    avg_transactions_per_account numeric(18, 2) NOT NULL,
    cnt_accounts_make_transactions int NOT NULL
)
PARTITION BY (global_metrics.date_update)
GROUP BY calendar_hierarchy_day(global_metrics.date_update, 3, 2);

CREATE PROJECTION VT2609246827BE__DWH.global_metrics_by_date
(
    date_update,
    currency_from,
    amount_total,
    cnt_transactions,
    avg_transactions_per_account,
    cnt_accounts_make_transactions
)
AS
SELECT
    global_metrics.date_update,
    global_metrics.currency_from,
    global_metrics.amount_total,
    global_metrics.cnt_transactions,
    global_metrics.avg_transactions_per_account,
    global_metrics.cnt_accounts_make_transactions
FROM VT2609246827BE__DWH.global_metrics
ORDER BY global_metrics.date_update, global_metrics.currency_from
SEGMENTED BY hash(global_metrics.date_update, global_metrics.currency_from) ALL NODES;
