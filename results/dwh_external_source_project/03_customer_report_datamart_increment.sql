/* Инкрементальное обновление витрины dwh.customer_report_datamart */

BEGIN;

WITH
last_load AS (
    SELECT COALESCE(MAX(load_dttm), DATE '1900-01-01') AS load_dttm
    FROM dwh.load_dates_customer_report_datamart
),
dwh_delta AS (
    SELECT
        dcs.customer_id,
        dcs.customer_name,
        dcs.customer_address,
        dcs.customer_birthday,
        dcs.customer_email,
        fo.order_id,
        fo.craftsman_id,
        dp.product_id,
        dp.product_price,
        dp.product_type,
        fo.order_completion_date - fo.order_created_date AS diff_order_date,
        fo.order_status,
        TO_CHAR(fo.order_created_date, 'yyyy-mm') AS report_period,
        fo.load_dttm AS order_load_dttm,
        dc.load_dttm AS craftsman_load_dttm,
        dcs.load_dttm AS customer_load_dttm,
        dp.load_dttm AS product_load_dttm
    FROM dwh.f_order AS fo
    INNER JOIN dwh.d_craftsman AS dc
        ON dc.craftsman_id = fo.craftsman_id
    INNER JOIN dwh.d_customer AS dcs
        ON dcs.customer_id = fo.customer_id
    INNER JOIN dwh.d_product AS dp
        ON dp.product_id = fo.product_id
    CROSS JOIN last_load AS ll
    WHERE fo.load_dttm > ll.load_dttm
       OR dc.load_dttm > ll.load_dttm
       OR dcs.load_dttm > ll.load_dttm
       OR dp.load_dttm > ll.load_dttm
),
affected_grains AS (
    SELECT DISTINCT
        dd.customer_id,
        dd.report_period
    FROM dwh_delta AS dd
    WHERE dd.report_period IS NOT NULL
),
grain_orders AS (
    SELECT
        dcs.customer_id,
        dcs.customer_name,
        dcs.customer_address,
        dcs.customer_birthday,
        dcs.customer_email,
        fo.order_id,
        fo.craftsman_id,
        dp.product_price,
        dp.product_type,
        fo.order_completion_date - fo.order_created_date AS diff_order_date,
        fo.order_status,
        TO_CHAR(fo.order_created_date, 'yyyy-mm') AS report_period
    FROM dwh.f_order AS fo
    INNER JOIN dwh.d_craftsman AS dc
        ON dc.craftsman_id = fo.craftsman_id
    INNER JOIN dwh.d_customer AS dcs
        ON dcs.customer_id = fo.customer_id
    INNER JOIN dwh.d_product AS dp
        ON dp.product_id = fo.product_id
    INNER JOIN affected_grains AS ag
        ON ag.customer_id = fo.customer_id
       AND ag.report_period = TO_CHAR(fo.order_created_date, 'yyyy-mm')
),
agg_metrics AS (
    SELECT
        go.customer_id,
        go.report_period,
        MAX(go.customer_name) AS customer_name,
        MAX(go.customer_address) AS customer_address,
        MAX(go.customer_birthday) AS customer_birthday,
        MAX(go.customer_email) AS customer_email,
        SUM(go.product_price)::NUMERIC(15, 2) AS customer_money,
        (SUM(go.product_price) * 0.1)::BIGINT AS platform_money,
        COUNT(*)::BIGINT AS count_order,
        AVG(go.product_price)::NUMERIC(10, 2) AS avg_price_order,
        (
            PERCENTILE_CONT(0.5) WITHIN GROUP (
                ORDER BY go.diff_order_date
            ) FILTER (WHERE go.diff_order_date IS NOT NULL)
        )::NUMERIC(10, 1) AS median_time_order_completed,
        SUM(CASE WHEN go.order_status = 'created' THEN 1 ELSE 0 END)::BIGINT AS count_order_created,
        SUM(CASE WHEN go.order_status IN ('in progress', 'in-progress') THEN 1 ELSE 0 END)::BIGINT AS count_order_in_progress,
        SUM(CASE WHEN go.order_status = 'delivery' THEN 1 ELSE 0 END)::BIGINT AS count_order_delivery,
        SUM(CASE WHEN go.order_status = 'done' THEN 1 ELSE 0 END)::BIGINT AS count_order_done,
        SUM(CASE WHEN go.order_status <> 'done' THEN 1 ELSE 0 END)::BIGINT AS count_order_not_done
    FROM grain_orders AS go
    GROUP BY go.customer_id, go.report_period
),
product_rank AS (
    SELECT
        go.customer_id,
        go.report_period,
        go.product_type,
        COUNT(*) AS category_count,
        ROW_NUMBER() OVER (
            PARTITION BY go.customer_id, go.report_period
            ORDER BY COUNT(*) DESC, go.product_type ASC
        ) AS rn
    FROM grain_orders AS go
    GROUP BY go.customer_id, go.report_period, go.product_type
),
top_product AS (
    SELECT
        pr.customer_id,
        pr.report_period,
        pr.product_type AS top_product_category
    FROM product_rank AS pr
    WHERE pr.rn = 1
),
craftsman_rank AS (
    SELECT
        go.customer_id,
        go.report_period,
        go.craftsman_id,
        COUNT(*) AS craftsman_count,
        ROW_NUMBER() OVER (
            PARTITION BY go.customer_id, go.report_period
            ORDER BY COUNT(*) DESC, go.craftsman_id ASC
        ) AS rn
    FROM grain_orders AS go
    GROUP BY go.customer_id, go.report_period, go.craftsman_id
),
top_craftsman AS (
    SELECT
        cr.customer_id,
        cr.report_period,
        cr.craftsman_id AS top_craftsman_id
    FROM craftsman_rank AS cr
    WHERE cr.rn = 1
),
datamart_result AS (
    SELECT
        am.customer_id,
        am.customer_name,
        am.customer_address,
        am.customer_birthday,
        am.customer_email,
        am.customer_money,
        am.platform_money,
        am.count_order,
        am.avg_price_order,
        am.median_time_order_completed,
        tp.top_product_category,
        tc.top_craftsman_id,
        am.count_order_created,
        am.count_order_in_progress,
        am.count_order_delivery,
        am.count_order_done,
        am.count_order_not_done,
        am.report_period
    FROM agg_metrics AS am
    INNER JOIN top_product AS tp
        ON tp.customer_id = am.customer_id
       AND tp.report_period = am.report_period
    INNER JOIN top_craftsman AS tc
        ON tc.customer_id = am.customer_id
       AND tc.report_period = am.report_period
),
upserted AS (
    INSERT INTO dwh.customer_report_datamart (
        customer_id,
        customer_name,
        customer_address,
        customer_birthday,
        customer_email,
        customer_money,
        platform_money,
        count_order,
        avg_price_order,
        median_time_order_completed,
        top_product_category,
        top_craftsman_id,
        count_order_created,
        count_order_in_progress,
        count_order_delivery,
        count_order_done,
        count_order_not_done,
        report_period
    )
    SELECT
        dr.customer_id,
        dr.customer_name,
        dr.customer_address,
        dr.customer_birthday,
        dr.customer_email,
        dr.customer_money,
        dr.platform_money,
        dr.count_order,
        dr.avg_price_order,
        dr.median_time_order_completed,
        dr.top_product_category,
        dr.top_craftsman_id,
        dr.count_order_created,
        dr.count_order_in_progress,
        dr.count_order_delivery,
        dr.count_order_done,
        dr.count_order_not_done,
        dr.report_period
    FROM datamart_result AS dr
    ON CONFLICT ON CONSTRAINT customer_report_datamart_customer_id_report_period_uq
    DO UPDATE SET
        customer_name = EXCLUDED.customer_name,
        customer_address = EXCLUDED.customer_address,
        customer_birthday = EXCLUDED.customer_birthday,
        customer_email = EXCLUDED.customer_email,
        customer_money = EXCLUDED.customer_money,
        platform_money = EXCLUDED.platform_money,
        count_order = EXCLUDED.count_order,
        avg_price_order = EXCLUDED.avg_price_order,
        median_time_order_completed = EXCLUDED.median_time_order_completed,
        top_product_category = EXCLUDED.top_product_category,
        top_craftsman_id = EXCLUDED.top_craftsman_id,
        count_order_created = EXCLUDED.count_order_created,
        count_order_in_progress = EXCLUDED.count_order_in_progress,
        count_order_delivery = EXCLUDED.count_order_delivery,
        count_order_done = EXCLUDED.count_order_done,
        count_order_not_done = EXCLUDED.count_order_not_done
    RETURNING customer_id, report_period, (xmax = 0) AS is_insert
),
watermark_insert AS (
    INSERT INTO dwh.load_dates_customer_report_datamart (load_dttm)
    SELECT COALESCE(
        GREATEST(
            MAX(dd.order_load_dttm::date),
            MAX(dd.craftsman_load_dttm::date),
            MAX(dd.customer_load_dttm::date),
            MAX(dd.product_load_dttm::date)
        ),
        CURRENT_DATE
    )
    FROM dwh_delta AS dd
    WHERE EXISTS (SELECT 1 FROM dwh_delta)
    RETURNING load_dttm
)
SELECT
    (SELECT COUNT(*) FROM dwh_delta) AS delta_rows,
    (SELECT COUNT(*) FROM affected_grains) AS affected_grains,
    (SELECT COUNT(*) FROM upserted) AS affected_rows,
    (SELECT COUNT(*) FILTER (WHERE is_insert) FROM upserted) AS inserted_rows,
    (SELECT COUNT(*) FILTER (WHERE NOT is_insert) FROM upserted) AS updated_rows,
    (SELECT load_dttm FROM watermark_insert) AS new_watermark;

COMMIT;
