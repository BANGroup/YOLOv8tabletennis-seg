delete from mart.f_customer_retention fcr
where fcr.period_name = 'weekly'
  and fcr.period_id = (
      select week_of_year
      from mart.d_calendar
      where date_actual = '{{ ds }}'::date
  );

insert into mart.f_customer_retention (
    period_name,
    period_id,
    item_id,
    new_customers_count,
    returning_customers_count,
    refunded_customer_count,
    new_customers_revenue,
    returning_customers_revenue,
    customers_refunded
)
with target_week as (
    select week_of_year as period_id
    from mart.d_calendar
    where date_actual = '{{ ds }}'::date
),
order_lines as (
    select
        dc.week_of_year as period_id,
        fs.item_id,
        fs.customer_id,
        fs.uniq_id,
        fs.status,
        fs.payment_amount
    from mart.f_sales fs
    inner join mart.d_calendar dc on fs.date_id = dc.date_id
    inner join target_week tw on dc.week_of_year = tw.period_id
),
customer_stats as (
    select
        ol.period_id,
        ol.item_id,
        ol.customer_id,
        count(distinct ol.uniq_id) as orders_in_period,
        count(distinct ol.uniq_id) filter (where ol.status = 'refunded') as refund_orders,
        bool_or(ol.status = 'refunded') as customer_had_refund,
        sum(ol.payment_amount) as customer_revenue
    from order_lines ol
    group by ol.period_id, ol.item_id, ol.customer_id
),
aggregated as (
    select
        cs.period_id,
        cs.item_id,
        count(*) filter (where cs.orders_in_period = 1) as new_customers_count,
        count(*) filter (where cs.orders_in_period > 1) as returning_customers_count,
        count(*) filter (where cs.customer_had_refund) as refunded_customer_count,
        coalesce(
            sum(cs.customer_revenue) filter (where cs.orders_in_period = 1),
            0
        ) as new_customers_revenue,
        coalesce(
            sum(cs.customer_revenue) filter (where cs.orders_in_period > 1),
            0
        ) as returning_customers_revenue,
        coalesce(sum(cs.refund_orders), 0) as customers_refunded
    from customer_stats cs
    group by cs.period_id, cs.item_id
)
select
    'weekly',
    a.period_id,
    a.item_id,
    a.new_customers_count,
    a.returning_customers_count,
    a.refunded_customer_count,
    a.new_customers_revenue,
    a.returning_customers_revenue,
    a.customers_refunded
from aggregated a;
