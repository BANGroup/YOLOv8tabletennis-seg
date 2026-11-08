delete from mart.f_sales fs
using mart.d_calendar dc
where fs.date_id = dc.date_id
  and dc.date_actual = '{{ ds }}'::date;

insert into mart.f_sales (
    date_id,
    item_id,
    customer_id,
    city_id,
    quantity,
    payment_amount,
    status,
    uniq_id
)
select
    dc.date_id,
    uol.item_id,
    uol.customer_id,
    uol.city_id,
    case
        when coalesce(uol.status, 'shipped') = 'refunded'
            then -abs(coalesce(uol.quantity, 0))
        else uol.quantity
    end as quantity,
    case
        when coalesce(uol.status, 'shipped') = 'refunded'
            then -abs(coalesce(uol.payment_amount, 0))
        else uol.payment_amount
    end as payment_amount,
    coalesce(uol.status, 'shipped') as status,
    uol.uniq_id
from staging.user_order_log uol
left join mart.d_calendar dc on uol.date_time::date = dc.date_actual
where uol.date_time::date = '{{ ds }}'::date;
