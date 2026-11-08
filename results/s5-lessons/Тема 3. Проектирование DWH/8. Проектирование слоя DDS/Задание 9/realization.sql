create table dds.fct_product_sales
(
    id            serial primary key,
    product_id    integer                  not null,
    order_id      integer                  not null,
    count         integer        default 0 not null
        constraint fct_product_sales_count_check
            check (count >= 0),
    price         numeric(14, 2) default 0 not null
        constraint fct_product_sales_price_check
            check (price >= (0)::numeric),
    total_sum     numeric(14, 2) default 0 not null
        constraint fct_product_sales_total_sum_check
            check (total_sum >= (0)::numeric),
    bonus_payment numeric(14, 2) default 0 not null
        constraint fct_product_sales_bonus_payment_check
            check (bonus_payment >= (0)::numeric),
    bonus_grant   numeric(14, 2) default 0 not null
        constraint fct_product_sales_bonus_grant_check
            check (bonus_grant >= (0)::numeric)
);