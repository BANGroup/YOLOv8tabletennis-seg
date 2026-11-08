create table dds.dm_products
(
    id            serial primary key,
		restaurant_id integer                  not null,
    product_id    varchar                  not null,
    product_name  varchar                  not null,
    product_price numeric(14, 2) default 0 not null
        constraint dm_products_product_price_check
            check (product_price >= (0)::numeric),
    active_from   timestamp                not null,
    active_to     timestamp                not null
);