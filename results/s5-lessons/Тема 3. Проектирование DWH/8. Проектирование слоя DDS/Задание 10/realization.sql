alter table dds.fct_product_sales add constraint fct_product_sales_product_id_fkey
        foreign key (product_id) references dds.dm_products;

alter table dds.fct_product_sales add constraint fct_product_sales_order_id_fkey
        foreign key (order_id) references dds.dm_orders;