alter table sales drop constraint sales_products_product_id_fk;

alter table products drop constraint products_pk;

alter table products add id serial;

alter table products add constraint products_pk primary key (id);

alter table products add valid_from timestamptz;

alter table products add valid_to timestamptz;

alter table sales add constraint sales_products_id_fk foreign key (product_id) references products;
