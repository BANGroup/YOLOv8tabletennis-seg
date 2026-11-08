-- STG ordersystem (MongoDB → PostgreSQL), Задание 2
-- Дата: 2026-09-23

create table if not exists stg.ordersystem_orders (
    id serial primary key,
    object_id varchar not null,
    object_value text not null,
    update_ts timestamp not null
);

create table if not exists stg.ordersystem_restaurants (
    id serial primary key,
    object_id varchar not null,
    object_value text not null,
    update_ts timestamp not null
);

create table if not exists stg.ordersystem_users (
    id serial primary key,
    object_id varchar not null,
    object_value text not null,
    update_ts timestamp not null
);

create unique index if not exists ordersystem_orders_object_id_uindex
    on stg.ordersystem_orders (object_id);

create unique index if not exists ordersystem_restaurants_object_id_uindex
    on stg.ordersystem_restaurants (object_id);

create unique index if not exists ordersystem_users_object_id_uindex
    on stg.ordersystem_users (object_id);
