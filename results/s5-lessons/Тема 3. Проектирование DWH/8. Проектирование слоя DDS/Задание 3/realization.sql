create table dds.dm_restaurants
(
    id              serial primary key,
    restaurant_id   varchar   not null,
    restaurant_name varchar   not null,
    active_from     timestamp not null,
    active_to       timestamp not null
);