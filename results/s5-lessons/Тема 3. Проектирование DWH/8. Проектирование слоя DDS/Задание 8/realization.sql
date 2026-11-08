alter table dds.dm_orders add constraint dm_orders_restaurant_id_fkey
        foreign key (restaurant_id) references dds.dm_restaurants;

alter table dds.dm_orders add constraint dm_orders_timestamp_id_fkey
        foreign key (timestamp_id) references dds.dm_timestamps;

alter table dds.dm_orders add constraint dm_orders_user_id_fkey
        foreign key (user_id) references dds.dm_users;