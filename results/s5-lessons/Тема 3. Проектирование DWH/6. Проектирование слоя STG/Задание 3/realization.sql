alter table stg.ordersystem_orders add constraint ordersystem_orders_object_id_uindex
	unique (object_id);

alter table stg.ordersystem_users add constraint ordersystem_users_object_id_uindex
	unique (object_id);

alter table stg.ordersystem_restaurants add constraint ordersystem_restaurants_object_id_uindex
	unique (object_id);
