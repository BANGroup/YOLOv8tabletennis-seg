SELECT COUNT(*) = 0
FROM user_order_log
WHERE date_time IS NULL
   OR city_id IS NULL
   OR city_name IS NULL
   OR customer_id IS NULL
   OR first_name IS NULL
   OR last_name IS NULL
   OR item_id IS NULL
   OR item_name IS NULL
   OR quantity IS NULL
   OR payment_amount IS NULL;
