SELECT COUNT(*) = 0
FROM user_activity_log
WHERE date_time IS NULL
   OR action_id IS NULL
   OR customer_id IS NULL
   OR quantity IS NULL;
