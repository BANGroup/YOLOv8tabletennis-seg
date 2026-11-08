SELECT DISTINCT product_payment::json->>'product_name' AS product_name 
FROM (
    SELECT json_array_elements(product_payments::json) AS product_payment  
    FROM (
        SELECT event_value::json->>'product_payments' AS product_payments 
        FROM outbox 
        WHERE event_value::json->>'product_payments' IS NOT NULL
    ) AS product_payment_jsons
) AS product_payments 
ORDER BY product_name ASC;
