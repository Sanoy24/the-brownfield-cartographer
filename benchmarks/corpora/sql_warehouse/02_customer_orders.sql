INSERT INTO marts.customer_orders
SELECT c.id, count(*) AS order_count
FROM staging.customers AS c
JOIN staging.orders AS o ON o.customer_id = c.id
GROUP BY c.id;
