CREATE VIEW marts.top_customers AS
SELECT customer_id
FROM (SELECT id AS customer_id, order_count FROM marts.customer_orders) AS ranked
WHERE order_count > 10;
