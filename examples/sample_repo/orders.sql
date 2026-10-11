CREATE TABLE analytics.orders AS
SELECT order_id, customer_id, amount
FROM raw.orders;
