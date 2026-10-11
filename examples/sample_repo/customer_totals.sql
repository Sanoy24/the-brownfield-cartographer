CREATE TABLE analytics.customer_totals AS
SELECT customer_id, SUM(amount) AS total
FROM analytics.orders
GROUP BY customer_id;
