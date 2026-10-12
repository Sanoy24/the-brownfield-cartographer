CREATE TABLE marts.order_summary AS
WITH recent AS (
    SELECT * FROM staging.orders WHERE ordered_at > '2024-01-01'
)
SELECT customer_id, count(*) AS recent_orders FROM recent GROUP BY customer_id;
