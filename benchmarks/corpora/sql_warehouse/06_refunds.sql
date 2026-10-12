UPDATE marts.order_summary
SET recent_orders = recent_orders - 1
FROM raw.refunds AS r
WHERE r.customer_id = marts.order_summary.customer_id;
