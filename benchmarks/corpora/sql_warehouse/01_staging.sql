CREATE TABLE staging.orders AS
SELECT id, customer_id, ordered_at FROM raw.orders;

CREATE TABLE staging.customers AS
SELECT id, name FROM raw.customers;
