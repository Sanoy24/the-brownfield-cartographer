def load_daily_revenue(cur):
    cur.execute("""
        INSERT INTO marts.daily_revenue
        SELECT o.order_date, sum(o.amount)
        FROM staging.orders AS o
        GROUP BY o.order_date
    """)


def refresh_top_customers(cur):
    cur.executescript(
        "CREATE TABLE marts.top_customers AS SELECT customer_id FROM marts.customer_totals; "
        "DELETE FROM marts.stale_rows;"
    )


def run_maintenance(executor, query):
    executor.execute("vacuum-task")
    executor.execute(query)
