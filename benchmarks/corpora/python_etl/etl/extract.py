import pandas as pd

from etl.queries import ACTIVE_USERS

ORDERS_SQL = "SELECT * FROM raw.orders WHERE created_at >= %(since)s"


def extract_active_users(conn):
    pd.read_sql(ACTIVE_USERS, conn).to_parquet("staging/active_users.parquet")


def extract_orders(conn, since):
    orders = pd.read_sql(ORDERS_SQL, conn, params={"since": since})
    orders.to_csv("staging/orders.csv", index=False)
