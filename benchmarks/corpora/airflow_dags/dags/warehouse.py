from datetime import datetime

from airflow import DAG
from airflow.models.baseoperator import chain
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

from pipelines.ingest import fetch_customers, fetch_orders

dag = DAG("warehouse", start_date=datetime(2024, 1, 1), template_searchpath=["dags/sql"])

start = EmptyOperator(task_id="start", dag=dag)
orders = PythonOperator(task_id="fetch_orders", python_callable=fetch_orders, dag=dag)
customers = PythonOperator(task_id="fetch_customers", python_callable=fetch_customers, dag=dag)
build_customers = SQLExecuteQueryOperator(task_id="build_customers", sql="build_customers.sql", dag=dag)
build_orders = SQLExecuteQueryOperator(
    task_id="build_orders",
    sql="""
        INSERT INTO marts.orders
        SELECT o.*, c.segment FROM staging.orders AS o JOIN marts.customers AS c ON c.id = o.customer_id
    """,
    dag=dag,
)
done = EmptyOperator(task_id="done", dag=dag)

start >> [orders, customers]
customers >> build_customers
chain([orders, build_customers], build_orders, done)

for region in ("eu", "us"):
    EmptyOperator(task_id=f"notify_{region}", dag=dag) << done
