from datetime import datetime

from airflow.decorators import dag, task


@dag(dag_id="weekly_reports", start_date=datetime(2024, 1, 1), schedule="@weekly")
def reports():
    @task
    def collect():
        return ["a", "b"]

    @task(task_id="render_pdf")
    def render(items):
        return len(items)

    @task
    def deliver(count):
        return count

    deliver(render(collect()))


reports()
