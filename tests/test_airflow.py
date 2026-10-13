"""Airflow DAGs: tasks, scheduling order, what each task runs, and impact through them."""
import textwrap

from brownfield_cartographer.analyzers.airflow_dag import parse_airflow
from brownfield_cartographer.orchestrator import Orchestrator
from brownfield_cartographer.query_api import QueryEngine

DAG_FILE = textwrap.dedent('''
    from airflow import DAG
    from airflow.decorators import task
    from airflow.models.baseoperator import chain
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import PythonOperator
    from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

    from etl.clean import clean_orders


    def notify():
        return "done"


    with DAG("orders_daily", schedule="@daily") as dag:
        extract = BashOperator(task_id="extract", bash_command="python etl/extract.py --day {{ ds }}")
        load = SQLExecuteQueryOperator(task_id="load", sql="queries/load_orders.sql")
        clean = PythonOperator(task_id="clean", python_callable=clean_orders)
        aggregate = SQLExecuteQueryOperator(
            task_id="aggregate",
            sql="INSERT INTO marts.daily_orders SELECT day, count(*) FROM staging.orders GROUP BY day",
        )
        report = PythonOperator(task_id="report", python_callable=notify)
        audit = BashOperator(task_id="audit", bash_command="echo audit")
        publish = BashOperator(task_id="publish", bash_command="echo publish")

        @task
        def summarize():
            return 1

        @task(task_id="email_summary")
        def email(value):
            return value

        extract >> load >> [clean, aggregate]
        report << aggregate
        clean.set_downstream(audit)
        chain(audit, publish)
        email(summarize())

        for name in ["a", "b"]:
            BashOperator(task_id=f"dynamic_{name}", bash_command="echo") >> publish
''')

KNOWN = {"dags/orders.py", "etl/extract.py", "etl/clean.py", "queries/load_orders.sql"}


def _short(node):
    return node.split("::", 1)[1]


def test_tasks_and_scheduling_order_with_line_evidence():
    result = parse_airflow(DAG_FILE, "dags/orders.py", KNOWN)
    tasks = {_short(t["id"]): t for t in result["tasks"]}
    assert set(tasks) == {f"orders_daily.{name}" for name in (
        "extract", "load", "clean", "aggregate", "report", "audit", "publish", "summarize", "email_summary")}
    assert tasks["orders_daily.load"]["operator"] == "SQLExecuteQueryOperator"
    assert tasks["orders_daily.summarize"]["operator"] == "@task"
    schedules = {(_short(u), _short(d)) for u, d, _ in result["schedules"]}
    assert schedules == {
        ("orders_daily.extract", "orders_daily.load"),
        ("orders_daily.load", "orders_daily.clean"),
        ("orders_daily.load", "orders_daily.aggregate"),
        ("orders_daily.aggregate", "orders_daily.report"),
        ("orders_daily.clean", "orders_daily.audit"),
        ("orders_daily.audit", "orders_daily.publish"),
        ("orders_daily.summarize", "orders_daily.email_summary"),
    }
    lines = {(_short(u), _short(d)): span for u, d, span in result["schedules"]}
    assert lines[("orders_daily.extract", "orders_daily.load")] == (36, 36)


def test_what_each_task_runs():
    result = parse_airflow(DAG_FILE, "dags/orders.py", KNOWN)
    runs = {(_short(task), target) for task, target, _ in result["runs"]}
    assert runs == {
        ("orders_daily.extract", "etl/extract.py"),
        ("orders_daily.load", "queries/load_orders.sql"),
        ("orders_daily.clean", "etl/clean.py::clean_orders"),
        ("orders_daily.report", "dags/orders.py::notify"),
        ("orders_daily.summarize", "dags/orders.py::summarize"),
        ("orders_daily.email_summary", "dags/orders.py::email"),
    }
    [(task, inline)] = result["transformations"]
    assert _short(task) == "orders_daily.aggregate"
    assert (inline.source_datasets, inline.target_datasets) == (["staging.orders"], ["marts.daily_orders"])
    assert inline.analysis_method.startswith("airflow_sql:")


def test_dynamic_tasks_are_reported_not_guessed():
    result = parse_airflow(DAG_FILE, "dags/orders.py", KNOWN)
    assert any("dynamic" in d["error"].lower() and d["line"] == 43 for d in result["diagnostics"])
    assert not any("dynamic_" in t["id"] for t in result["tasks"])


def test_non_airflow_files_are_ignored():
    assert parse_airflow("import pandas\n", "etl/x.py", set()) is None


def test_tasks_are_not_datasets_and_impact_flows_through_them(tmp_path):
    files = {
        "dags/orders.py": DAG_FILE,
        "etl/__init__.py": "",
        "etl/extract.py": "print('extract')\n",
        "etl/clean.py": "def clean_orders():\n    return 1\n",
        "queries/load_orders.sql": "INSERT INTO staging.orders SELECT * FROM raw.orders;\n",
    }
    for path, text in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    Orchestrator().run(str(tmp_path))
    engine = QueryEngine.open(tmp_path)
    graph = engine.graph.graph
    datasets = {n for n, d in graph.nodes(data=True) if d.get("node_type") == "dataset"}
    assert not datasets & {"extract", "load", "clean", "aggregate", "report"}
    impact = {row["node"] for row in engine.analyze_impact("queries/load_orders.sql")["data"]["results"]}
    assert {"task:dags/orders.py::orders_daily.load", "task:dags/orders.py::orders_daily.clean",
            "task:dags/orders.py::orders_daily.audit", "task:dags/orders.py::orders_daily.publish",
            "staging.orders"} <= impact
    load = "task:dags/orders.py::orders_daily.load"
    downstream = engine.trace_dependencies(load, direction="dependents")["data"]["results"]
    assert {r["node"].split("::")[1] for r in downstream} == {
        "orders_daily.clean", "orders_daily.aggregate", "orders_daily.report",
        "orders_daily.audit", "orders_daily.publish"}
    upstream = engine.trace_dependencies(load, direction="dependencies")["data"]["results"]
    assert [r["node"].split("::")[1] for r in upstream] == ["orders_daily.extract"]
    found = engine.find_symbol("orders_daily.load")["data"]["matches"]
    assert found[0]["type"] == "task" and found[0]["evidence"]["line_range"] == [18, 18]
    assert engine.get_architecture()["data"]["counts"]["tasks"] == 9


def test_computed_dag_ids_use_the_file_name_idiom_or_are_reported():
    source = textwrap.dedent('''
        import os
        from pathlib import Path
        from airflow.decorators import dag, task

        @dag(dag_id=Path(__file__).stem)
        def generate_dag():
            @task
            def deps():
                return 1
            deps()

        @dag(dag_id=os.environ["DAG_NAME"])
        def other():
            @task
            def step():
                return 1
            step()
    ''')
    result = parse_airflow(source, "dags/dag_dbt.py", set())
    ids = {_short(t["id"]) for t in result["tasks"]}
    assert ids == {"dag_dbt.deps", "_.step"}
    assert any("dag_id" in d["error"] and d["line"] == 13 for d in result["diagnostics"])
