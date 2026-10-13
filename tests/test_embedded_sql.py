"""SQL embedded in Python: reads, statement-level writes, constants, and honest unknowns."""
import textwrap

import pytest

from brownfield_cartographer.orchestrator import Orchestrator
from brownfield_cartographer.query_api import QueryEngine


def _analyze(tmp_path, source):
    (tmp_path / "job.py").write_text(textwrap.dedent(source), encoding="utf-8")
    result = Orchestrator().run(str(tmp_path))
    engine = QueryEngine.open(tmp_path)
    graph = engine.graph.graph
    edges = set()
    for tx, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "transformation":
            continue
        reads = [u for u, _, d in graph.in_edges(tx, data=True) if d["edge_type"] == "CONSUMES"]
        writes = [v for _, v, d in graph.out_edges(tx, data=True) if d["edge_type"] == "PRODUCES"]
        edges.update((r, w) for r in reads for w in writes)
    diagnostics = [row for stage in engine.get_architecture()["coverage"]["diagnostics"].values() for row in stage]
    return result["status"], edges, diagnostics, graph


def test_select_read_joins_python_write_in_same_scope(tmp_path):
    status, edges, _, _ = _analyze(tmp_path, """
        import pandas as pd

        def export(conn):
            df = pd.read_sql("SELECT id, total FROM raw.orders WHERE total > 0", conn)
            df.to_parquet("out/orders.parquet")
    """)
    assert status == "complete"
    assert edges == {("raw.orders", "out/orders.parquet")}


def test_insert_select_is_an_exact_statement_level_transformation(tmp_path):
    status, edges, _, graph = _analyze(tmp_path, """
        def load(cursor):
            cursor.execute(
                "INSERT INTO marts.daily SELECT day, count(*) FROM staging.orders o "
                "JOIN staging.days d ON d.day = o.day WHERE o.region = %s GROUP BY day",
                ("eu",),
            )
    """)
    assert status == "complete"
    assert edges == {("staging.orders", "marts.daily"), ("staging.days", "marts.daily")}
    tx = next(attrs for _, attrs in graph.nodes(data=True) if attrs.get("node_type") == "transformation")
    assert tx["analysis_method"] == "embedded_sql"
    assert (tx["line_start"], tx["line_end"]) == (3, 7)


def test_module_constant_text_wrapper_and_spark_sql(tmp_path):
    status, edges, _, _ = _analyze(tmp_path, '''
        from sqlalchemy import text

        CUSTOMERS = """
            SELECT * FROM raw.customers
        """

        def customers(spark, session):
            frame = spark.sql(CUSTOMERS)
            frame.write.saveAsTable("analytics.customers")
            session.execute(text("CREATE TABLE analytics.vip AS SELECT * FROM analytics.customers WHERE tier = :tier"))
    ''')
    assert status == "complete"
    assert edges == {("raw.customers", "analytics.customers"), ("analytics.customers", "analytics.vip")}


def test_read_sql_table_and_multi_statement_scripts(tmp_path):
    status, edges, _, _ = _analyze(tmp_path, """
        import pandas as pd

        def copy(conn, con):
            pd.read_sql_table("raw.events", conn).to_csv("events.csv")
            con.executescript("CREATE TABLE a.x AS SELECT * FROM b.y; INSERT INTO a.z SELECT * FROM a.x;")
    """)
    assert status == "complete"
    assert edges == {("raw.events", "events.csv"), ("b.y", "a.x"), ("a.x", "a.z")}


def test_dynamic_sql_is_reported_not_guessed(tmp_path):
    status, edges, diagnostics, _ = _analyze(tmp_path, """
        def dump(cursor, table):
            cursor.execute(f"SELECT * FROM {table}")
            cursor.execute("SELECT * FROM " + table)
    """)
    assert status == "complete"
    assert edges == set()
    unresolved = [d for d in diagnostics if d.get("path") == "job.py" and "Dynamic SQL" in d["error"]]
    assert [d["line"] for d in unresolved] == [3, 4]
    assert not any(d.get("incomplete") for d in unresolved)


def test_non_sql_strings_are_ignored(tmp_path):
    status, edges, diagnostics, _ = _analyze(tmp_path, """
        def run(executor, db):
            executor.execute("nightly-task")
            db.execute("Selection of the best rows")
    """)
    assert status == "complete"
    assert edges == set()
    assert diagnostics == []


@pytest.mark.parametrize("statement", ["SELECT FROM WHERE", "DELETE FROM WHERE"])
def test_unsupported_or_invalid_sql_keeps_other_lineage_and_makes_coverage_partial(tmp_path, statement):
    status, edges, diagnostics, _ = _analyze(tmp_path, f"""
        import pandas as pd

        def job(cursor):
            pd.read_csv("raw.csv").to_csv("clean.csv")
            cursor.execute("{statement}")
    """)
    assert status == "partial"
    assert edges == {("raw.csv", "clean.csv")}
    assert any(d.get("path") == "job.py" and d.get("line") == 6 and d.get("incomplete") for d in diagnostics)


def test_supported_statements_survive_an_unsupported_one_in_the_same_script(tmp_path):
    status, edges, diagnostics, _ = _analyze(tmp_path, """
        def refresh(cur):
            cur.executescript("CREATE TABLE a.top AS SELECT id FROM a.totals; DELETE FROM WHERE; INSERT INTO a.log SELECT * FROM a.top;")
    """)
    assert status == "partial"
    assert edges == {("a.totals", "a.top"), ("a.top", "a.log")}
    assert any("Invalid SQL" in d["error"] and d.get("incomplete") for d in diagnostics)


def test_spark_dataframe_writer_paths_are_writes(tmp_path):
    status, edges, _, _ = _analyze(tmp_path, """
        def export(spark):
            spark.table("marts.totals").write.parquet("exports/totals")
            spark.read.csv("raw/events.csv").write.mode("overwrite").json("exports/events")
    """)
    assert status == "complete"
    assert edges == {("marts.totals", "exports/totals"), ("raw/events.csv", "exports/totals"),
                     ("marts.totals", "exports/events"), ("raw/events.csv", "exports/events")}
