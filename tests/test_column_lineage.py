"""Column-level lineage: which source columns each written column comes from."""
from brownfield_cartographer.analyzers.sql_lineage import sql_statements
from brownfield_cartographer.orchestrator import Orchestrator
from brownfield_cartographer.query_api import QueryEngine


def _columns(sql):
    [statement] = sql_statements(sql, "job.sql")
    return ({c["target"]: (c["sources"], c["kind"]) for c in statement["columns"]},
            statement["unresolved_columns"])


def test_copies_renames_expressions_and_constants():
    columns, unresolved = _columns(
        "CREATE TABLE m.orders AS SELECT o.id AS order_id, c.name, o.amount * 100 AS cents, 'web' AS channel "
        "FROM raw.orders o JOIN raw.customers c ON c.id = o.customer_id")
    assert columns == {
        "order_id": (["raw.orders::id"], "copy"),
        "name": (["raw.customers::name"], "copy"),
        "cents": (["raw.orders::amount"], "derived"),
        "channel": ([], "derived"),
    }
    assert unresolved == []


def test_ctes_unions_and_insert_column_lists():
    columns, _ = _columns(
        "INSERT INTO m.events (event_id, source) "
        "WITH w AS (SELECT id FROM raw.web) SELECT id, 'web' FROM w UNION ALL SELECT id, 'app' FROM raw.app")
    assert columns["event_id"] == (["raw.app::id", "raw.web::id"], "copy")
    assert columns["source"] == ([], "derived")


def test_star_and_ambiguous_columns_are_unresolved_not_guessed():
    columns, unresolved = _columns(
        "CREATE TABLE m.t AS SELECT name, o.amount FROM raw.orders o JOIN raw.customers c ON c.id = o.customer_id")
    assert columns == {"amount": (["raw.orders::amount"], "copy")}
    assert unresolved == ["name"]
    columns, unresolved = _columns("CREATE TABLE m.copy AS SELECT * FROM raw.orders")
    assert columns == {} and unresolved == ["*"]


def test_column_queries_across_sql_python_and_dbt(tmp_path):
    files = {
        "stage.sql": "CREATE TABLE s.orders AS SELECT id, amount, amount * 0.2 AS tax FROM raw.orders;\n",
        "load.py": (
            "def load(cur):\n"
            "    cur.execute('INSERT INTO m.daily (order_id, gross) SELECT id, amount + tax FROM s.orders')\n"
        ),
        "shop/dbt_project.yml": "name: shop\n",
        "shop/models/summary.sql": (
            "{#- daily summary -#}\n"
            "select d.order_id, d.gross as revenue from {{ ref('daily') }} as d\n"
        ),
        "shop/models/looped.sql": "{% for c in ['a'] %}select {{ c }} from {{ ref('summary') }}{% endfor %}\n",
    }
    for path, text in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    result = Orchestrator().run(str(tmp_path))
    assert result["status"] == "complete", result["hydrologist"]["diagnostics"]
    engine = QueryEngine.open(tmp_path)

    gross = engine.trace_data_lineage("m.daily::gross")["data"]["results"]
    assert [(r["node"], r["distance"]) for r in gross] == [
        ("s.orders::amount", 1), ("s.orders::tax", 1), ("raw.orders::amount", 2)]
    assert gross[0]["evidence"]["path"] == "load.py"

    downstream = engine.trace_data_lineage("raw.orders::amount", direction="downstream")["data"]["results"]
    assert {r["node"] for r in downstream} >= {"s.orders::amount", "s.orders::tax", "m.daily::gross"}

    impact = {r["node"] for r in engine.analyze_impact("raw.orders::amount")["data"]["results"]}
    assert "m.daily::gross" in impact

    # dbt model with only ref(): columns resolved through the rendered SQL. Loops: skipped, counted.
    revenue = engine.trace_data_lineage("summary::revenue", depth=1)["data"]["results"]
    assert [r["node"] for r in revenue] == ["daily::gross"]
    assert result["hydrologist"]["column_lineage"]["skipped_templates"] == 1
    assert engine.get_architecture()["data"]["counts"]["columns"] > 0
