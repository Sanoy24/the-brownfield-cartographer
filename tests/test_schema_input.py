"""Schemas from the repository resolve `SELECT *` over tables and unqualified columns in joins."""
import json

from brownfield_cartographer.analyzers.sql_lineage import known_schema, sql_statements
from brownfield_cartographer.graph.knowledge_graph import KnowledgeGraph
from brownfield_cartographer.orchestrator import Orchestrator


def _columns(sql, tables):
    with known_schema(tables):
        [statement] = sql_statements(sql, dialect="duckdb")
    return {c["target"]: c["sources"] for c in statement["columns"]}, statement["unresolved_columns"]


def test_known_tables_resolve_join_columns_and_expand_stars():
    tables = {"s.orders": ["id", "customer_id", "amount"], "s.customers": ["id", "name"]}
    columns, unresolved = _columns(
        "CREATE TABLE m.x AS SELECT c.name, amount FROM s.orders o JOIN s.customers c ON o.customer_id = c.id",
        tables)
    assert columns == {"name": ["s.customers::name"], "amount": ["s.orders::amount"]} and unresolved == []
    columns, unresolved = _columns("CREATE TABLE m.y AS SELECT * FROM s.customers", tables)
    assert columns == {"id": ["s.customers::id"], "name": ["s.customers::name"]} and unresolved == []
    columns, _ = _columns("CREATE TABLE m.z AS SELECT o.*, c.name FROM s.orders o JOIN s.customers c ON o.customer_id = c.id",
                          tables)
    assert set(columns) == {"id", "customer_id", "amount", "name"}


def test_ambiguous_or_unknown_columns_stay_unresolved():
    tables = {"s.orders": ["id", "amount"], "s.customers": ["id", "name"]}
    # `id` is in both tables.
    _, unresolved = _columns("CREATE TABLE m.x AS SELECT id FROM s.orders o JOIN s.customers c ON o.id = c.id", tables)
    assert unresolved == ["id"]
    # raw.events has no known columns, so `amount` could come from either side.
    _, unresolved = _columns("CREATE TABLE m.x AS SELECT amount FROM s.orders o JOIN raw.events e ON o.id = e.id", tables)
    assert unresolved == ["amount"]
    # Joined stars with a shared column name are not expanded.
    _, unresolved = _columns("CREATE TABLE m.x AS SELECT * FROM s.orders o JOIN s.customers c ON o.id = c.id", tables)
    assert unresolved == ["*"]


def _derives(tmp_path, files):
    for path, text in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    result = Orchestrator().run(str(tmp_path))
    graph = KnowledgeGraph()
    graph.load_index(tmp_path / ".cartography" / tmp_path.name / "index.json")
    edges = {(u, v) for u, v, d in graph.graph.edges(data=True) if d.get("edge_type") == "DERIVES"}
    return edges, result


def test_schemas_come_from_ddl_and_from_statements_that_define_every_column(tmp_path):
    edges, result = _derives(tmp_path, {
        "ddl.sql": "CREATE TABLE raw.orders (id INT, customer_id INT, amount INT);\n",
        "stage.sql": "CREATE TABLE s.customers AS SELECT id, name FROM raw.customers;\n",
        # Columns written by an INSERT are a subset of the table, never its schema.
        "partial.sql": "INSERT INTO s.events (id) SELECT id FROM raw.clicks;\n",
        "mart.sql": (
            "CREATE TABLE m.revenue AS SELECT c.name, amount FROM raw.orders o "
            "JOIN s.customers c ON o.customer_id = c.id;\n"
            "CREATE TABLE m.copy AS SELECT * FROM s.customers;\n"
            "CREATE TABLE m.events AS SELECT * FROM s.events;\n"
        ),
    })
    assert ("raw.orders::amount", "m.revenue::amount") in edges
    assert ("s.customers::name", "m.copy::name") in edges
    assert not any(target.startswith("m.events::") for _, target in edges)
    assert result["hydrologist"]["column_lineage"]["schema_tables"] == 4


def test_dbt_catalog_supplies_columns_for_sources(tmp_path):
    catalog = {"metadata": {"dbt_schema_version": "https://schemas.getdbt.com/dbt/catalog/v1.json"},
               "nodes": {}, "sources": {"source.shop.raw.orders": {
                   "metadata": {"database": "db", "schema": "raw", "name": "orders"},
                   "columns": {"ID": {"name": "ID"}, "AMOUNT": {"name": "AMOUNT"}}}}}
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    (tmp_path / "mart.sql").write_text(
        "CREATE TABLE m.x AS SELECT * FROM raw.orders;\n", encoding="utf-8")
    from brownfield_cartographer.scanner import ScanSettings
    Orchestrator(scan_settings=ScanSettings(dbt_catalog="target/catalog.json")).run(str(tmp_path))
    graph = KnowledgeGraph()
    graph.load_index(tmp_path / ".cartography" / tmp_path.name / "index.json")
    edges = {(u, v) for u, v, d in graph.graph.edges(data=True) if d.get("edge_type") == "DERIVES"}
    assert {("raw.orders::id", "m.x::id"), ("raw.orders::amount", "m.x::amount")} <= edges


def test_star_except_is_not_expanded_and_using_joins_keep_their_lineage():
    tables = {"dim": ["id", "mode", "name"], "matched": ["participant_id", "txn_id", "off_id"]}
    columns, _ = _columns(
        "CREATE TABLE out AS SELECT * EXCEPT (mode), upper(mode) AS mode FROM dim", tables)
    assert columns["mode"] == ["dim::mode"]
    # sqlglot cannot trace a USING join when only one side has known columns;
    # the result without the schema is kept rather than losing the columns.
    columns, unresolved = _columns(
        "CREATE TABLE out AS WITH dt AS (SELECT * FROM unioned), pairs AS (SELECT * FROM matched) "
        "SELECT pairs.participant_id, tap_on.customer_id AS customer_id_1 "
        "FROM pairs LEFT JOIN dt AS tap_on USING (txn_id)", tables)
    assert columns == {"participant_id": ["matched::participant_id"], "customer_id_1": ["unioned::customer_id"]}
    assert unresolved == []
