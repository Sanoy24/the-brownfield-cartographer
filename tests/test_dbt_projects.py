"""dbt projects anywhere in a repository, with repository-relative evidence."""
from brownfield_cartographer.orchestrator import Orchestrator
from brownfield_cartographer.query_api import QueryEngine


def _write(root, files):
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def _upstream(engine, dataset):
    results = engine.trace_data_lineage(dataset)["data"]["results"]
    return {row["node"] for row in results if row["node_type"] == "dataset"}


def test_nested_project_lineage_with_repository_relative_evidence(tmp_path):
    _write(tmp_path, {
        "analytics/dbt/dbt_project.yml": "name: shop\nmodel-paths: [transform]\n",
        "analytics/dbt/transform/staging/stg_orders.sql": "select * from {{ source('raw', 'orders') }}\n",
        "analytics/dbt/transform/marts/orders.sql": "select o.* from {{ ref('stg_orders') }} as o\n",
        "analytics/dbt/macros/cents.sql": "{% macro cents(col) %}({{ col }} / 100){% endmacro %}\n",
        "analytics/dbt/tests/no_negative.sql": "select * from {{ ref('orders') }} where amount < 0\n",
        "etl/load.sql": "INSERT INTO raw.orders SELECT * FROM landing.orders;\n",
    })
    result = Orchestrator().run(str(tmp_path))
    assert result["status"] == "complete", result["hydrologist"]["diagnostics"]
    assert result["hydrologist"]["dbt_projects"] == ["analytics/dbt"]
    engine = QueryEngine.open(tmp_path)
    assert _upstream(engine, "orders") == {"stg_orders", "raw.orders", "landing.orders"}
    evidence = next(row["evidence"] for row in engine.trace_data_lineage("orders")["data"]["results"]
                    if row["node"] == "stg_orders")
    assert evidence["path"] == "analytics/dbt/transform/marts/orders.sql"


def test_several_projects_and_root_project_macros(tmp_path):
    _write(tmp_path, {
        "dbt_project.yml": "name: root\n",
        "models/a.sql": "select * from {{ source('raw', 'a') }}\n",
        "macros/m.sql": "{% macro m() %}1{% endmacro %}\n",
        "teams/finance/dbt_project.yml": "name: finance\n",
        "teams/finance/models/ledger.sql": "select * from {{ source('erp', 'ledger') }}\n",
    })
    result = Orchestrator().run(str(tmp_path))
    assert result["status"] == "complete", result["hydrologist"]["diagnostics"]
    assert result["hydrologist"]["dbt_projects"] == [".", "teams/finance"]
    engine = QueryEngine.open(tmp_path)
    assert _upstream(engine, "a") == {"raw.a"}
    assert _upstream(engine, "ledger") == {"erp.ledger"}
