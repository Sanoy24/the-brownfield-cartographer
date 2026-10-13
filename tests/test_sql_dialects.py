"""SQL dialects: best-fit selection per file, explicit pinning, and lineage-free commands."""
import logging

import pytest
from typer.testing import CliRunner

from brownfield_cartographer.analyzers.sql_lineage import sql_statements
from brownfield_cartographer.cli import app
from brownfield_cartographer.orchestrator import Orchestrator
from brownfield_cartographer.query_api import QueryEngine
from brownfield_cartographer.scanner import ScanSettings

BIGQUERY = (
    "CREATE OR REPLACE TABLE `proj.ds.trips` PARTITION BY DATE(pickup) AS\n"
    "SELECT * FROM `proj.ds.raw_trips`;\n"
    "SELECT count(*) FROM `proj.ds.trips`;\n"
)
MYSQL = (
    "# Seed the reporting copy\n"
    "INSERT INTO shop.orders_copy SELECT * FROM shop.orders;\n"
    "GRANT ALL PRIVILEGES ON shop.* TO 'report'@'%';\n"
)


def _run(tmp_path, files, settings=None):
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    result = Orchestrator(scan_settings=settings).run(str(tmp_path))
    return result, QueryEngine.open(tmp_path)


def _producer(engine, dataset):
    row = next(r for r in engine.trace_data_lineage(dataset)["data"]["results"] if r["node_type"] == "dataset")
    return row["node"], row["evidence"]["method"]


@pytest.mark.parametrize("files,dataset,expected", [
    ({"warehouse.sql": BIGQUERY}, "proj.ds.trips", ("proj.ds.raw_trips", {"sql:bigquery"})),
    # Any dialect that accepts '#' comments parses this; the label names the grammar used, not the engine.
    ({"seed.sql": MYSQL}, "shop.orders_copy", ("shop.orders", {"sql:bigquery", "sql:mysql"})),
    ({"plain.sql": "CREATE TABLE a.b AS SELECT * FROM a.c;"}, "a.b", ("a.c", {"sql:duckdb"})),
])
def test_best_fit_dialect_is_chosen_per_file_and_recorded(tmp_path, caplog, files, dataset, expected):
    with caplog.at_level(logging.WARNING, logger="sqlglot"):
        result, engine = _run(tmp_path, files)
    assert result["status"] == "complete", result["hydrologist"]["diagnostics"]
    producer, method = _producer(engine, dataset)
    assert producer == expected[0] and method in expected[1]
    assert not [r for r in caplog.records if r.name.startswith("sqlglot")]


def test_pinned_dialect_disables_fallback(tmp_path):
    result, _ = _run(tmp_path, {"warehouse.sql": BIGQUERY}, ScanSettings(sql_dialect="duckdb"))
    assert result["status"] == "partial"


def test_unknown_dialect_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported SQL dialect"):
        ScanSettings(sql_dialect="cobol")
    result = CliRunner().invoke(app, ["analyze", str(tmp_path), "--dialect", "cobol"])
    assert result.exit_code == 1
    assert "Unsupported SQL dialect" in result.output


def test_lineage_free_commands_are_skipped_but_opaque_creates_are_not():
    errors = []
    script = ("CREATE CATALOG hudi WITH ('type'='hudi'); SET sql-client.verbose=true; "
              "GRANT ALL PRIVILEGES ON shop.* TO 'r'@'%'; CREATE DATABASE IF NOT EXISTS hudi.flink;")
    assert sql_statements(script, "init.sql", "duckdb", errors) == []
    assert errors == []
    sql_statements("CREATE OR REPLACE TABLE `a.b` PARTITION BY DATE(d) AS SELECT * FROM `a.c`", "x.sql", "duckdb", errors)
    assert len(errors) == 1


def test_embedded_spark_sql_uses_a_fitting_dialect(tmp_path):
    result, engine = _run(tmp_path, {"job.py": (
        "def run(spark):\n"
        "    spark.sql('INSERT OVERWRITE TABLE db.daily SELECT * FROM db.events')\n"
    )})
    assert result["status"] == "complete", result["hydrologist"]["diagnostics"]
    assert _producer(engine, "db.daily")[0] == "db.events"


def test_queries_in_scripts_are_reads_not_file_name_writes(tmp_path):
    result, engine = _run(tmp_path, {
        "worksheet.sql": "CREATE TABLE a.t AS SELECT * FROM a.s;\nSELECT count(*) FROM a.t;\nSELECT * FROM a.other;\n",
        "daily_view.sql": "SELECT day, count(*) FROM a.events GROUP BY day\n",
    })
    assert result["status"] == "complete"
    graph = engine.graph.graph
    datasets = {n for n, d in graph.nodes(data=True) if d.get("node_type") == "dataset"}
    assert "worksheet" not in datasets
    assert _producer(engine, "a.t") == ("a.s", "sql:duckdb")
    upstream = next(r for r in engine.trace_data_lineage("daily_view")["data"]["results"] if r["node_type"] == "dataset")
    assert upstream["node"] == "a.events"
    assert (upstream["evidence"]["method"], upstream["evidence"]["confidence"]) == ("sql:duckdb+file_name", "inferred")
