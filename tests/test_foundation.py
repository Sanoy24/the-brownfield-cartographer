"""Source-level regressions for the local, deterministic foundation."""
import json
from pathlib import Path

import pytest

from brownfield_cartographer.agents.hydrologist import Hydrologist
from brownfield_cartographer.agents.surveyor import Surveyor
from brownfield_cartographer.analyzers.sql_lineage import analyze_sql_file
from brownfield_cartographer.graph.knowledge_graph import KnowledgeGraph
from brownfield_cartographer.models.schemas import (
    DatasetNode,
    FunctionNode,
    ModuleNode,
    TransformationNode,
)
from brownfield_cartographer.query_api import QueryEngine


def _engine(kg, repo):
    """Query an in-memory graph through the shared engine."""
    kg.graph.graph["repository"] = str(repo)
    return QueryEngine(kg, repo)


def _impact(kg, repo, target):
    return {row["node"] for row in _engine(kg, repo).analyze_impact(target)["data"]["results"]}


def test_reverse_import_impact_excludes_start_in_cycle(tmp_path):
    kg = KnowledgeGraph()
    for path in ("a.py", "b.py", "c.py"):
        kg.add_module_node(ModuleNode(path=path))
    kg.add_import_edge("a.py", "b.py")
    kg.add_import_edge("c.py", "a.py")
    kg.add_import_edge("b.py", "a.py")
    assert _impact(kg, tmp_path, "b.py") == {"a.py", "c.py"}


def test_lineage_does_not_follow_configuration_edges(tmp_path):
    kg = KnowledgeGraph()
    kg.add_dataset_node(DatasetNode(name="raw"))
    kg.add_transformation_node(TransformationNode(
        source_file="job.sql", source_datasets=["raw"], target_datasets=["out"],
        line_range=(2, 3),
    ))
    kg.add_configures_edge("settings.yml", "out")
    upstream = _engine(kg, tmp_path).trace_data_lineage("out")["data"]["results"]
    assert {row["node"] for row in upstream} >= {"raw"}
    assert "settings.yml" not in {row["node"] for row in upstream}
    assert _impact(kg, tmp_path, "job.sql") >= {"out"}


def test_transformations_do_not_merge_independent_operations():
    import networkx as nx
    kg = KnowledgeGraph()
    for name in ("a", "b"):
        kg.add_transformation_node(TransformationNode(
            source_file="job.py", line_range=(1, 20),
            source_datasets=[f"raw_{name}"], target_datasets=[f"out_{name}"],
        ))
    assert kg.summary()["transformations"] == 2
    assert not nx.has_path(kg.graph, "raw_a", "out_b")
    assert kg.summary()["datasets"] == 4


def test_complete_graph_round_trip(tmp_path):
    kg = KnowledgeGraph()
    kg.add_module_node(ModuleNode(path="job.py", imports=["package"], purpose_statement="Purpose"))
    kg.add_function_node(FunctionNode(
        qualified_name="job.py::f", parent_module="job.py", line_start=3, line_end=5,
    ))
    kg.add_dataset_node(DatasetNode(name="out", owner="team", schema_snapshot={"id": "int"}))
    kg.add_configures_edge("schema.yml", "out")
    kg.add_transformation_node(TransformationNode(
        source_file="job.py", source_datasets=["raw"], target_datasets=["out"], line_range=(3, 5),
    ))
    kg.serialize_index(tmp_path / "index.json")
    restored = KnowledgeGraph.from_artifacts(tmp_path)
    assert restored.summary() == kg.summary()
    assert dict(restored.graph.nodes(data=True)) == dict(kg.graph.nodes(data=True))
    assert list(restored.graph.edges(data=True)) == list(kg.graph.edges(data=True))
    assert restored.graph.nodes["job.py::f"]["line_start"] == 3
    assert restored._module_nodes["job.py"].imports == ["package"]
    assert restored._dataset_nodes["out"].owner == "team"


def test_sql_statement_roles_and_locations(tmp_path):
    path = tmp_path / "job.sql"
    path.write_text("CREATE TABLE output_a AS SELECT * FROM input_a;\nCREATE TABLE output_b AS SELECT * FROM input_b;\n")
    txs = analyze_sql_file(path, tmp_path)
    assert [(t.source_datasets, t.target_datasets) for t in txs] == [
        (["input_a"], ["output_a"]), (["input_b"], ["output_b"]),
    ]
    assert [t.line_range for t in txs] == [(1, 1), (2, 2)]


def test_decorated_and_relative_python_symbols(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "helper.py").write_text("VALUE = 1\n")
    (package / "consumer.py").write_text("from . import helper\n@decorator\ndef run():\n    return helper.VALUE\n")
    kg = KnowledgeGraph()
    Surveyor().run(tmp_path, kg)
    assert "pkg/consumer.py::run" in kg._function_nodes
    assert kg.graph.has_edge("pkg/consumer.py", "pkg/helper.py")


def test_incremental_keeps_unchanged_importers(tmp_path):
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("VALUE = 1\n")
    kg = KnowledgeGraph()
    Surveyor().run(tmp_path, kg)
    (tmp_path / "b.py").write_text("VALUE = 2\n")
    Surveyor().run(tmp_path, kg, changed_files={"b.py"})
    assert kg.graph.has_edge("a.py", "b.py")


def test_dbt_incremental_preserves_refs_and_removes_deleted_assets(tmp_path):
    (tmp_path / "dbt_project.yml").write_text("name: fixture\n")
    models = tmp_path / "models"
    models.mkdir()
    model = models / "out.sql"
    model.write_text("select * from {{ ref('upstream') }}\n")
    kg = KnowledgeGraph()
    Hydrologist().run(tmp_path, kg)
    model.write_text("select id from {{ ref('upstream') }}\n")
    Hydrologist().run(tmp_path, kg, changed_files={"models/out.sql"})
    assert Hydrologist.trace_upstream(kg, "out") == ["upstream"]
    model.unlink()
    Hydrologist().run(tmp_path, kg, changed_files={"models/out.sql"})
    assert "out" not in kg.graph


def test_scanner_ignores_binary_secrets_and_limits(tmp_path):
    from brownfield_cartographer.scanner import ScanSettings, scan_repository
    (tmp_path / ".gitignore").write_text("ignored/\n")
    (tmp_path / ".cartographerignore").write_text("private.py\n")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "x.py").write_text("SECRET = 1\n")
    (tmp_path / "private.py").write_text("SECRET = 1\n")
    (tmp_path / ".env").write_text("SECRET=value\n")
    (tmp_path / "ok.py").write_text("VALUE=1\n")
    (tmp_path / "binary.py").write_bytes(b"a\x00b")
    (tmp_path / "big.py").write_text("x" * 101)
    result = scan_repository(tmp_path, ScanSettings(max_file_bytes=100))
    assert [p.name for p in result.files] == ["ok.py"]
    assert not result.complete
    assert {d.reason for d in result.diagnostics} >= {"binary", "oversized"}


def test_scanner_symlink_outside_scope(tmp_path):
    from brownfield_cartographer.scanner import scan_repository
    inside = tmp_path / "repo"
    inside.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET=1\n")
    try:
        (inside / "escape.py").symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this host")
    result = scan_repository(inside)
    assert result.files == ()
    assert any(d.reason == "symlink" for d in result.diagnostics)


def test_nested_ignore_negation_and_file_limit(tmp_path):
    from brownfield_cartographer.scanner import ScanSettings, scan_repository
    (tmp_path / ".gitignore").write_text("*.py\n!keep.py\n")
    nested = tmp_path / "pkg"
    nested.mkdir()
    (nested / ".gitignore").write_text("!local.py\n")
    (nested / "local.py").write_text("VALUE=1\n")
    (nested / "hidden.py").write_text("VALUE=1\n")
    (tmp_path / "keep.py").write_text("VALUE=1\n")
    assert [p.relative_to(tmp_path).as_posix() for p in scan_repository(tmp_path).files] == ["keep.py", "pkg/local.py"]
    result = scan_repository(tmp_path, ScanSettings(max_files=1))
    assert len(result.files) == 1 and not result.complete


def test_python_io_ignores_comments_and_keeps_function_scopes(tmp_path):
    from brownfield_cartographer.agents.hydrologist import _analyze_python_data_flow
    path = tmp_path / "io.py"
    path.write_text('# pd.read_csv("fake.csv")\nimport pandas as data\n'
                    'def one():\n    df=data.read_csv("one.csv")\n    df.to_csv("one_out.csv")\n'
                    'def two():\n    df=data.read_csv("two.csv")\n    df.to_csv("two_out.csv")\n')
    transformations = _analyze_python_data_flow(path, tmp_path)
    assert [(t.source_datasets, t.target_datasets) for t in transformations] == [
        (["one.csv"], ["one_out.csv"]), (["two.csv"], ["two_out.csv"])]
    assert [t.line_range for t in transformations] == [(4, 5), (7, 8)]


def test_incomplete_scan_and_bad_file_do_not_checkpoint(tmp_path):
    from brownfield_cartographer.orchestrator import Orchestrator
    from brownfield_cartographer.scanner import ScanSettings
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ok.py").write_text("VALUE=1\n")
    (repo / "bad.py").write_text("def broken(\n")
    out = tmp_path / "output"
    result = Orchestrator().run(str(repo), output_dir=out)
    assert result["status"] == "partial"
    artifacts = Path(result["artifact_directory"])
    assert "ok.py" in KnowledgeGraph.from_artifacts(artifacts).graph
    assert not (artifacts / "state.json").exists()
    (repo / "bad.py").unlink()
    (repo / "big.py").write_text("x" * 101)
    result = Orchestrator(scan_settings=ScanSettings(max_file_bytes=100)).run(str(repo), output_dir=out)
    assert result["status"] == "partial"
    assert not (artifacts / "state.json").exists()


def test_unchanged_run_preserves_history_and_loads_full_graph(tmp_path):
    from brownfield_cartographer.orchestrator import Orchestrator
    (tmp_path / "run.py").write_text("def run():\n    pass\n")
    first = Orchestrator()
    first.run(str(tmp_path))
    artifacts = tmp_path / ".cartography" / tmp_path.name
    history = (artifacts / "cartography_trace.jsonl").read_text()
    second = Orchestrator()
    assert second.run(str(tmp_path))["status"] == "no_changes"
    assert (artifacts / "cartography_trace.jsonl").read_text().startswith(history)
    assert second.knowledge_graph.summary() == first.knowledge_graph.summary()


def test_incompatible_index_version_is_rejected(tmp_path):
    path = tmp_path / "index.json"
    path.write_text('{"schema_version": 999, "graph": {}}')
    with pytest.raises(ValueError, match="Unsupported index"):
        KnowledgeGraph.from_artifacts(tmp_path)


def test_artifact_failure_preserves_checkpoint(tmp_path, monkeypatch):
    from brownfield_cartographer.orchestrator import Orchestrator
    (tmp_path / "ok.py").write_text("VALUE=1\n")
    Orchestrator().run(str(tmp_path))
    artifacts = tmp_path / ".cartography" / tmp_path.name
    checkpoint = (artifacts / "state.json").read_bytes()
    (tmp_path / "ok.py").write_text("VALUE=2\n")
    runner = Orchestrator()
    def fail(*args):
        raise OSError("controlled write failure")
    monkeypatch.setattr(runner.knowledge_graph, "serialize_index", fail)
    with pytest.raises(OSError, match="controlled write failure"):
        runner.run(str(tmp_path))
    assert (artifacts / "state.json").read_bytes() == checkpoint
    assert json.loads((artifacts / "analysis_summary.json").read_text())["status"] == "failed"
    assert Orchestrator().run(str(tmp_path))["status"] == "complete"


def test_no_network_for_local_analysis_and_queries(tmp_path, monkeypatch):
    import socket

    from brownfield_cartographer.orchestrator import Orchestrator
    def disallow(*args, **kwargs):
        pytest.fail("Local analysis attempted network access")
    monkeypatch.setattr(socket, "create_connection", disallow)
    monkeypatch.setattr(socket, "getaddrinfo", disallow)
    (tmp_path / "input.sql").write_text("select * from raw\n")
    runner = Orchestrator()
    assert runner.run(str(tmp_path))["status"] == "complete"
    engine = QueryEngine.open(tmp_path)
    assert engine.trace_data_lineage("input")["data"]["results"]
    assert engine.analyze_impact("raw")["data"]["results"]


def test_partial_cli_has_nonzero_exit(tmp_path):
    from typer.testing import CliRunner

    from brownfield_cartographer.cli import app
    (tmp_path / "broken.py").write_text("def broken(\n")
    result = CliRunner().invoke(app, ["analyze", str(tmp_path)])
    assert result.exit_code == 2


def test_pipeline_offline_changes_and_failed_checkpoint(tmp_path, monkeypatch):
    from brownfield_cartographer.orchestrator import Orchestrator
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def run():\n    return 1\n")
    output = tmp_path / "output"
    first = Orchestrator().run(str(repo), output_dir=output)
    assert first["status"] == "complete"
    artifacts = Path(first["artifact_directory"])
    before = (artifacts / "state.json").read_bytes()
    (repo / "a.py").write_text("def run():\n    return 2\n")
    runner = Orchestrator()
    def fail(*args, **kwargs):
        raise RuntimeError("controlled failure")
    monkeypatch.setattr(runner.surveyor, "run", fail)
    failed = runner.run(str(repo), output_dir=output)
    assert failed["status"] == "partial"
    assert (artifacts / "state.json").read_bytes() == before
    assert json.loads((artifacts / "analysis_summary.json").read_text())["agent_errors"]["surveyor"]
    assert Orchestrator().run(str(repo), output_dir=output)["status"] == "complete"
    assert (artifacts / "state.json").read_bytes() != before


@pytest.mark.parametrize("operation", ["add", "edit", "delete", "rename"])
def test_pipeline_worktree_full_equivalence(tmp_path, operation):
    from brownfield_cartographer.orchestrator import Orchestrator
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("import b\n")
    (repo / "b.py").write_text("VALUE=1\n")
    output = tmp_path / "cached"
    Orchestrator().run(str(repo), output_dir=output)
    if operation == "add":
        (repo / "new.py").write_text("import a\n")
    elif operation == "edit":
        (repo / "a.py").write_text("import b\nVALUE=2\n")
    elif operation == "delete":
        (repo / "b.py").unlink()
    else:
        (repo / "b.py").rename(repo / "renamed.py")
    updated = Orchestrator()
    assert updated.run(str(repo), output_dir=output)["status"] == "complete"
    fresh = Orchestrator()
    fresh.run(str(repo), output_dir=tmp_path / "fresh", full=True)
    assert dict(updated.knowledge_graph.graph.nodes(data=True)) == dict(fresh.knowledge_graph.graph.nodes(data=True))
    assert list(updated.knowledge_graph.graph.edges(data=True)) == list(fresh.knowledge_graph.graph.edges(data=True))
