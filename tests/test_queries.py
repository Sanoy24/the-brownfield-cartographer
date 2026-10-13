"""The Python API and CLI must expose identical deterministic facts."""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from brownfield_cartographer.cli import app
from brownfield_cartographer.orchestrator import Orchestrator
from brownfield_cartographer.query_api import QueryEngine, QueryError


@pytest.fixture
def indexed_repo(tmp_path):
    (tmp_path / "consumer.py").write_text("import producer\ndef run():\n    return producer.VALUE\n")
    (tmp_path / "producer.py").write_text("VALUE=1\n")
    (tmp_path / "output.sql").write_text("select * from schema.raw\n")
    result = Orchestrator().run(str(tmp_path))
    assert result["status"] == "complete"
    return tmp_path


def test_shared_queries_and_function_evidence(indexed_repo):
    engine = QueryEngine.open(indexed_repo)
    found = engine.find_symbol("run")["data"]["matches"]
    assert found[0]["id"] == "consumer.py::run"
    assert found[0]["evidence"]["line_range"] == [2, 3]
    assert engine.trace_dependencies("consumer.py")["data"]["results"][0]["node"] == "producer.py"
    assert engine.analyze_impact("producer.py")["data"]["results"][0]["node"] == "consumer.py"
    assert engine.analyze_impact("producer.py")["data"]["results"][0]["evidence"]["line_range"] == [1, 1]
    assert engine.trace_data_lineage("output")["data"]["results"][-1]["node"] == "schema.raw"
    assert engine.get_code_context("consumer.py", 2, 3)["data"]["content"] == "def run():\n    return producer.VALUE"


@pytest.mark.parametrize("command,args,method,kwargs", [
    ("overview", [], "get_architecture", {}),
    ("find", ["run"], "find_symbol", {"pattern": "run"}),
    ("impact", ["producer.py"], "analyze_impact", {"target": "producer.py"}),
    ("lineage", ["output"], "trace_data_lineage", {"dataset": "output"}),
])
def test_cli_json_matches_python(indexed_repo, command, args, method, kwargs):
    arguments = [command, *args]
    arguments += [str(indexed_repo)] if command == "overview" else ["--repo", str(indexed_repo)]
    result = CliRunner().invoke(app, [*arguments, "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == getattr(QueryEngine.open(indexed_repo), method)(**kwargs)


def test_interactive_query_uses_shared_engine(indexed_repo):
    script = "find run\nimpact producer.py\nlineage-down schema.raw\ncontext consumer.py 2 3\nbogus\nimpact\nimpact nowhere.py\nquit\n"
    result = CliRunner().invoke(app, ["query", str(indexed_repo)], input=script)
    assert result.exit_code == 0, result.output
    assert "consumer.py::run" in result.output
    assert "consumer.py:1-1" in result.output
    assert "output" in result.output
    assert "def run():" in result.output
    assert "Unknown command: bogus" in result.output
    assert "Usage: impact <target>" in result.output
    assert "Target not found: nowhere.py" in result.output


def test_interactive_query_ends_at_end_of_input(indexed_repo):
    result = CliRunner().invoke(app, ["query", str(indexed_repo)], input="overview\n")
    assert result.exit_code == 0, result.output
    assert "get_architecture" in result.output


def test_version_reports_installed_distribution():
    from importlib.metadata import version

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == f"cartographer {version('brownfield-cartographer')}"


def test_analyze_json_stdout_is_clean(indexed_repo):
    result = CliRunner().invoke(app, ["analyze", str(indexed_repo), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "no_changes"


def test_source_scope_and_stale_context(indexed_repo):
    engine = QueryEngine.open(indexed_repo)
    for path in ("../secret.py", str(indexed_repo / "consumer.py"), ".env"):
        with pytest.raises(QueryError):
            engine.get_code_context(path)
    (indexed_repo / "consumer.py").write_text("VALUE=2\n")
    with pytest.raises(QueryError, match="changed since indexing"):
        engine.get_code_context("consumer.py")


def test_policy_changes_prevent_source_read(indexed_repo):
    engine = QueryEngine.open(indexed_repo)
    (indexed_repo / ".cartographerignore").write_text("consumer.py\n")
    with pytest.raises(QueryError, match="ignore policy"):
        engine.get_code_context("consumer.py")


def test_query_engine_observes_newly_analyzed_snapshot(indexed_repo):
    engine = QueryEngine.open(indexed_repo)
    old_snapshot = engine.snapshot_id
    (indexed_repo / "producer.py").write_text("VALUE=2\n")
    Orchestrator().run(str(indexed_repo))
    assert engine.get_code_context("producer.py")["data"]["content"] == "VALUE=2"
    assert engine.snapshot_id != old_snapshot


def test_existing_engine_observes_failed_run_without_new_index(indexed_repo, monkeypatch):
    engine = QueryEngine.open(indexed_repo)
    index = Path(engine.index_dir) / "index.json"
    original = index.read_bytes()
    runner = Orchestrator()
    def fail(*args):
        raise OSError("controlled index publication failure")
    monkeypatch.setattr(runner.knowledge_graph, "serialize_index", fail)
    with pytest.raises(OSError, match="publication failure"):
        runner.run(str(indexed_repo), full=True)
    assert index.read_bytes() == original
    assert engine.get_architecture()["status"] == "failed"
    assert engine.find_symbol("run")["status"] == "failed"
    Orchestrator().run(str(indexed_repo), full=True)
    assert engine.get_architecture()["status"] == "complete"


@pytest.mark.parametrize("content", ["[]", "{", '{"status": "invalid"}'])
def test_invalid_summary_is_a_query_error(indexed_repo, content):
    engine = QueryEngine.open(indexed_repo)
    (Path(engine.index_dir) / "analysis_summary.json").write_text(content)
    with pytest.raises(QueryError):
        engine.get_architecture()
    result = CliRunner().invoke(app, ["overview", str(indexed_repo), "--json"])
    assert result.exit_code == 1
    assert "error" in json.loads(result.stdout)


@pytest.mark.parametrize("content", ["[]", "{", '{"schema_version": 1, "graph": {}}'])
def test_invalid_index_is_a_query_error(indexed_repo, content):
    engine = QueryEngine.open(indexed_repo)
    (Path(engine.index_dir) / "index.json").write_text(content)
    with pytest.raises(QueryError):
        engine.find_symbol("run")


def test_removed_index_is_not_served_from_memory(indexed_repo):
    engine = QueryEngine.open(indexed_repo)
    (Path(engine.index_dir) / "index.json").unlink()
    with pytest.raises(QueryError, match="No versioned index"):
        engine.get_architecture()


def test_bounds_and_directions(indexed_repo):
    engine = QueryEngine.open(indexed_repo)
    assert engine.find_symbol("py", limit=1)["truncated"]
    for call in (lambda: engine.find_symbol("run", 201),
                 lambda: engine.analyze_impact("producer.py", depth=100),
                 lambda: engine.trace_data_lineage("output", direction="sideways"),
                 lambda: engine.get_code_context("consumer.py", 1, 300)):
        with pytest.raises(QueryError):
            call()


def test_index_cannot_be_rebound_to_another_repository(indexed_repo, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    index = indexed_repo / ".cartography" / indexed_repo.name
    with pytest.raises(QueryError, match="scope does not match"):
        QueryEngine.open(other, index)


def test_ambiguous_symbol_is_not_silently_selected(indexed_repo):
    for name in ("a", "b"):
        directory = indexed_repo / name
        directory.mkdir()
        (directory / "common.py").write_text("VALUE=1\n")
    Orchestrator().run(str(indexed_repo))
    with pytest.raises(QueryError, match="Ambiguous"):
        QueryEngine.open(indexed_repo).analyze_impact("common.py")
