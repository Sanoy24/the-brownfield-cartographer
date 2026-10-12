"""Artifact lineage must retain dbt identities and honest source evidence."""
import json
from pathlib import Path

import pytest

from brownfield_cartographer.analyzers.dbt_manifest import (
    SCHEMA_URL,
    parse_dbt_manifest,
)
from brownfield_cartographer.graph.knowledge_graph import KnowledgeGraph
from brownfield_cartographer.orchestrator import Orchestrator
from brownfield_cartographer.query_api import QueryEngine
from brownfield_cartographer.scanner import ScanSettings, scan_repository


@pytest.fixture
def manifest():
    def resource(uid, path, dependencies=()):
        return {"unique_id": uid, "resource_type": uid.split(".")[0],
                "original_file_path": path, "depends_on": {"nodes": list(dependencies)},
                "relation_name": "analytics.orders"}
    return {"metadata": {"dbt_schema_version": SCHEMA_URL}, "nodes": {
        "model.shop.orders.v1": resource("model.shop.orders.v1", "models/orders.sql", ["source.shop.raw.orders"]),
        "model.shop.orders.v2": resource("model.shop.orders.v2", "models/orders_v2.sql", ["model.shop.orders.v1"]),
        "model.other.orders": resource("model.other.orders", "dbt_packages/other/models/orders.sql"),
        "seed.shop.countries": resource("seed.shop.countries", "seeds/countries.csv"),
        "snapshot.shop.history": resource("snapshot.shop.history", "snapshots/history.sql", ["model.shop.orders.v2"]),
        "test.shop.orders": resource("test.shop.orders", "tests/orders.sql", ["model.shop.orders.v1"]),
    }, "sources": {"source.shop.raw.orders": resource("source.shop.raw.orders", "models/sources.yml")}}


def test_manifest_lineage_round_trip_and_evidence(manifest, tmp_path):
    result = parse_dbt_manifest(json.dumps(manifest), "target/manifest.json")
    assert len(result["datasets"]) == 6
    assert len(result["transformations"]) == 5
    assert all(not row["incomplete"] for row in result["diagnostics"])
    graph = KnowledgeGraph()
    for node in result["datasets"]:
        graph.add_dataset_node(node)
    for node in result["transformations"]:
        graph.add_transformation_node(node)
    graph.graph.graph.update(repository=str(tmp_path), snapshot={}, run_status="complete")
    graph.serialize_index(tmp_path / "index.json")
    restored = KnowledgeGraph.from_artifacts(tmp_path)
    assert restored.graph.nodes["model.shop.orders.v1"]["relation_name"] == "analytics.orders"
    response = QueryEngine(restored, tmp_path).trace_data_lineage("snapshot.shop.history")
    rows = response["data"]["results"]
    assert {row["node"] for row in rows} >= {
        "model.shop.orders.v2", "model.shop.orders.v1", "source.shop.raw.orders",
    }
    for row in rows:
        assert row["evidence"]["method"] == "dbt_manifest"
        assert row["evidence"]["confidence"] == "artifact"
        assert row["evidence"]["artifact_file"] == "target/manifest.json"
        assert not row["evidence"]["location_available"]
        assert row["evidence"]["line_range"] is None
    assert "model.other.orders" not in {row["node"] for row in rows}


def test_missing_dependency_is_partial_without_invented_dataset(manifest):
    manifest["nodes"]["model.shop.orders.v1"]["depends_on"]["nodes"] = ["model.missing.orders"]
    result = parse_dbt_manifest(json.dumps(manifest), "target/manifest.json")
    assert any(row["incomplete"] for row in result["diagnostics"])
    assert "model.missing.orders" not in {node.name for node in result["datasets"]}


@pytest.mark.parametrize("path", ["../private.sql", "/private.sql", "C:/private.sql", "models\\private.sql"])
def test_external_definition_paths_are_rejected(manifest, path):
    manifest["nodes"]["model.shop.orders.v1"]["original_file_path"] = path
    with pytest.raises(ValueError, match="paths must"):
        parse_dbt_manifest(json.dumps(manifest), "target/manifest.json")


def test_unsupported_manifest_version_is_rejected(manifest):
    manifest["metadata"]["dbt_schema_version"] = SCHEMA_URL.replace("v12", "v99")
    with pytest.raises(ValueError, match="v12"):
        parse_dbt_manifest(json.dumps(manifest), "target/manifest.json")


def test_resource_and_dependency_limits(manifest, monkeypatch):
    import brownfield_cartographer.analyzers.dbt_manifest as adapter
    monkeypatch.setattr(adapter, "MAX_RESOURCES", 1)
    with pytest.raises(ValueError, match="resource limit"):
        adapter.parse_dbt_manifest(json.dumps(manifest), "target/manifest.json")
    monkeypatch.setattr(adapter, "MAX_RESOURCES", 10_000)
    monkeypatch.setattr(adapter, "MAX_DEPENDENCIES", 1)
    with pytest.raises(ValueError, match="dependency limit"):
        adapter.parse_dbt_manifest(json.dumps(manifest), "target/manifest.json")


@pytest.fixture
def manifest_repo(tmp_path, manifest):
    (tmp_path / ".gitignore").write_text("target/\n")
    (tmp_path / "dbt_project.yml").write_text("name: shop\n")
    (tmp_path / "models").mkdir()
    (tmp_path / "models/orders.sql").write_text("select * from {{ source('raw', 'orders') }}\n")
    (tmp_path / "target").mkdir()
    (tmp_path / "target/manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_cli_manifest_overrides_ignore_and_preserves_source_context(manifest_repo):
    from typer.testing import CliRunner

    from brownfield_cartographer.cli import app
    result = CliRunner().invoke(app, ["analyze", str(manifest_repo), "--dbt-manifest", "target/manifest.json", "--json"])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["hydrologist"]["dbt_mode"] == "manifest_v12"
    engine = QueryEngine.open(manifest_repo)
    lineage = engine.trace_data_lineage("model.shop.orders.v1")
    assert "source.shop.raw.orders" in {row["node"] for row in lineage["data"]["results"]}
    assert "source('raw'" in engine.get_code_context("models/orders.sql")["data"]["content"]
    snapshot = json.loads((Path(summary["artifact_directory"]) / "state.json").read_text())["fingerprints"]
    assert "target/manifest.json" in snapshot
    assert "target/manifest.json" not in scan_repository(manifest_repo).fingerprints
    assert "orders" not in engine.graph.graph
    impact = engine.analyze_impact("models/orders.sql")["data"]["results"]
    assert impact[0]["evidence"]["method"] == "dbt_manifest"
    assert impact[0]["evidence"]["artifact_file"] == "target/manifest.json"
    assert not impact[0]["evidence"]["location_available"]


def test_manifest_changes_invalidate_cache_and_source_context(manifest_repo, manifest):
    from brownfield_cartographer.query_api import QueryError
    settings = ScanSettings(dbt_manifest="target/manifest.json")
    Orchestrator(scan_settings=settings).run(str(manifest_repo))
    engine = QueryEngine.open(manifest_repo)
    assert Orchestrator(scan_settings=settings).run(str(manifest_repo))["status"] == "no_changes"
    manifest["nodes"]["model.shop.orders.v1"]["depends_on"]["nodes"] = ["seed.shop.countries"]
    (manifest_repo / "target/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(QueryError, match="changed since indexing"):
        engine.get_code_context("models/orders.sql")
    assert Orchestrator(scan_settings=settings).run(str(manifest_repo))["status"] == "complete"
    rows = engine.trace_data_lineage("model.shop.orders.v1")["data"]["results"]
    assert "seed.shop.countries" in {row["node"] for row in rows}
    assert "source.shop.raw.orders" not in {row["node"] for row in rows}


@pytest.mark.parametrize("failure", ["missing", "invalid", "oversized"])
def test_bad_artifact_never_advances_checkpoint(manifest_repo, failure):
    settings = ScanSettings(dbt_manifest="target/manifest.json")
    first = Orchestrator(scan_settings=settings).run(str(manifest_repo))
    checkpoint_path = Path(first["artifact_directory"]) / "state.json"
    checkpoint = checkpoint_path.read_bytes()
    artifact = manifest_repo / "target/manifest.json"
    if failure == "missing":
        artifact.unlink()
    elif failure == "invalid":
        artifact.write_text("[]")
    else:
        artifact.write_text(" " * (settings.max_file_bytes + 1))
    result = Orchestrator(scan_settings=settings).run(str(manifest_repo))
    assert result["status"] == "partial"
    assert checkpoint_path.read_bytes() == checkpoint
    assert result["hydrologist"]["dbt_mode"] == "manifest_v12"


@pytest.mark.parametrize("path", ["../manifest.json", "/manifest.json", "C:/manifest.json", ".aws/manifest.json"])
def test_cli_rejects_external_or_protected_artifacts(manifest_repo, path):
    from typer.testing import CliRunner

    from brownfield_cartographer.cli import app
    result = CliRunner().invoke(app, ["analyze", str(manifest_repo), "--dbt-manifest", path, "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "failed"


def test_selected_artifact_cannot_cross_linked_ancestor(manifest_repo, monkeypatch):
    import brownfield_cartographer.scanner as scanner
    original = scanner._link
    monkeypatch.setattr(scanner, "_link", lambda path: path == manifest_repo / "target" or original(path))
    inventory = scanner.scan_repository(manifest_repo, ScanSettings(dbt_manifest="target/manifest.json"))
    assert not inventory.complete
    assert inventory.dbt_manifest_content is None
    assert "target/manifest.json" not in inventory.fingerprints
    assert any(row.reason == "symlink" for row in inventory.diagnostics)
