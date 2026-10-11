"""Artifact lineage must retain dbt identities and honest source evidence."""
import json

import pytest

from brownfield_cartographer.analyzers.dbt_manifest import SCHEMA_URL, parse_dbt_manifest
from brownfield_cartographer.graph.knowledge_graph import KnowledgeGraph
from brownfield_cartographer.query_api import QueryEngine


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
