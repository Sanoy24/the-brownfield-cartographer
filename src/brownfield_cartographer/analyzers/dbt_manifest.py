"""Extract declared dataset dependencies from a supplied dbt v12 artifact.

This parser consumes JSON text only: it never resolves packages, runs dbt,
executes compiled SQL, or reads paths declared by the artifact.
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from brownfield_cartographer.analyzers.sql_lineage import (
    sql_statements,
    validate_dialect,
)
from brownfield_cartographer.models.schemas import (
    DatasetNode,
    TransformationNode,
    TransformationType,
)

SCHEMA_URL = "https://schemas.getdbt.com/dbt/manifest/v12.json"
MAX_RESOURCES = 10_000
MAX_DEPENDENCIES = 50_000
DATASET_TYPES = {"model", "seed", "snapshot", "source"}


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 1000 or "\\" in value:
        raise ValueError("dbt definition paths must be relative forward-slash paths")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts or str(path) == ".":
        raise ValueError("dbt definition paths must stay within the repository")
    return path.as_posix()


def _relation_key(name: Any) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    return name.translate(str.maketrans("", "", '"`[]')).lower()


def _compiled_columns(compiled: str, path: str, relations: dict[str, str],
                      dialect: str | None) -> tuple[list[dict], list[str]] | None:
    """Column lineage of a model's compiled SQL, with tables mapped to dbt unique IDs; None if unparseable."""
    errors: list[dict] = []
    statements = sql_statements(f"CREATE TABLE __cartographer_model AS {compiled.strip().rstrip(';')}",
                                path, dialect, errors)
    if errors or len(statements) != 1:
        return None
    columns = []
    for entry in statements[0]["columns"]:
        sources = []
        for source in entry["sources"]:
            table, _, column = source.rpartition("::")
            sources.append(f"{relations.get(table.lower(), table)}::{column}")
        columns.append({**entry, "sources": sorted(sources)})
    return columns, statements[0]["unresolved_columns"]


def parse_dbt_manifest(content: str, artifact_path: str) -> dict[str, Any]:
    """Read the lineage subset of manifest v12, reporting unresolved resources.

    Dataset names use dbt unique IDs so packages and model versions stay
    distinct. Locations identify definitions but have no invented line spans.
    Unsupported non-dataset resources (tests/macros/etc.) are deliberately
    outside this adapter's scope. This is not full JSON-schema validation.
    """
    artifact_path = _relative_path(artifact_path)
    raw = json.loads(content)
    if not isinstance(raw, dict) or not isinstance(raw.get("metadata"), dict):
        raise ValueError("Invalid dbt manifest metadata")
    if raw["metadata"].get("dbt_schema_version") != SCHEMA_URL:
        raise ValueError("Only dbt manifest v12 is supported")
    if any(not isinstance(raw.get(key), dict) for key in ("nodes", "sources")):
        raise ValueError("dbt manifest nodes and sources must be objects")
    if len(raw["nodes"]) + len(raw["sources"]) > MAX_RESOURCES:
        raise ValueError("dbt manifest exceeds the resource limit")

    resources = {}
    datasets = []
    transformations = []
    diagnostics = []
    dependency_count = 0
    for collection in ("nodes", "sources"):
        for unique_id, node in sorted(raw[collection].items()):
            if not isinstance(node, dict):
                raise ValueError("dbt resources must be objects")
            kind = node.get("resource_type")
            if not isinstance(kind, str):
                raise ValueError("dbt resources must declare a resource type")
            if kind not in DATASET_TYPES:
                continue
            if (not unique_id or len(unique_id) > 1000 or node.get("unique_id") != unique_id
                    or not unique_id.startswith(f"{kind}.")):
                raise ValueError("Invalid dbt resource unique ID")
            if unique_id in resources:
                raise ValueError("Duplicate dbt resource unique ID")
            config = node.get("config", {})
            if not isinstance(config, dict):
                raise ValueError("dbt resource config must be an object")
            if config.get("enabled") is False:
                continue
            path = _relative_path(node.get("original_file_path"))
            resources[unique_id] = (node, path)
            datasets.append(DatasetNode(
                name=unique_id, dbt_unique_id=unique_id,
                relation_name=node.get("relation_name"), is_source_of_truth=kind == "source",
            ))

    # Column lineage comes from compiled SQL, where refs and macros are already expanded.
    relations = {key: uid for uid, (node, _) in resources.items()
                 if (key := _relation_key(node.get("relation_name"))) is not None}
    adapter = raw["metadata"].get("adapter_type")
    try:
        dialect = validate_dialect(adapter) if isinstance(adapter, str) else None
    except ValueError:
        dialect = None
    uncompiled, unparsed = 0, 0

    for unique_id, (node, path) in sorted(resources.items()):
        if node["resource_type"] == "source":
            continue
        columns: list[dict] = []
        unresolved: list[str] = []
        if node["resource_type"] in {"model", "snapshot"}:
            compiled = node.get("compiled_code")
            if not isinstance(compiled, str) or not compiled.strip():
                uncompiled += 1
            elif (parsed := _compiled_columns(compiled, path, relations, dialect)) is None:
                unparsed += 1
            else:
                columns, unresolved = parsed
        depends_on = node.get("depends_on", {})
        if not isinstance(depends_on, dict) or not isinstance(depends_on.get("nodes", []), list):
            raise ValueError("dbt node dependencies must be a list")
        dependencies = depends_on.get("nodes", [])
        dependency_count += len(dependencies)
        if dependency_count > MAX_DEPENDENCIES:
            raise ValueError("dbt manifest exceeds the dependency limit")
        upstream = []
        for dependency in dependencies:
            if not isinstance(dependency, str):
                raise ValueError("dbt dependency IDs must be strings")
            if dependency not in resources:
                diagnostics.append({"path": artifact_path, "resource_id": unique_id,
                                    "error": f"Unresolved dataset dependency: {dependency}", "incomplete": True})
            else:
                upstream.append(dependency)
        transformations.append(TransformationNode(
            source_file=path, source_datasets=sorted(set(upstream)), target_datasets=[unique_id],
            transformation_type=TransformationType.DBT_MODEL, analysis_method="dbt_manifest",
            confidence="artifact", artifact_file=artifact_path, resource_id=unique_id,
            column_lineage=columns, unresolved_columns=unresolved,
        ))
    if uncompiled:
        diagnostics.append({"path": artifact_path, "incomplete": False, "error": (
            f"{uncompiled} model(s) have no compiled SQL in the manifest; run `dbt compile` "
            "before analysis to include their column lineage.")})
    if unparsed:
        diagnostics.append({"path": artifact_path, "incomplete": False, "error": (
            f"{unparsed} model(s) have compiled SQL that could not be parsed for column lineage.")})
    diagnostics.append({"path": artifact_path, "error": (
        "Lineage describes the supplied manifest; freshness against working-tree SQL is not verified. "
        "Only models, seeds, snapshots and sources are included."
    ), "incomplete": False})
    return {"datasets": datasets, "transformations": transformations, "diagnostics": diagnostics,
            "definition_files": sorted({path for _, path in resources.values()}),
            "manifest_version": 12}
