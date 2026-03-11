"""
DAG Config Parser — extracts pipeline topology from config files.

Handles two primary config formats:
  - dbt: parses dbt_project.yml, schema.yml, and model SQL files for ref() calls
  - Airflow: parses Python DAG files for task dependency declarations

Produces TransformationNodes and ConfiguresEdges for the knowledge graph.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import yaml

from src.models.schemas import (
    ConfiguresEdge,
    DatasetNode,
    StorageType,
    TransformationNode,
    TransformationType,
)

logger = logging.getLogger(__name__)

# Regex to match dbt ref('model_name') and source('schema', 'table') calls
REF_PATTERN = re.compile(r"""\{\{\s*ref\(\s*['"](\w+)['"]\s*\)\s*\}\}""")
SOURCE_PATTERN = re.compile(
    r"""\{\{\s*source\(\s*['"](\w+)['"]\s*,\s*['"](\w+)['"]\s*\)\s*\}\}"""
)


# ---------------------------------------------------------------------------
# dbt Project Parsing
# ---------------------------------------------------------------------------

def parse_dbt_project(project_root: Path) -> dict:
    """
    Parse a dbt project and extract the full model dependency graph.

    Scans:
      - dbt_project.yml for project-level config
      - models/**/*.sql for ref() and source() calls
      - models/**/*.yml for schema definitions and column tests

    Returns:
        dict with keys:
          - "transformations": list[TransformationNode]
          - "datasets": list[DatasetNode]
          - "config_edges": list[ConfiguresEdge]
    """
    transformations: list[TransformationNode] = []
    datasets: list[DatasetNode] = []
    config_edges: list[ConfiguresEdge] = []

    # --- Parse dbt_project.yml for metadata ---
    project_file = project_root / "dbt_project.yml"
    if project_file.exists():
        try:
            project_config = yaml.safe_load(project_file.read_text(encoding="utf-8"))
            project_name = project_config.get("name", "unknown_dbt_project")
            logger.info("Parsing dbt project: %s", project_name)
        except Exception as exc:
            logger.warning("Failed to parse dbt_project.yml: %s", exc)

    # --- Scan model SQL files for ref() and source() dependencies ---
    models_dir = project_root / "models"
    if not models_dir.exists():
        # Some dbt projects use a different directory
        for candidate in ("models", "dbt", "transformations"):
            alt = project_root / candidate
            if alt.exists():
                models_dir = alt
                break

    if models_dir.exists():
        for sql_file in models_dir.rglob("*.sql"):
            relative_path = str(sql_file.relative_to(project_root)).replace("\\", "/")
            model_name = sql_file.stem
            sql_text = sql_file.read_text(encoding="utf-8", errors="replace")

            # Extract ref() dependencies → these are upstream dbt models
            ref_deps = REF_PATTERN.findall(sql_text)
            source_deps = SOURCE_PATTERN.findall(sql_text)

            source_datasets = list(ref_deps)
            for schema_name, table_name in source_deps:
                source_datasets.append(f"{schema_name}.{table_name}")

            transformations.append(
                TransformationNode(
                    source_datasets=source_datasets,
                    target_datasets=[model_name],
                    transformation_type=TransformationType.DBT_MODEL,
                    source_file=relative_path,
                    line_range=(1, len(sql_text.split("\n"))),
                    sql_query_if_applicable=sql_text[:500],
                )
            )

    # --- Parse schema YAML files for dataset metadata ---
    for yml_file in (models_dir if models_dir.exists() else project_root).rglob("*.yml"):
        relative_yml = str(yml_file.relative_to(project_root)).replace("\\", "/")
        try:
            schema_data = yaml.safe_load(yml_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse YAML %s: %s", relative_yml, exc)
            continue

        if not isinstance(schema_data, dict):
            continue

        # Extract model definitions
        for model in schema_data.get("models", []):
            if not isinstance(model, dict):
                continue
            model_name = model.get("name", "")
            columns = model.get("columns", [])
            schema_snapshot = {
                col.get("name"): col.get("data_type", "unknown")
                for col in columns
                if isinstance(col, dict) and col.get("name")
            } if columns else None

            datasets.append(
                DatasetNode(
                    name=model_name,
                    storage_type=StorageType.TABLE,
                    schema_snapshot=schema_snapshot,
                    owner=model.get("meta", {}).get("owner") if isinstance(model.get("meta"), dict) else None,
                )
            )
            config_edges.append(
                ConfiguresEdge(source=relative_yml, target=model_name)
            )

        # Extract source definitions
        for source in schema_data.get("sources", []):
            if not isinstance(source, dict):
                continue
            source_schema = source.get("name", "")
            for table in source.get("tables", []):
                if not isinstance(table, dict):
                    continue
                table_name = table.get("name", "")
                datasets.append(
                    DatasetNode(
                        name=f"{source_schema}.{table_name}",
                        storage_type=StorageType.TABLE,
                        is_source_of_truth=True,
                    )
                )

    return {
        "transformations": transformations,
        "datasets": datasets,
        "config_edges": config_edges,
    }


# ---------------------------------------------------------------------------
# Airflow DAG Parsing (basic)
# ---------------------------------------------------------------------------

# Regex patterns for Airflow task dependencies in Python files
TASK_DEPENDENCY_PATTERN = re.compile(r"(\w+)\s*>>\s*(\w+)")
OPERATOR_PATTERN = re.compile(
    r"""(\w+)\s*=\s*(\w+(?:Operator|Sensor|Task))\s*\("""
)


def parse_airflow_dag(file_path: Path, repo_root: Path) -> list[TransformationNode]:
    """
    Parse an Airflow DAG Python file to extract task dependencies.

    Uses regex-based heuristics to find:
      - task_a >> task_b dependency chains
      - Operator instantiations (BashOperator, PythonOperator, etc.)

    This is a best-effort parser; complex dynamic DAGs may not be
    fully captured.

    Args:
        file_path: Path to the Airflow DAG Python file.
        repo_root: Repository root for relative paths.

    Returns:
        List of TransformationNodes representing task dependencies.
    """
    relative_path = str(file_path.relative_to(repo_root)).replace("\\", "/")

    try:
        source_text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read DAG file %s: %s", relative_path, exc)
        return []

    transformations: list[TransformationNode] = []

    # Find task dependency chains: task_a >> task_b
    dependencies = TASK_DEPENDENCY_PATTERN.findall(source_text)
    for upstream, downstream in dependencies:
        transformations.append(
            TransformationNode(
                source_datasets=[upstream],
                target_datasets=[downstream],
                transformation_type=TransformationType.CONFIG_PIPELINE,
                source_file=relative_path,
                line_range=(1, len(source_text.split("\n"))),
            )
        )

    return transformations
