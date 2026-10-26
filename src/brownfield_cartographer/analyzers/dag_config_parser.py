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

import yaml

from brownfield_cartographer.analyzers import jinja_lite
from brownfield_cartographer.models.schemas import (
    ConfiguresEdge,
    DatasetNode,
    StorageType,
    TransformationNode,
    TransformationType,
)

logger = logging.getLogger(__name__)

# Regex to match dbt ref('model_name') and source('schema', 'table') calls
_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.DOTALL)
_JINJA_CONFIG = re.compile(r"\{\{\s*config\(.*?\)\s*\}\}", re.DOTALL)
_REF_RENDER = re.compile(r"""\{\{\s*ref\(\s*(?:['"]\w+['"]\s*,\s*)?['"](\w+)['"]\s*\)\s*\}\}""")
_SOURCE_RENDER = re.compile(r"""\{\{\s*source\(\s*['"](\w+)['"]\s*,\s*['"](\w+)['"]\s*\)\s*\}\}""")


def render_simple_model(sql_text: str, macros: dict[str, jinja_lite.Macro] | None = None) -> str | None:
    """Render a model whose Jinja is comments, config(), ref(), source() and literal set/for loops.

    Calls to `macros` (the project's own, from `project_macros`) are expanded when
    their arguments are literals. Anything else (package macros, adapter calls,
    other conditions) returns None.
    """
    text = _JINJA_CONFIG.sub("", _JINJA_COMMENT.sub("", sql_text))
    text = _SOURCE_RENDER.sub(r"\1.\2", _REF_RENDER.sub(r"\1", text))
    if "{{" in text or "{%" in text:
        rendered = jinja_lite.render(text, macros)
        if rendered is None or "{{" in rendered or "{%" in rendered:
            return None
        text = rendered
    return text.strip().rstrip(";")


def project_macros(files: list[Path]) -> dict[str, jinja_lite.Macro]:
    """Macros defined in a dbt project's `.sql` files, with ref() and source() pre-rendered."""
    texts = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "macro" in text:
            texts.append(_SOURCE_RENDER.sub(r"\1.\2", _REF_RENDER.sub(r"\1", _JINJA_COMMENT.sub("", text))))
    return jinja_lite.parse_macros(*texts)


REF_PATTERN = re.compile(r"""\{\{\s*ref\(\s*['"](\w+)['"]\s*\)\s*\}\}""")
SOURCE_PATTERN = re.compile(
    r"""\{\{\s*source\(\s*['"](\w+)['"]\s*,\s*['"](\w+)['"]\s*\)\s*\}\}"""
)


# ---------------------------------------------------------------------------
# dbt Project Parsing
# ---------------------------------------------------------------------------

_DIALECT_VOTES: dict[int, str | None] = {}


def _project_dialect(texts: list[str]) -> str | None:
    """The first candidate dialect that parses the most rendered models; None if all parse as many."""
    import sqlglot

    from brownfield_cartographer.analyzers.sql_lineage import CANDIDATE_DIALECTS
    key = hash(tuple(texts))
    if key not in _DIALECT_VOTES:
        scores = {}
        for candidate in CANDIDATE_DIALECTS:
            parsed = 0
            for text in texts:
                try:
                    sqlglot.parse(text, read=candidate)
                    parsed += 1
                except (sqlglot.errors.SqlglotError, ValueError, RecursionError):
                    pass
            scores[candidate] = parsed
        best = max(scores.values(), default=0)
        winners = [name for name, score in scores.items() if score == best]
        if len(_DIALECT_VOTES) > 32:
            _DIALECT_VOTES.clear()
        # When every candidate parses as many models there is no signal; keep per-file choice.
        _DIALECT_VOTES[key] = winners[0] if best and len(winners) < len(scores) else None
    return _DIALECT_VOTES[key]


def parse_dbt_project(project_root: Path, source_files: tuple[Path, ...] | None = None,
                      repo_root: Path | None = None, dialect: str | None = None) -> dict:
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

    Evidence paths are relative to `repo_root`, which defaults to the project root.
    """
    repo_root = repo_root or project_root
    transformations: list[TransformationNode] = []
    datasets: list[DatasetNode] = []
    config_edges: list[ConfiguresEdge] = []
    column_skipped = 0
    if source_files is None:
        from brownfield_cartographer.scanner import scan_repository
        source_files = scan_repository(project_root).files
    allowed = set(source_files)

    # --- Parse dbt_project.yml for metadata ---
    project_file = project_root / "dbt_project.yml"
    project_config = {}
    if project_file in allowed:
        try:
            project_config = yaml.safe_load(project_file.read_text(encoding="utf-8")) or {}
            project_name = project_config.get("name", "unknown_dbt_project")
            logger.info("Parsing dbt project: %s", project_name)
        except Exception as exc:
            logger.warning("Failed to parse dbt_project.yml: %s", exc)

    # --- Scan model SQL files for ref() and source() dependencies ---
    model_paths = project_config.get("model-paths", ["models"])
    model_dirs = [(project_root / p).resolve() for p in model_paths]
    if any(not p.is_relative_to(project_root.resolve()) for p in model_dirs):
        raise ValueError("dbt model-paths must stay within the repository")
    model_files = [p for p in source_files if any(p.is_relative_to(d) for d in model_dirs)]
    macros = project_macros([p for p in source_files if p.suffix == ".sql" and p.is_relative_to(project_root)])
    models = []
    for sql_file in (p for p in model_files if p.suffix == ".sql"):
        sql_text = sql_file.read_text(encoding="utf-8", errors="replace")
        models.append((sql_file, sql_text, render_simple_model(sql_text, macros)))
    # A project runs on one adapter: trace every model in the dialect that parses the most of them.
    project_dialect = dialect or _project_dialect([rendered for *_, rendered in models if rendered])
    if models:
        for sql_file, sql_text, rendered in models:
            relative_path = sql_file.relative_to(repo_root).as_posix()
            model_name = sql_file.stem

            # Extract ref() dependencies → these are upstream dbt models
            ref_deps = REF_PATTERN.findall(sql_text)
            source_deps = SOURCE_PATTERN.findall(sql_text)

            source_datasets = list(ref_deps)
            for schema_name, table_name in source_deps:
                source_datasets.append(f"{schema_name}.{table_name}")

            # Column lineage from the rendered model; models with other Jinja are skipped.
            columns, unresolved = [], []
            if rendered is None:
                column_skipped += 1
            else:
                from brownfield_cartographer.analyzers.sql_lineage import sql_statements
                errors: list[dict] = []
                model_sql = f"CREATE TABLE {model_name} AS {rendered}"
                statements = sql_statements(model_sql, relative_path, project_dialect, errors)
                if errors and project_dialect and not dialect:
                    errors = []
                    statements = sql_statements(model_sql, relative_path, None, errors)
                if errors or len(statements) != 1:
                    column_skipped += 1
                else:
                    columns, unresolved = statements[0]["columns"], statements[0]["unresolved_columns"]
                    # Tables read through macros (a ref() inside a macro body) appear only once rendered.
                    source_datasets += [t for t in statements[0]["sources"] if t not in source_datasets]

            transformations.append(
                TransformationNode(
                    source_datasets=source_datasets,
                    target_datasets=[model_name],
                    transformation_type=TransformationType.DBT_MODEL,
                    source_file=relative_path,
                    line_range=(1, len(sql_text.split("\n"))),
                    sql_query_if_applicable=sql_text[:500],
                    column_lineage=columns, unresolved_columns=unresolved,
                )
            )

    # --- Parse schema YAML files for dataset metadata ---
    for yml_file in (p for p in model_files if p.suffix in {".yml", ".yaml"}):
        relative_yml = yml_file.relative_to(repo_root).as_posix()
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
        "column_skipped": column_skipped,
    }
