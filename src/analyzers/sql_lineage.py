"""
SQL Lineage Analyzer — extracts table-level data lineage from SQL files.

Uses sqlglot to parse SQL queries across multiple dialects and build a
dependency graph of source tables → target tables (via SELECT/FROM/JOIN/CTE).
This is the core engine for the Hydrologist agent's SQL lineage detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import sqlglot
from sqlglot import exp

from src.models.schemas import TransformationNode, TransformationType

logger = logging.getLogger(__name__)

# Dialects supported by the analyzer (sqlglot dialect names)
SUPPORTED_DIALECTS = ("postgres", "bigquery", "snowflake", "duckdb", "sqlite")
DEFAULT_DIALECT = "duckdb"


def _extract_source_tables(expression: exp.Expression) -> set[str]:
    """
    Recursively extract all source table references from a parsed SQL AST.

    Walks the expression tree to find FROM and JOIN clauses,
    resolving schema-qualified names (e.g., schema.table → schema.table).
    Filters out CTE names that shadow real tables.
    """
    tables: set[str] = set()
    cte_names: set[str] = set()

    # Collect CTE names so we can exclude them from source tables
    for cte in expression.find_all(exp.CTE):
        alias = cte.args.get("alias")
        if alias:
            cte_names.add(alias.name.lower())

    # Find all Table references
    for table in expression.find_all(exp.Table):
        table_name = table.name
        if not table_name:
            continue

        # Build qualified name: schema.table or just table
        schema = table.args.get("db")
        if schema:
            qualified = f"{schema.name}.{table_name}"
        else:
            qualified = table_name

        # Skip CTE self-references
        if qualified.lower() not in cte_names:
            tables.add(qualified)

    return tables


def _infer_target_name(sql_text: str, file_path: str) -> str:
    """
    Infer the target dataset name for a SQL transformation.

    Strategy (in order):
    1. Look for CREATE TABLE / INSERT INTO statements → use the table name
    2. For dbt-style models (.sql in models/ dir) → use the filename stem
    3. Fallback → use the filename stem
    """
    try:
        parsed = sqlglot.parse(sql_text, error_level=sqlglot.ErrorLevel.IGNORE)
        for statement in parsed:
            if statement is None:
                continue
            # Look for CREATE or INSERT targets
            for create in statement.find_all(exp.Create):
                table = create.find(exp.Table)
                if table and table.name:
                    return table.name
            for insert in statement.find_all(exp.Insert):
                table = insert.find(exp.Table)
                if table and table.name:
                    return table.name
    except Exception:
        pass

    # Default: derive from filename (dbt convention)
    return Path(file_path).stem


def analyze_sql_file(
    file_path: Path,
    repo_root: Path,
    dialect: str = DEFAULT_DIALECT,
) -> list[TransformationNode]:
    """
    Parse a SQL file and extract data lineage as TransformationNodes.

    Each SELECT statement in the file produces one TransformationNode with
    source_datasets (tables read) and target_datasets (inferred from
    CREATE/INSERT or filename).

    Args:
        file_path: Absolute path to the .sql file.
        repo_root: Repository root for computing relative paths.
        dialect: SQL dialect to use for parsing (default: duckdb).

    Returns:
        List of TransformationNodes representing data lineage edges.
    """
    relative_path = str(file_path.relative_to(repo_root)).replace("\\", "/")

    try:
        sql_text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read SQL file %s: %s", relative_path, exc)
        return []

    if not sql_text.strip():
        return []

    transformations: list[TransformationNode] = []

    try:
        statements = sqlglot.parse(sql_text, dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)
    except Exception as exc:
        logger.warning("Failed to parse SQL in %s: %s", relative_path, exc)
        return []

    for statement in statements:
        if statement is None:
            continue

        source_tables = _extract_source_tables(statement)
        if not source_tables:
            continue

        target_name = _infer_target_name(sql_text, relative_path)

        transformations.append(
            TransformationNode(
                source_datasets=sorted(source_tables),
                target_datasets=[target_name],
                transformation_type=TransformationType.SQL_QUERY,
                source_file=relative_path,
                line_range=(1, len(sql_text.split("\n"))),
                sql_query_if_applicable=sql_text[:500],  # truncate for storage
            )
        )

    return transformations


def analyze_sql_string(
    sql_text: str,
    source_file: str = "<inline>",
    dialect: str = DEFAULT_DIALECT,
) -> list[TransformationNode]:
    """
    Parse a raw SQL string and extract lineage.

    Useful for analyzing SQL embedded in Python code or config files.
    """
    if not sql_text.strip():
        return []

    transformations: list[TransformationNode] = []

    try:
        statements = sqlglot.parse(sql_text, dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)
    except Exception as exc:
        logger.warning("Failed to parse inline SQL from %s: %s", source_file, exc)
        return []

    for statement in statements:
        if statement is None:
            continue

        source_tables = _extract_source_tables(statement)
        if not source_tables:
            continue

        transformations.append(
            TransformationNode(
                source_datasets=sorted(source_tables),
                target_datasets=[Path(source_file).stem],
                transformation_type=TransformationType.SQL_QUERY,
                source_file=source_file,
                line_range=(1, 1),
                sql_query_if_applicable=sql_text[:500],
            )
        )

    return transformations
