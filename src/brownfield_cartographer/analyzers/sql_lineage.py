"""Conservative statement-level SQL reads/writes with exact statement spans."""
from __future__ import annotations

from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from sqlglot.tokens import TokenType

from brownfield_cartographer.models.schemas import (
    TransformationNode,
    TransformationType,
)

SUPPORTED_DIALECTS = ("postgres", "bigquery", "snowflake", "duckdb", "sqlite")
DEFAULT_DIALECT = "duckdb"


def _qualified(table: exp.Table) -> str:
    return ".".join(part.name for part in table.parts)


def _extract_source_tables(expression: exp.Expression) -> set[str]:
    query = expression.expression if isinstance(expression, (exp.Create, exp.Insert)) else expression
    if query is None:
        return set()
    return {_qualified(source) for scope in traverse_scope(query)
            for source in scope.sources.values() if isinstance(source, exp.Table)}


def _target(expression: exp.Expression, source_file: str) -> str:
    if isinstance(expression, (exp.Create, exp.Insert)):
        table = expression.this
        if isinstance(table, exp.Schema):
            table = table.this
        if isinstance(table, exp.Table):
            return _qualified(table)
    return Path(source_file).stem


def _statements(text: str, dialect: str):
    tokens = sqlglot.Dialect.get_or_raise(dialect).tokenize(text)
    boundaries = [token.end + 1 for token in tokens if token.token_type == TokenType.SEMICOLON]
    if not boundaries or boundaries[-1] < len(text):
        boundaries.append(len(text))
    start = 0
    for end in boundaries:
        local = [token for token in tokens if start <= token.start < end
                 and token.token_type != TokenType.SEMICOLON]
        if local:
            yield text[start:end], (text.count("\n", 0, local[0].start) + 1,
                                    text.count("\n", 0, local[-1].end) + 1)
        start = end


def sql_statements(sql_text: str, source_file: str = "<inline>",
                   dialect: str = DEFAULT_DIALECT, errors: list[dict] | None = None) -> list[dict]:
    """Return each statement's source tables, written table (None for queries), span and text.

    Without `errors`, the first invalid or unsupported statement raises ValueError.
    With `errors`, each such statement is appended there as {"line", "error"} and
    the remaining statements are still returned.
    """
    if not sql_text.strip():
        return []
    try:
        segments = list(_statements(sql_text, dialect))
    except sqlglot.errors.SqlglotError as exc:
        raise ValueError(f"Invalid SQL in {source_file}: {exc}") from exc
    results = []
    for text, span in segments:
        try:
            statement = sqlglot.parse_one(text, dialect=dialect)
            if not isinstance(statement, (exp.Select, exp.Union, exp.Create, exp.Insert)):
                raise ValueError(f"Unsupported SQL statement in {source_file}: {statement.key}")
        except (sqlglot.errors.SqlglotError, ValueError) as exc:
            message = str(exc) if isinstance(exc, ValueError) else f"Invalid SQL in {source_file}: {exc}"
            if errors is None:
                raise ValueError(message) from exc
            errors.append({"line": span[0], "error": message})
            continue
        writes = isinstance(statement, (exp.Create, exp.Insert))
        results.append({"sources": sorted(_extract_source_tables(statement)),
                        "target": _target(statement, source_file) if writes else None,
                        "line_range": span, "text": text.strip()})
    return results


def analyze_sql_string(sql_text: str, source_file: str = "<inline>",
                       dialect: str = DEFAULT_DIALECT, errors: list[dict] | None = None) -> list[TransformationNode]:
    transformations = []
    for statement in sql_statements(sql_text, source_file, dialect, errors):
        if statement["sources"]:
            transformations.append(TransformationNode(
                source_datasets=statement["sources"],
                target_datasets=[statement["target"] or Path(source_file).stem],
                transformation_type=TransformationType.SQL_QUERY, source_file=source_file,
                line_range=statement["line_range"], sql_query_if_applicable=statement["text"][:500],
            ))
    return transformations


def analyze_sql_file(file_path: Path, repo_root: Path, dialect: str = DEFAULT_DIALECT,
                     errors: list[dict] | None = None) -> list[TransformationNode]:
    return analyze_sql_string(file_path.read_text(encoding="utf-8"),
                              file_path.relative_to(repo_root).as_posix(), dialect, errors)
