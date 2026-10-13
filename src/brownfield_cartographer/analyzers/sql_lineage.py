"""Conservative statement-level SQL reads/writes with exact statement spans.

Queries (SELECT, UNION) read tables. CREATE ... AS and INSERT write their target
from the query they contain. UPDATE, DELETE and MERGE write their target from
every other table the statement reads (FROM/USING clauses, joins, subqueries).
Statements without table-to-table lineage (DROP, ALTER, transactions, indexes)
are skipped; anything else, such as COPY, is reported as unsupported.

Without an explicit dialect, each text is parsed with candidate dialects and
the one with the fewest invalid or unsupported statements is used; its name is
recorded as the analysis method (for example `sql:bigquery`).
"""
from __future__ import annotations

import logging
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from sqlglot.tokens import TokenType

from brownfield_cartographer.models.schemas import (
    TransformationNode,
    TransformationType,
)

CANDIDATE_DIALECTS = ("duckdb", "bigquery", "snowflake", "postgres", "mysql", "spark", "tsql", "trino")
# Commands sqlglot cannot model structurally but that carry no table lineage.
_NO_LINEAGE_COMMANDS = {"GRANT", "REVOKE", "SET", "USE", "VACUUM", "SHOW", "DESCRIBE", "COMMENT"}
_NO_LINEAGE_CREATES = {"CATALOG", "DATABASE", "SCHEMA", "ROLE", "USER", "FUNCTION", "INDEX"}


class _DropCommandFallbackWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Falling back to parsing as a 'Command'" not in record.getMessage()


logging.getLogger("sqlglot").addFilter(_DropCommandFallbackWarnings())


def validate_dialect(name: str) -> str:
    try:
        sqlglot.Dialect.get_or_raise(name)
    except ValueError as exc:
        raise ValueError(f"Unsupported SQL dialect: {name}") from exc
    return name


def _qualified(table: exp.Table) -> str:
    return ".".join(part.name for part in table.parts)


DML = (exp.Update, exp.Delete, exp.Merge)
SUPPORTED_STATEMENTS = (exp.Select, exp.Union, exp.Create, exp.Insert, *DML)
NO_LINEAGE = (exp.Drop, exp.TruncateTable, exp.Alter, exp.Transaction, exp.Commit, exp.Rollback,
              exp.Set, exp.Use, exp.Grant, exp.Analyze)


def _has_no_lineage(statement: exp.Expression) -> bool:
    if isinstance(statement, exp.Command):
        keyword = str(statement.this).upper()
        following = str(statement.expression or "").split()
        return keyword in _NO_LINEAGE_COMMANDS or (
            keyword == "CREATE" and bool(following) and following[0].upper() in _NO_LINEAGE_CREATES)
    if isinstance(statement, exp.Create):
        return statement.expression is None and str(statement.args.get("kind", "")).upper() != "TABLE"
    return isinstance(statement, NO_LINEAGE)


def _dml_sources(statement: exp.Expression) -> set[str]:
    target = statement.this
    ctes = {cte.alias for cte in statement.find_all(exp.CTE)}
    names = {_qualified(table) for table in statement.find_all(exp.Table)
             if table is not target and not (len(table.parts) == 1 and table.name in ctes)}
    return names - {_qualified(target)} if isinstance(target, exp.Table) else names


def _extract_source_tables(expression: exp.Expression) -> set[str]:
    if isinstance(expression, DML):
        return _dml_sources(expression)
    query = expression.expression if isinstance(expression, (exp.Create, exp.Insert)) else expression
    if query is None:
        return set()
    return {_qualified(source) for scope in traverse_scope(query)
            for source in scope.sources.values() if isinstance(source, exp.Table)}


def _target(expression: exp.Expression, source_file: str) -> str:
    if isinstance(expression, (exp.Create, exp.Insert, *DML)):
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


def _parse(sql_text: str, source_file: str, dialect: str) -> tuple[list[dict], list[dict]]:
    try:
        segments = list(_statements(sql_text, dialect))
    except sqlglot.errors.SqlglotError as exc:
        return [], [{"line": 1, "error": f"Invalid SQL in {source_file}: {exc}"}]
    results, errors = [], []
    for text, span in segments:
        try:
            statement = sqlglot.parse_one(text, dialect=dialect)
            if _has_no_lineage(statement):
                continue
            if not isinstance(statement, SUPPORTED_STATEMENTS):
                raise ValueError(f"Unsupported SQL statement in {source_file}: {statement.key}")
        except (sqlglot.errors.SqlglotError, ValueError) as exc:
            message = str(exc) if isinstance(exc, ValueError) else f"Invalid SQL in {source_file}: {exc}"
            errors.append({"line": span[0], "error": message})
            continue
        writes = isinstance(statement, (exp.Create, exp.Insert, *DML))
        results.append({"sources": sorted(_extract_source_tables(statement)),
                        "target": _target(statement, source_file) if writes else None,
                        "line_range": span, "text": text.strip(), "dialect": dialect})
    return results, errors


def sql_statements(sql_text: str, source_file: str = "<inline>",
                   dialect: str | None = None, errors: list[dict] | None = None) -> list[dict]:
    """Return each statement's source tables, written table (None for queries), span, text and dialect.

    With `dialect` None, candidate dialects are tried and the best fit is used.
    Without `errors`, the first invalid or unsupported statement raises ValueError.
    With `errors`, each such statement is appended there as {"line", "error"} and
    the remaining statements are still returned.
    """
    if not sql_text.strip():
        return []
    best: tuple[list[dict], list[dict]] | None = None
    for candidate in (dialect,) if dialect else CANDIDATE_DIALECTS:
        attempt = _parse(sql_text, source_file, candidate)
        if best is None or len(attempt[1]) < len(best[1]):
            best = attempt
        if not attempt[1]:
            break
    assert best is not None
    results, problems = best
    if problems and errors is None:
        raise ValueError(problems[0]["error"])
    if errors is not None:
        errors.extend(problems)
    return results


def analyze_sql_string(sql_text: str, source_file: str = "<inline>",
                       dialect: str | None = None, errors: list[dict] | None = None) -> list[TransformationNode]:
    transformations = []
    statements = sql_statements(sql_text, source_file, dialect, errors)
    # A file that is a single query is treated as defining a dataset named after
    # the file (a view or model file); queries inside scripts only read.
    single_query = len(statements) == 1 and statements[0]["target"] is None
    for statement in statements:
        if not statement["sources"] or (statement["target"] is None and not single_query):
            continue
        inferred = statement["target"] is None
        transformations.append(TransformationNode(
            source_datasets=statement["sources"],
            target_datasets=[statement["target"] or Path(source_file).stem],
            transformation_type=TransformationType.SQL_QUERY, source_file=source_file,
            line_range=statement["line_range"], sql_query_if_applicable=statement["text"][:500],
            analysis_method=f"sql:{statement['dialect']}" + ("+file_name" if inferred else ""),
            confidence="inferred" if inferred else "syntax",
        ))
    return transformations


def analyze_sql_file(file_path: Path, repo_root: Path, dialect: str | None = None,
                     errors: list[dict] | None = None) -> list[TransformationNode]:
    return analyze_sql_string(file_path.read_text(encoding="utf-8"),
                              file_path.relative_to(repo_root).as_posix(), dialect, errors)
