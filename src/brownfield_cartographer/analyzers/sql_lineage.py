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
from sqlglot.lineage import Node, lineage
from sqlglot.optimizer.qualify import qualify
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


def _leaves(node: Node, parent: Node | None = None) -> list[tuple[Node, str | None]]:
    """Leaf nodes with their column name. A pass-through `SELECT *` keeps the name asked for above it."""
    if node.downstream:
        return [leaf for child in node.downstream for leaf in _leaves(child, node)]
    name = node.name.split(".")[-1]
    if name == "*":
        name = parent.name.split(".")[-1] if parent is not None else "*"
    return [(node, None if name == "*" else name.lower())]


def _is_copy(node: Node) -> bool:
    """True when every step down to the sources is a column reference or a pass-through `SELECT *`."""
    expression = node.expression.unalias() if isinstance(node.expression, exp.Expression) else None
    if node.downstream and not isinstance(expression, (exp.Column, exp.Star)):
        return False
    return all(_is_copy(child) for child in node.downstream)


def _column_lineage(statement: exp.Expression, dialect: str) -> tuple[list[dict], list[str]]:
    """Map each written column to its source columns. `*` and ambiguous columns are unresolved."""
    query = statement.expression if isinstance(statement, (exp.Create, exp.Insert)) else None
    if not isinstance(query, exp.Query):
        return [], []
    query = query.copy()
    if statement.args.get("with_") and not query.args.get("with_"):
        query.set("with_", statement.args["with_"].copy())
    if isinstance(query, exp.SetOperation) and query.args.get("with_"):
        # Column lineage resolves CTEs per branch: give each branch the CTEs it uses.
        ctes = query.args["with_"]
        names = {cte.alias for cte in ctes.expressions}
        query.set("with_", None)
        branches, pending = [], [query]
        while pending:
            current = pending.pop()
            if isinstance(current, exp.SetOperation):
                pending += [current.this, current.expression]
            else:
                branches.append(current)
        for branch in branches:
            if any(len(table.parts) == 1 and table.name in names for table in branch.find_all(exp.Table)):
                branch.set("with_", ctes.copy())
    try:
        # With a single source, an unqualified column can only come from that source.
        for scope in traverse_scope(query):
            if len(scope.selected_sources) == 1:
                [source_name] = scope.selected_sources
                for column in scope.columns:
                    if not column.table:
                        column.set("table", exp.to_identifier(source_name))
    except (sqlglot.errors.SqlglotError, ValueError, KeyError):
        pass
    try:
        # Expand `SELECT *` over CTEs and subqueries with known columns (no schema needed).
        query = qualify(query, schema={}, dialect=dialect, expand_stars=True,
                        validate_qualify_columns=False, identify=False)
    except (sqlglot.errors.SqlglotError, ValueError, KeyError):
        pass
    first = query
    while isinstance(first, exp.SetOperation):
        first = first.this
    if not isinstance(first, exp.Select):
        return [], []
    schema = statement.this if isinstance(statement.this, exp.Schema) else None
    insert_columns = [c.name for c in schema.expressions] if isinstance(statement, exp.Insert) and schema else []
    # Unnamed expressions are traced under a positional alias; only an INSERT column list names them.
    for index, select in enumerate(list(first.selects)):
        if not select.alias_or_name:
            select.replace(exp.alias_(select.copy(), f"_cartographer_col_{index}"))
    outputs = [(s.alias_or_name, s.unalias()) for s in first.selects]
    try:
        nodes = lineage(None, query, {}, dialect=dialect)
    except (sqlglot.errors.SqlglotError, ValueError, KeyError):
        return [], [name.lower() for name, _ in outputs]
    columns, unresolved = [], []
    for index, (name, expression) in enumerate(outputs):
        target = (insert_columns[index] if index < len(insert_columns) else name).lower()
        node = nodes.get(name)
        if target.startswith("_cartographer_col_"):
            unresolved.append(f"#{index + 1}")
            continue
        if name == "*" or isinstance(expression, exp.Star) or node is None:
            unresolved.append(target if name != "*" else "*")
            continue
        leaves = _leaves(node)
        if any(isinstance(leaf.source, exp.Placeholder) or (isinstance(leaf.source, exp.Table) and column is None)
               for leaf, column in leaves):
            unresolved.append(target)
            continue
        sources = sorted({f"{_qualified(leaf.source)}::{column}"
                          for leaf, column in leaves if isinstance(leaf.source, exp.Table)})
        columns.append({"target": target, "sources": sources,
                        "kind": "copy" if isinstance(expression, exp.Column) and _is_copy(node) else "derived"})
    return columns, unresolved


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
        columns, unresolved = _column_lineage(statement, dialect) if writes else ([], [])
        results.append({"sources": sorted(_extract_source_tables(statement)),
                        "target": _target(statement, source_file) if writes else None,
                        "line_range": span, "text": text.strip(), "dialect": dialect,
                        "columns": columns, "unresolved_columns": unresolved})
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
            column_lineage=statement["columns"], unresolved_columns=statement["unresolved_columns"],
        ))
    return transformations


def analyze_sql_file(file_path: Path, repo_root: Path, dialect: str | None = None,
                     errors: list[dict] | None = None) -> list[TransformationNode]:
    return analyze_sql_string(file_path.read_text(encoding="utf-8"),
                              file_path.relative_to(repo_root).as_posix(), dialect, errors)
