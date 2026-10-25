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

import copy
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
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


@dataclass
class SchemaContext:
    """Known table columns for column lineage, and the column lists this run defines."""
    tables: dict[str, list[str]]
    # Table -> every column it is defined with; None when definitions disagree.
    declared: dict[str, list[str] | None] = field(default_factory=dict)
    # Parse results reused across passes while the columns of the tables a text mentions are unchanged.
    cache: dict = field(default_factory=dict)

    def declare(self, table: str, columns: list[str] | None) -> None:
        table = table.lower()
        columns = [c.lower() for c in columns] if columns is not None else None
        if columns is None or (table in self.declared and self.declared[table] != columns):
            self.declared[table] = None
        else:
            self.declared[table] = columns


_SCHEMA: ContextVar[SchemaContext | None] = ContextVar("cartographer_sql_schema", default=None)


@contextmanager
def known_schema(tables: dict[str, list[str]], cache: dict | None = None) -> Iterator[SchemaContext]:
    """Resolve column lineage against `tables` ({qualified name: columns}) within this block.

    Pass the same `cache` to successive blocks to reuse unaffected parse results.
    """
    context = SchemaContext({name.lower(): [c.lower() for c in columns] for name, columns in tables.items()},
                            cache=cache if cache is not None else {})
    token = _SCHEMA.set(context)
    try:
        yield context
    finally:
        _SCHEMA.reset(token)


def _projection(expression: exp.Expression) -> list[str] | None:
    """Output column names of a query, or None while any is unknown (such as an unexpanded `*`)."""
    while isinstance(expression, exp.SetOperation):
        expression = expression.this
    if not isinstance(expression, exp.Select):
        return None
    names = []
    for select in expression.selects:
        if isinstance(select, exp.Star) or (isinstance(select, exp.Column) and isinstance(select.this, exp.Star)):
            return None
        if not select.alias_or_name:
            return None
        names.append(select.alias_or_name.lower())
    return names


def _modified(star: exp.Star) -> bool:
    """`* EXCEPT (...)`, `* REPLACE (...)` and similar select a changed column list; they are not expanded."""
    return any(star.args.get(key) for key in ("except_", "replace", "rename", "ilike"))


def _resolve_with_schema(query: exp.Expression, tables: dict[str, list[str]]) -> None:
    """Qualify unqualified columns and expand stars where the columns of every source are known.

    A column is assigned to a source only when all sources of its scope are
    known and exactly one has it. Stars are expanded over known sources unless
    two of them share a column name. Inner scopes come first, so expanded CTEs
    and subqueries become known to the queries that read them.
    """
    for scope in traverse_scope(query):
        known: dict[str, list[str] | None] = {}
        for name, (_, source) in scope.selected_sources.items():
            if isinstance(source, exp.Table):
                known[name] = tables.get(_qualified(source).lower())
            else:
                known[name] = _projection(source.expression)
        if len(known) == 1:
            # With a single source, an unqualified column can only come from that source.
            [only] = known
            for column in scope.columns:
                if not column.table:
                    column.set("table", exp.to_identifier(only))
        elif known and all(columns is not None for columns in known.values()):
            for column in scope.columns:
                if not column.table:
                    owners = [name for name, columns in known.items() if column.name.lower() in (columns or [])]
                    if len(owners) == 1:
                        column.set("table", exp.to_identifier(owners[0]))
        if not isinstance(scope.expression, exp.Select):
            continue
        expanded: list[exp.Expression] = []
        for select in scope.expression.selects:
            if isinstance(select, exp.Star) and not _modified(select):
                pairs = [(n, c) for n, columns in known.items() for c in (columns or [])]
                plain = [c for _, c in pairs]
                if pairs and all(columns is not None for columns in known.values()) and len(set(plain)) == len(plain):
                    expanded += [exp.column(c, table=n) for n, c in pairs]
                    continue
            elif isinstance(select, exp.Column) and isinstance(select.this, exp.Star) and not _modified(select.this):
                columns = known.get(select.table)
                if columns:
                    expanded += [exp.column(c, table=select.table) for c in columns]
                    continue
            expanded.append(select)
        scope.expression.set("expressions", expanded)


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
    """Map each written column to its source columns. `*` and ambiguous columns are unresolved.

    Known table columns are applied first. Some queries that trace without them
    fail with them (sqlglot cannot join `USING` a column one known side lacks),
    so a schema never leaves more columns unresolved than tracing without it.
    """
    context = _SCHEMA.get()
    tables = context.tables if context else {}
    result = _trace_columns(statement, dialect, tables)
    if tables and result[1]:
        plain = _trace_columns(statement, dialect, {})
        if len(plain[1]) < len(result[1]):
            return plain
    return result


def _trace_columns(statement: exp.Expression, dialect: str,
                   tables: dict[str, list[str]]) -> tuple[list[dict], list[str]]:
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
        _resolve_with_schema(query, tables)
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
    context = _SCHEMA.get()
    if context is None:
        return _parse_text(sql_text, source_file, dialect)
    # Reuse the previous pass's result while the tables this text mentions have the same known columns.
    lowered = sql_text.lower()
    relevant = tuple(sorted((table, tuple(columns)) for table, columns in context.tables.items()
                            if table.rsplit(".", 1)[-1] in lowered))
    key = (sql_text, source_file, dialect, relevant)
    if key not in context.cache:
        own = SchemaContext(context.tables)
        token = _SCHEMA.set(own)
        try:
            context.cache[key] = (_parse_text(sql_text, source_file, dialect), own.declared)
        finally:
            _SCHEMA.reset(token)
    result, declared = context.cache[key]
    for table, columns in declared.items():
        context.declare(table, columns)
    return copy.deepcopy(result)


def _parse_text(sql_text: str, source_file: str, dialect: str) -> tuple[list[dict], list[dict]]:
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
        if (context := _SCHEMA.get()) is not None and isinstance(statement, exp.Create):
            # A table's schema comes from its full definition, never from an INSERT,
            # which may fill only some columns.
            target = _target(statement, source_file)
            if isinstance(statement.this, exp.Schema) and statement.expression is None:
                defined = [c.name for c in statement.this.expressions if isinstance(c, exp.ColumnDef)]
                if defined:
                    context.declare(target, defined)
            elif columns and not unresolved:
                context.declare(target, [c["target"] for c in columns])
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
