"""Find SQL passed to database and dataframe APIs in Python source.

Recognizes SQL given to `execute`/`executemany`/`executescript`, `.sql()`
(Spark, DuckDB) and `read_sql`/`read_sql_query`, optionally wrapped in
SQLAlchemy `text()`. SQL is resolved from string literals and from names
assigned exactly once to a string literal in the same function or at module
level. Strings count as SQL only when they start with a SQL keyword. SQL built
with f-strings, concatenation or formatting is reported as unresolved.
"""

from __future__ import annotations

import ast
import re
from collections import Counter

SQL_METHODS = {"execute", "executemany", "executescript", "sql", "read_sql", "read_sql_query"}
TABLE_READ_METHODS = {"read_sql_table"}
_SQL_ARGUMENTS = {"sql", "query", "statement", "operation", "sql_script", "sqlQuery"}
_SQL_START = re.compile(
    r"^\s*(?:(?:--[^\n]*\n|/\*.*?\*/)\s*)*(select|with|insert|create|update|delete|merge|replace)\b",
    re.IGNORECASE | re.DOTALL,
)
_PYFORMAT = re.compile(r"%\(\w+\)s|%s")


def looks_like_sql(text: str) -> bool:
    return bool(_SQL_START.match(text))


def normalize_placeholders(text: str) -> str:
    """Replace DB-API pyformat placeholders, which sqlglot cannot parse, with `?`."""
    return _PYFORMAT.sub("?", text).replace("%%", "%")


def string_constants(scope: ast.AST) -> dict[str, str]:
    """Names assigned exactly once in this scope, to a string literal."""
    counts: Counter[str] = Counter()
    values: dict[str, str] = {}
    for node in _scope_nodes(scope):
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
        for target in targets:
            for name in (n.id for n in ast.walk(target) if isinstance(n, ast.Name)):
                counts[name] += 1
                value = getattr(node, "value", None)
                if (isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(target, ast.Name)
                        and isinstance(value, ast.Constant) and isinstance(value.value, str)):
                    values[name] = value.value
    return {name: value for name, value in values.items() if counts[name] == 1}


def _scope_nodes(scope: ast.AST):
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield child
        yield from _scope_nodes(child)


def _unwrap_text(node: ast.AST) -> ast.AST:
    if isinstance(node, ast.Call) and node.args:
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name == "text":
            return node.args[0]
    return node


def _leading_constant(node: ast.AST) -> str | None:
    """Return the literal text a dynamically built string starts with, if any."""
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        return _leading_constant(node.left)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        return _leading_constant(node.func.value)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def resolve_sql(node: ast.AST, constants: dict[str, str]) -> tuple[str | None, bool]:
    """Return (sql_text, dynamic). Non-SQL or unknown values return (None, False)."""
    node = _unwrap_text(node)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value, False) if looks_like_sql(node.value) else (None, False)
    if isinstance(node, ast.Name) and node.id in constants:
        value = constants[node.id]
        return (value, False) if looks_like_sql(value) else (None, False)
    if isinstance(node, ast.JoinedStr):
        parts = [v.value for v in node.values if isinstance(v, ast.Constant)]
        if len(parts) == len(node.values):
            value = "".join(str(part) for part in parts)
            return (value, False) if looks_like_sql(value) else (None, False)
    prefix = _leading_constant(node)
    if prefix is not None and looks_like_sql(prefix):
        return None, True
    return None, False


def sql_argument(call: ast.Call) -> ast.AST | None:
    if call.args:
        return call.args[0]
    return next((k.value for k in call.keywords if k.arg in _SQL_ARGUMENTS), None)
