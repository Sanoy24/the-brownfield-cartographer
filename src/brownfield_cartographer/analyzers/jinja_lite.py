"""Render the literal-list Jinja that dbt models commonly use, and nothing else.

Supported: `{% set name = <literal> %}` (a string, number, or list of them,
read with `ast.literal_eval`), `{% for item in name_or_literal_list %}`
(nested), `{{ name }}`, and `{% if [not] loop.first|loop.last %}` with an
optional `{% else %}`, plus `-` whitespace control. Any other tag, filter,
call or undefined name makes the whole text unrenderable (None), so callers
report it rather than guess. No template code is executed.
"""
from __future__ import annotations

import ast
import re
from typing import Any

_TAG = re.compile(r"(\{%-?.*?-?%\}|\{\{-?.*?-?\}\}|\{#.*?#\})", re.DOTALL)
_NAME = r"[A-Za-z_]\w*"
_SET = re.compile(rf"set\s+({_NAME})\s*=\s*(.+)", re.DOTALL)
_FOR = re.compile(rf"for\s+({_NAME})\s+in\s+(.+)", re.DOTALL)
_CONDITIONS = {"loop.first", "not loop.first", "loop.last", "not loop.last"}
MAX_OUTPUT = 1_000_000


class _Unsupported(Exception):
    pass


def _literal(text: str) -> Any:
    try:
        value = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError, MemoryError, RecursionError) as exc:
        raise _Unsupported from exc
    scalar = (str, int, float)
    if isinstance(value, scalar) and not isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, scalar) and not isinstance(v, bool) for v in value):
        return list(value)
    raise _Unsupported


def _tokens(text: str) -> list[tuple[str, str]]:
    """("text" | "stmt" | "expr", content) with whitespace control applied."""
    parts = _TAG.split(text)
    tokens: list[tuple[str, str]] = []
    strip_next = False
    for index, part in enumerate(parts):
        if index % 2 == 0:
            if strip_next:
                part = part.lstrip()
            tokens.append(("text", part))
            continue
        if part.startswith("{#"):
            strip_next = False
            continue
        kind = "stmt" if part.startswith("{%") else "expr"
        inner = part[2:-2]
        if inner.startswith("-"):
            inner = inner[1:]
            if tokens and tokens[-1][0] == "text":
                tokens[-1] = ("text", tokens[-1][1].rstrip())
        strip_next = inner.endswith("-")
        if strip_next:
            inner = inner[:-1]
        tokens.append((kind, inner.strip()))
    return tokens


def _parse(tokens: list[tuple[str, str]], position: int, closers: set[str]) -> tuple[list, int, str | None]:
    """Nodes until one of `closers`; returns (nodes, next position, closer found)."""
    nodes: list = []
    while position < len(tokens):
        kind, content = tokens[position]
        position += 1
        if kind != "stmt":
            nodes.append((kind, content))
            continue
        keyword = content.split(None, 1)[0] if content else ""
        if keyword in closers:
            return nodes, position, keyword
        if (match := _SET.fullmatch(content)):
            nodes.append(("set", match[1], _literal(match[2])))
        elif (match := _FOR.fullmatch(content)):
            body, position, closer = _parse(tokens, position, {"endfor"})
            if closer != "endfor":
                raise _Unsupported
            nodes.append(("for", match[1], match[2].strip(), body))
        elif keyword == "if" and content[2:].strip() in _CONDITIONS:
            then, position, closer = _parse(tokens, position, {"else", "endif"})
            otherwise: list = []
            if closer == "else":
                otherwise, position, closer = _parse(tokens, position, {"endif"})
            if closer != "endif":
                raise _Unsupported
            nodes.append(("if", content[2:].strip(), then, otherwise))
        else:
            raise _Unsupported
    if closers:
        raise _Unsupported
    return nodes, position, None


def _render(nodes: list, scope: dict[str, Any], out: list[str], size: list[int]) -> None:
    for node in nodes:
        kind = node[0]
        if kind == "text":
            out.append(node[1])
            size[0] += len(node[1])
        elif kind == "expr":
            value = scope.get(node[1]) if re.fullmatch(_NAME, node[1]) else None
            if value is None or isinstance(value, (list, dict)):
                raise _Unsupported
            out.append(str(value))
            size[0] += len(out[-1])
        elif kind == "set":
            scope[node[1]] = node[2]
        elif kind == "for":
            _, name, source, body = node
            items = scope.get(source) if re.fullmatch(_NAME, source) else _literal(source)
            if not isinstance(items, list):
                raise _Unsupported
            for index, item in enumerate(items):
                inner = {**scope, name: item, "__loop": (index == 0, index == len(items) - 1)}
                _render(body, inner, out, size)
        elif kind == "if":
            _, condition, then, otherwise = node
            if "__loop" not in scope:
                raise _Unsupported
            first, last = scope["__loop"]
            value = first if "first" in condition else last
            _render(then if value != condition.startswith("not") else otherwise, scope, out, size)
        if size[0] > MAX_OUTPUT:
            raise _Unsupported


def render(text: str) -> str | None:
    """Rendered text, or None when the template uses anything beyond the supported subset."""
    try:
        nodes, _, _ = _parse(_tokens(text), 0, set())
        out: list[str] = []
        _render(nodes, {}, out, [0])
    except _Unsupported:
        return None
    return "".join(out)
