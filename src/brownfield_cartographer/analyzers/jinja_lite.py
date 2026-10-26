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
from dataclasses import dataclass
from typing import Any

_TAG = re.compile(r"(\{%-?.*?-?%\}|\{\{-?.*?-?\}\}|\{#.*?#\})", re.DOTALL)
_NAME = r"[A-Za-z_]\w*"
_SET = re.compile(rf"set\s+({_NAME})\s*=\s*(.+)", re.DOTALL)
_FOR = re.compile(rf"for\s+({_NAME})\s+in\s+(.+)", re.DOTALL)
# `[not] loop.first|loop.last`, or `[not] name` for a variable's truthiness.
_CONDITION = re.compile(rf"(?:not\s+)?(?:loop\.first|loop\.last|{_NAME})")
_MACRO = re.compile(
    rf"(?P<open>\{{%-?\s*macro\s+(?P<name>{_NAME})\s*\((?P<params>[^)]*)\)\s*-?%\}})"
    r"(?P<body>.*?)(?P<close>\{%-?\s*endmacro\s*-?%\})", re.DOTALL)
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
        elif keyword == "if" and _CONDITION.fullmatch(content[2:].strip()):
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


_UNDEFINED = object()
MAX_DEPTH = 8


@dataclass
class Macro:
    params: list[str]
    defaults: dict[str, Any]
    body: str
    nodes: list | None = None


@dataclass
class _State:
    macros: dict[str, Macro]
    size: int = 0
    depth: int = 0


def parse_macros(*texts: str) -> dict[str, Macro]:
    """Macros defined in dbt `.sql` files. A name defined more than once, or with
    parameters that are not plain names with literal defaults, is left out."""
    found: dict[str, Macro | None] = {}
    for text in texts:
        for match in _MACRO.finditer(text):
            name, params, body = match["name"], match["params"], match["body"]
            if match["open"].endswith("-%}"):
                body = body.lstrip()
            if match["close"].startswith("{%-"):
                body = body.rstrip()
            macro = _signature(params, body)
            found[name] = None if name in found or macro is None else macro
    return {name: macro for name, macro in found.items() if macro is not None}


def _signature(params: str, body: str) -> Macro | None:
    try:
        function = ast.parse(f"def _m({params}):\n    pass").body[0]
    except (SyntaxError, ValueError, RecursionError):
        return None
    assert isinstance(function, ast.FunctionDef)
    args = function.args
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
        return None
    names = [a.arg for a in args.args]
    defaults: dict[str, Any] = {}
    for name, default in zip(names[len(names) - len(args.defaults):], args.defaults, strict=True):
        if not isinstance(default, ast.Constant) or isinstance(default.value, bool | bytes):
            return None
        defaults[name] = default.value
    return Macro(names, defaults, body)


def _value(node: ast.expr, scope: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, str | int | float) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and scope.get(node.id, _UNDEFINED) is not _UNDEFINED:
        return scope[node.id]
    raise _Unsupported


def _expression(content: str, scope: dict[str, Any], state: _State) -> str:
    if re.fullmatch(_NAME, content):
        value = scope.get(content, _UNDEFINED)
        if value is _UNDEFINED or value is None or isinstance(value, list | dict):
            raise _Unsupported
        return str(value)
    try:
        call = ast.parse(content, mode="eval").body
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise _Unsupported from exc
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in state.macros):
        raise _Unsupported
    macro = state.macros[call.func.id]
    if len(call.args) > len(macro.params) or state.depth >= MAX_DEPTH:
        raise _Unsupported
    bound: dict[str, Any] = dict(zip(macro.params, (_value(a, scope) for a in call.args), strict=False))
    for keyword in call.keywords:
        if keyword.arg not in macro.params or keyword.arg in bound:
            raise _Unsupported
        bound[keyword.arg] = _value(keyword.value, scope)
    for name in macro.params:
        if name not in bound:
            if name not in macro.defaults:
                raise _Unsupported
            bound[name] = macro.defaults[name]
    if macro.nodes is None:
        macro.nodes, _, _ = _parse(_tokens(macro.body), 0, set())
    out: list[str] = []
    state.depth += 1
    try:
        # A macro sees only its own arguments, not the caller's variables.
        _render(macro.nodes, bound, out, state)
    finally:
        state.depth -= 1
    return "".join(out)


def _render(nodes: list, scope: dict[str, Any], out: list[str], state: _State) -> None:
    for node in nodes:
        kind = node[0]
        if kind == "text":
            out.append(node[1])
        elif kind == "expr":
            out.append(_expression(node[1], scope, state))
        elif kind == "set":
            scope[node[1]] = node[2]
        elif kind == "for":
            _, name, source, body = node
            items = scope.get(source) if re.fullmatch(_NAME, source) else _literal(source)
            if not isinstance(items, list):
                raise _Unsupported
            for index, item in enumerate(items):
                inner = {**scope, name: item, "__loop": (index == 0, index == len(items) - 1)}
                _render(body, inner, out, state)
        elif kind == "if":
            _, condition, then, otherwise = node
            negated = condition.startswith("not ")
            subject = condition[4:].strip() if negated else condition
            if subject in {"loop.first", "loop.last"}:
                if "__loop" not in scope:
                    raise _Unsupported
                first, last = scope["__loop"]
                value = first if subject == "loop.first" else last
            else:
                value = scope.get(subject, _UNDEFINED)
                if value is _UNDEFINED:
                    raise _Unsupported
            _render(then if bool(value) != negated else otherwise, scope, out, state)
        if kind in {"text", "expr"}:
            state.size += len(out[-1])
        if state.size > MAX_OUTPUT:
            raise _Unsupported


def render(text: str, macros: dict[str, Macro] | None = None) -> str | None:
    """Rendered text, or None when the template uses anything beyond the supported subset."""
    try:
        nodes, _, _ = _parse(_tokens(text), 0, set())
        out: list[str] = []
        _render(nodes, {}, out, _State(macros or {}))
    except _Unsupported:
        return None
    return "".join(out)
