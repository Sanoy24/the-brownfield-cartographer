"""Static Python call extraction: calls inside functions to names bound by imports or definitions.

Only references that resolve through the module's own bindings are reported:
same-module functions and classes, `from x import f` names, module attributes
reached through `import x` / `import x as m` / `from pkg import mod`, and
`self.method()` / `cls.method()` inside a class. Calls on other objects,
dynamic calls and names rebound inside the calling function are skipped
rather than guessed.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

# A binding is ("module", dotted) for `import`, ("from", module, name) for
# `from module import name`, or ("local", qualified) for a definition here.
Binding = tuple[str, ...]
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class _Scope:
    shadowed: set[str] = field(default_factory=set)
    nested: dict[str, str] = field(default_factory=dict)  # nested def name -> qualified name
    class_prefix: str = ""  # "Class." for a method
    receiver: str | None = None  # `self` or `cls`


def _join(module: str, name: str) -> str:
    return module + name if module.endswith(".") else f"{module}.{name}"


def _module_statements(body: list[ast.stmt]):
    """Module-level statements, including those under `if`, `try` and `with` (optional imports)."""
    for node in body:
        yield node
        if isinstance(node, (ast.If, ast.With, ast.AsyncWith)):
            yield from _module_statements(node.body + getattr(node, "orelse", []))
        elif isinstance(node, (ast.Try, ast.TryStar)):
            blocks = [*node.body, *node.orelse, *node.finalbody, *(s for h in node.handlers for s in h.body)]
            yield from _module_statements(blocks)


def _module_bindings(tree: ast.Module) -> dict[str, Binding]:
    bindings: dict[str, Binding] = {}
    for node in _module_statements(tree.body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bindings[alias.asname] = ("module", alias.name)
                else:
                    head = alias.name.split(".")[0]
                    bindings[head] = ("module", head)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = ("from", module, alias.name)
        elif isinstance(node, (*FUNCTIONS, ast.ClassDef)):
            bindings[node.name] = ("local", node.name)
    return bindings


def _bound_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Names the function binds itself (parameters, assignments, imports, loop targets)."""
    args = function.args
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    names.update(a.arg for a in (args.vararg, args.kwarg) if a)
    for node in ast.walk(function):
        if node is function:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.difference_update(node.names)
    return names


def _chain(node: ast.expr) -> list[str] | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return [node.id, *reversed(parts)]


def _reference(chain: list[str], bindings: dict[str, Binding], scope: _Scope) -> tuple[str, str] | None:
    """(module, name) for a call chain; module "" means a definition in this file."""
    head, attrs = chain[0], chain[1:]
    if head in scope.nested and not attrs:
        return "", scope.nested[head]
    if scope.receiver and head == scope.receiver and len(attrs) == 1:
        return "", scope.class_prefix + attrs[0]
    if head in scope.shadowed or head not in bindings:
        return None
    binding = bindings[head]
    if binding[0] == "local":
        return ("", binding[1] if not attrs else ".".join([binding[1], *attrs])) if len(attrs) <= 1 else None
    if binding[0] == "from":
        module, name = binding[1], binding[2]
        if not attrs:
            return module, name
        return ".".join([_join(module, name), *attrs[:-1]]), attrs[-1]
    if not attrs:
        return None
    return ".".join([binding[1], *attrs[:-1]]), attrs[-1]


def extract_calls(source: str) -> list[tuple[str, str, str, int]]:
    """(caller qualified name, module reference, callee name, line), first call site per pair.

    Caller names follow the function index: `Class.method` and `outer.inner`.
    """
    tree = ast.parse(source)
    bindings = _module_bindings(tree)
    calls: dict[tuple[str, str, str], int] = {}

    def visit(body: list[ast.stmt], prefix: str, function: str | None, scope: _Scope | None) -> None:
        for statement in body:
            if isinstance(statement, FUNCTIONS):
                qualified = prefix + statement.name
                class_prefix = prefix if prefix and function is None else ""
                params = [*statement.args.posonlyargs, *statement.args.args]
                nested = {s.name: f"{qualified}.{s.name}" for s in statement.body if isinstance(s, FUNCTIONS)}
                static = any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in statement.decorator_list)
                # A method's first parameter is the instance or class unless it is a staticmethod.
                receiver = (params[0].arg if class_prefix and params and not static
                            and params[0].arg in {"self", "cls"} else None)
                shadowed = (_bound_names(statement) | (scope.shadowed if scope else set())) - set(nested)
                visit(statement.body, f"{qualified}.", qualified, _Scope(shadowed, nested, class_prefix, receiver))
            elif isinstance(statement, ast.ClassDef):
                visit(statement.body, f"{prefix}{statement.name}.", function, scope)
            elif function is not None and scope is not None:
                for node in ast.walk(statement):
                    if isinstance(node, ast.Call) and (chain := _chain(node.func)):
                        reference = _reference(chain, bindings, scope)
                        if reference:
                            calls.setdefault((function, *reference), node.lineno)

    visit(tree.body, "", None, None)
    return sorted(((caller, module, name, line) for (caller, module, name), line in calls.items()),
                  key=lambda row: (row[3], row))
