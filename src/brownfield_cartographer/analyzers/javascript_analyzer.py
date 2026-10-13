"""JavaScript and TypeScript structure extraction with tree-sitter.

Extracts static module specifiers (ESM imports and re-exports, type-only
imports, and `require()` / `import()` calls with literal arguments), function
and method definitions, and top-level classes. Only relative specifiers are
resolved to repository files; package names and tsconfig path aliases are not.
"""

from __future__ import annotations

import json
import posixpath
import re
from dataclasses import dataclass
from functools import cache

import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser

from brownfield_cartographer.models.schemas import FunctionNode, ModuleNode

JS_EXTENSIONS = (".js", ".mjs", ".cjs", ".jsx")
TS_EXTENSIONS = (".ts", ".mts", ".cts", ".tsx")
EXTENSIONS = JS_EXTENSIONS + TS_EXTENSIONS
# Compiled-output specifiers that TypeScript ESM code uses for its own sources.
_SOURCE_FOR_OUTPUT = {".js": (".ts", ".tsx"), ".mjs": (".mts",), ".cjs": (".cts",), ".jsx": (".tsx",)}

_BRANCHES = {
    "if_statement", "for_statement", "for_in_statement", "while_statement", "do_statement",
    "ternary_expression", "catch_clause", "switch_case",
}
_FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
_FUNCTION_DECLARATIONS = {"function_declaration", "generator_function_declaration"}
_CLASS_DECLARATIONS = {"class_declaration", "abstract_class_declaration", "class"}


@cache
def _language(suffix: str) -> Language:
    if suffix == ".tsx":
        return Language(tstypescript.language_tsx())
    if suffix in TS_EXTENSIONS:
        return Language(tstypescript.language_typescript())
    return Language(tsjavascript.language())


def parse(source: bytes, suffix: str) -> Node:
    """Parse source and raise ValueError at the first syntax error."""
    root = Parser(_language(suffix)).parse(source).root_node
    if root.has_error:
        line = _first_error_line(root)
        raise ValueError(f"Invalid JavaScript/TypeScript syntax at line {line}")
    return root


def _first_error_line(node: Node) -> int:
    for child in node.children:
        if child.is_error or child.is_missing:
            return child.start_point[0] + 1
        if child.has_error:
            return _first_error_line(child)
    return node.start_point[0] + 1


def _text(node: Node | None) -> str:
    return node.text.decode("utf-8") if node is not None and node.text is not None else ""


def _string_value(node: Node | None) -> str | None:
    """Return a plain string literal's value; template or computed values return None."""
    if node is None or node.type != "string":
        return None
    return "".join(_text(child) for child in node.children if child.type == "string_fragment")


def _walk(node: Node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _specifiers(root: Node) -> list[tuple[str, tuple[int, int]]]:
    found = []
    for node in _walk(root):
        source = None
        if node.type in {"import_statement", "export_statement", "import_require_clause"}:
            source = _string_value(node.child_by_field_name("source"))
        elif node.type == "call_expression":
            function = node.child_by_field_name("function")
            arguments = node.child_by_field_name("arguments")
            if function is not None and (function.type == "import" or _text(function) == "require"):
                values = [child for child in arguments.named_children] if arguments is not None else []
                if len(values) == 1:
                    source = _string_value(values[0])
        if source:
            found.append((source, (node.start_point[0] + 1, node.end_point[0] + 1)))
    return found


def _functions(root: Node, module_path: str, source: bytes) -> list[FunctionNode]:
    functions: list[FunctionNode] = []

    def add(name: str, node: Node) -> None:
        signature = source[node.start_byte:node.end_byte].decode("utf-8", "replace").split("\n")[0]
        functions.append(FunctionNode(
            qualified_name=f"{module_path}::{name}", parent_module=module_path,
            signature=signature.strip(), is_public_api=not name.startswith("_"),
            line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1))

    def visit_class(node: Node) -> None:
        name = _text(node.child_by_field_name("name"))
        body = node.child_by_field_name("body")
        if not name or body is None:
            return
        for member in body.named_children:
            if member.type == "method_definition":
                add(f"{name}.{_text(member.child_by_field_name('name'))}", member)

    def visit(node: Node) -> None:
        for child in node.named_children:
            if child.type == "export_statement":
                declaration = child.child_by_field_name("declaration")
                value = child.child_by_field_name("value")
                if declaration is not None:
                    visit_declaration(declaration, default_name=None)
                elif value is not None and value.type in _FUNCTION_VALUES | _CLASS_DECLARATIONS:
                    visit_declaration(value, default_name="default")
                else:
                    for grandchild in child.named_children:
                        visit_declaration(grandchild, default_name="default")
            else:
                visit_declaration(child, default_name=None)

    def visit_declaration(node: Node, default_name: str | None) -> None:
        if node.type in _FUNCTION_DECLARATIONS | _FUNCTION_VALUES:
            name = _text(node.child_by_field_name("name")) or default_name
            if name:
                add(name, node)
        elif node.type in _CLASS_DECLARATIONS:
            visit_class(node)
        elif node.type in {"lexical_declaration", "variable_declaration"}:
            for declarator in node.named_children:
                value = declarator.child_by_field_name("value")
                if declarator.type == "variable_declarator" and value is not None and value.type in _FUNCTION_VALUES:
                    add(_text(declarator.child_by_field_name("name")), declarator)

    visit(root)
    return functions


def _classes(root: Node) -> list[str]:
    names = []
    for child in root.named_children:
        node = child.child_by_field_name("declaration") if child.type == "export_statement" else child
        if node is not None and node.type in _CLASS_DECLARATIONS:
            name = _text(node.child_by_field_name("name"))
            if name:
                names.append(name)
    return names


def analyze(module: ModuleNode, source: bytes, suffix: str) -> list[FunctionNode]:
    """Populate imports, classes and complexity on module; return its functions."""
    root = parse(source, suffix)
    for specifier, lines in _specifiers(root):
        module.import_locations.setdefault(specifier, []).append(lines)
    module.imports = sorted(module.import_locations)
    module.classes = _classes(root)
    functions = _functions(root, module.path, source)
    module.public_functions = [fn.qualified_name.split("::")[-1] for fn in functions if fn.is_public_api]
    branches = sum(1 for node in _walk(root) if node.type in _BRANCHES or (
        node.type == "binary_expression" and _text(node.child_by_field_name("operator")) in {"&&", "||", "??"}))
    module.complexity_score = round((1 + branches) / max(module.lines_of_code, 1) * 100.0, 2)
    return functions


def _file_for(base: str, known_paths: set[str]) -> str | None:
    base = posixpath.normpath(base)
    if base.startswith("../") or base == "..":
        return None
    stem, extension = posixpath.splitext(base)
    candidates = [base]
    candidates += [stem + source for source in _SOURCE_FOR_OUTPUT.get(extension, ())]
    candidates += [base + ext for ext in TS_EXTENSIONS + JS_EXTENSIONS]
    candidates += [f"{base}/index{ext}" for ext in TS_EXTENSIONS + JS_EXTENSIONS]
    return next((candidate for candidate in candidates if candidate in known_paths), None)


@dataclass(frozen=True)
class AliasScope:
    """Path aliases from the tsconfig/jsconfig in `directory` (repository-relative, '' for root)."""
    directory: str
    paths: tuple[tuple[str, tuple[str, ...]], ...] = ()
    base_url: str | None = None


def _jsonc(text: str) -> dict:
    """Parse JSON with comments and trailing commas, as tsconfig files allow."""
    out, index, in_string = [], 0, False
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if char == "\\" and index + 1 < len(text):
                out.append(text[index + 1])
                index += 1
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
            out.append(char)
        elif text.startswith("//", index):
            index = text.find("\n", index)
            index = len(text) if index == -1 else index
            continue
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = len(text) if end == -1 else end + 2
            continue
        else:
            out.append(char)
        index += 1
    value = json.loads(re.sub(r",(\s*[}\]])", r"\1", "".join(out)))
    return value if isinstance(value, dict) else {}


def load_alias_scopes(configs: dict[str, str]) -> list[AliasScope]:
    """Build alias scopes from {relative config path: text}; `extends` is followed inside the repository."""
    def options(path: str, seen: frozenset[str]) -> tuple[dict, dict[str, str]]:
        """Merged compilerOptions and, for paths/baseUrl, the directory of the config that set them."""
        if path in seen or path not in configs:
            return {}, {}
        try:
            data = _jsonc(configs[path])
        except ValueError:
            return {}, {}
        merged: dict = {}
        origin: dict[str, str] = {}
        parents = data.get("extends")
        for parent in parents if isinstance(parents, list) else [parents]:
            if isinstance(parent, str) and parent.startswith("."):
                target = posixpath.normpath(posixpath.join(posixpath.dirname(path), parent))
                target = target if target.endswith(".json") else target + ".json"
                inherited, inherited_origin = options(target, seen | {path})
                merged.update(inherited)
                origin.update(inherited_origin)
        own = data.get("compilerOptions")
        if isinstance(own, dict):
            for key in ("paths", "baseUrl"):
                if key in own:
                    merged[key] = own[key]
                    origin[key] = posixpath.dirname(path)
        return merged, origin

    scopes = []
    for path in sorted(configs):
        name = posixpath.basename(path).lower()
        if name not in {"tsconfig.json", "jsconfig.json"}:
            continue
        merged, origin = options(path, frozenset())
        base_url = None
        if isinstance(merged.get("baseUrl"), str):
            base_url = posixpath.normpath(posixpath.join(origin["baseUrl"], merged["baseUrl"]))
        paths = []
        if isinstance(merged.get("paths"), dict):
            root = base_url if base_url is not None else origin["paths"]
            for pattern, targets in merged["paths"].items():
                if isinstance(pattern, str) and isinstance(targets, list):
                    paths.append((pattern, tuple(posixpath.normpath(posixpath.join(root, t))
                                                 for t in targets if isinstance(t, str))))
        if paths or base_url is not None:
            directory = posixpath.dirname(path)
            scopes.append(AliasScope("" if directory == "." else directory, tuple(paths), base_url))
    # Nearest configuration first.
    return sorted(scopes, key=lambda scope: len(scope.directory), reverse=True)


def _alias_targets(specifier: str, scope: AliasScope) -> list[str]:
    exact = [targets for pattern, targets in scope.paths if "*" not in pattern and pattern == specifier]
    if exact:
        return list(exact[0])
    matches = []
    for pattern, targets in scope.paths:
        if pattern.count("*") == 1:
            prefix, suffix = pattern.split("*")
            if specifier.startswith(prefix) and specifier.endswith(suffix) and len(specifier) >= len(prefix) + len(suffix):
                captured = specifier[len(prefix):len(specifier) - len(suffix)]
                matches.append((len(prefix), [t.replace("*", captured) for t in targets]))
    # TypeScript prefers the pattern with the longest prefix.
    return max(matches, key=lambda m: m[0])[1] if matches else []


def resolve(specifier: str, importer: str, known_paths: set[str],
            aliases: list[AliasScope] | None = None) -> str | None:
    """Resolve relative specifiers and tsconfig/jsconfig aliases to repository files; packages return None."""
    if specifier.startswith(("./", "../")) or specifier in {".", ".."}:
        return _file_for(posixpath.join(posixpath.dirname(importer), specifier), known_paths)
    for scope in aliases or []:
        if scope.directory and not importer.startswith(scope.directory + "/"):
            continue
        for target in _alias_targets(specifier, scope):
            found = _file_for(target, known_paths)
            if found:
                return found
        if scope.base_url is not None:
            return _file_for(posixpath.join(scope.base_url, specifier), known_paths)
        return None
    return None
