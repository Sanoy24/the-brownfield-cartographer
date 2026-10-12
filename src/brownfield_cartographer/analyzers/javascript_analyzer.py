"""JavaScript and TypeScript structure extraction with tree-sitter.

Extracts static module specifiers (ESM imports and re-exports, type-only
imports, and `require()` / `import()` calls with literal arguments), function
and method definitions, and top-level classes. Only relative specifiers are
resolved to repository files; package names and tsconfig path aliases are not.
"""

from __future__ import annotations

import posixpath
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


def resolve(specifier: str, importer: str, known_paths: set[str]) -> str | None:
    """Resolve a relative specifier to a repository file; other specifiers return None."""
    if not specifier.startswith(("./", "../")) and specifier not in {".", ".."}:
        return None
    base = posixpath.normpath(posixpath.join(posixpath.dirname(importer), specifier))
    if base.startswith("../"):
        return None
    stem, extension = posixpath.splitext(base)
    candidates = [base]
    candidates += [stem + source for source in _SOURCE_FOR_OUTPUT.get(extension, ())]
    candidates += [base + ext for ext in TS_EXTENSIONS + JS_EXTENSIONS]
    candidates += [f"{base}/index{ext}" for ext in TS_EXTENSIONS + JS_EXTENSIONS]
    return next((candidate for candidate in candidates if candidate in known_paths), None)
