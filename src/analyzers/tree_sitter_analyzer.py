"""
Tree-sitter based multi-language AST analyzer.

Provides structural code analysis for Python, SQL, and YAML files using
tree-sitter grammars. The LanguageRouter selects the correct parser based
on file extension, enabling unified analysis across polyglot codebases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

from src.models.schemas import (
    FunctionNode,
    Language as LangEnum,
    ModuleNode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language Router
# ---------------------------------------------------------------------------

# Extension → Language enum mapping
EXTENSION_MAP: dict[str, LangEnum] = {
    ".py": LangEnum.PYTHON,
    ".sql": LangEnum.SQL,
    ".yml": LangEnum.YAML,
    ".yaml": LangEnum.YAML,
    ".js": LangEnum.JAVASCRIPT,
    ".ts": LangEnum.TYPESCRIPT,
}


def _get_python_language() -> Language:
    """Load the tree-sitter Python grammar."""
    return Language(tspython.language())


def detect_language(file_path: Path) -> LangEnum:
    """Route a file to the correct language based on its extension."""
    return EXTENSION_MAP.get(file_path.suffix.lower(), LangEnum.UNKNOWN)


def get_parser(lang: LangEnum) -> Optional[Parser]:
    """
    Create a tree-sitter parser for the given language.

    Currently supports Python with full AST analysis.
    SQL and YAML are handled by specialized analyzers (sqlglot, PyYAML).
    """
    if lang == LangEnum.PYTHON:
        parser = Parser(language=_get_python_language())
        return parser
    # SQL and YAML are handled by dedicated analyzers, not tree-sitter
    return None


# ---------------------------------------------------------------------------
# Python AST Analysis
# ---------------------------------------------------------------------------

def _extract_imports(root_node: Node, source_code: bytes) -> list[str]:
    """
    Walk the AST to find all import statements.

    Handles both `import X` and `from X import Y` styles.
    Returns a list of imported module paths.
    """
    imports: list[str] = []

    for child in root_node.children:
        if child.type == "import_statement":
            # import foo, bar, baz
            for name_node in child.children:
                if name_node.type == "dotted_name":
                    imports.append(name_node.text.decode("utf-8"))
                elif name_node.type == "aliased_import":
                    dotted = name_node.child_by_field_name("name")
                    if dotted:
                        imports.append(dotted.text.decode("utf-8"))

        elif child.type == "import_from_statement":
            # from foo.bar import baz
            module_node = child.child_by_field_name("module_name")
            if module_node:
                imports.append(module_node.text.decode("utf-8"))

    return imports


def _extract_functions(
    root_node: Node, source_code: bytes, module_path: str
) -> list[FunctionNode]:
    """
    Extract all top-level and class-level function definitions.

    Returns FunctionNode instances with signature, line range,
    and public/private classification.
    """
    functions: list[FunctionNode] = []

    def _walk(node: Node, class_prefix: str = "") -> None:
        for child in node.children:
            if child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                name = name_node.text.decode("utf-8")
                qualified = f"{class_prefix}{name}" if class_prefix else name
                # Extract the signature line
                first_line = source_code[
                    child.start_byte : child.end_byte
                ].decode("utf-8").split("\n")[0]

                functions.append(
                    FunctionNode(
                        qualified_name=f"{module_path}::{qualified}",
                        parent_module=module_path,
                        signature=first_line.strip(),
                        is_public_api=not name.startswith("_"),
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                    )
                )

            elif child.type == "class_definition":
                cls_name_node = child.child_by_field_name("name")
                if cls_name_node:
                    cls_name = cls_name_node.text.decode("utf-8")
                    prefix = f"{class_prefix}{cls_name}." if class_prefix else f"{cls_name}."
                    # Recurse into class body
                    body = child.child_by_field_name("body")
                    if body:
                        _walk(body, class_prefix=prefix)

    _walk(root_node)
    return functions


def _extract_classes(root_node: Node, source_code: bytes) -> list[str]:
    """Extract names of all top-level class definitions."""
    classes: list[str] = []
    for child in root_node.children:
        if child.type == "class_definition":
            name_node = child.child_by_field_name("name")
            if name_node:
                classes.append(name_node.text.decode("utf-8"))
    return classes


def _compute_comment_ratio(source_code: str) -> float:
    """Calculate the fraction of lines that are comments or docstrings."""
    lines = source_code.split("\n")
    if not lines:
        return 0.0
    comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
    return round(comment_lines / len(lines), 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_module(file_path: Path, repo_root: Path) -> Optional[ModuleNode]:
    """
    Perform full structural analysis on a single source file.

    Uses tree-sitter for Python files to extract imports, functions,
    classes, and complexity metrics. Non-Python files get basic metadata.

    Args:
        file_path: Absolute path to the source file.
        repo_root: Root directory of the repository (for relative paths).

    Returns:
        ModuleNode with extracted structural data, or None if unparseable.
    """
    relative_path = str(file_path.relative_to(repo_root)).replace("\\", "/")
    lang = detect_language(file_path)

    try:
        source_bytes = file_path.read_bytes()
        source_text = source_bytes.decode("utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Skipping unreadable file %s: %s", relative_path, exc)
        return None

    lines_of_code = len(source_text.split("\n"))

    # Default node with basic metadata
    module = ModuleNode(
        path=relative_path,
        language=lang,
        lines_of_code=lines_of_code,
        comment_ratio=_compute_comment_ratio(source_text),
    )

    # Deep analysis for Python files via tree-sitter
    if lang == LangEnum.PYTHON:
        parser = get_parser(lang)
        if parser:
            tree = parser.parse(source_bytes)
            root = tree.root_node

            module.imports = _extract_imports(root, source_bytes)
            module.classes = _extract_classes(root, source_bytes)
            module.public_functions = [
                fn.qualified_name.split("::")[-1]
                for fn in _extract_functions(root, source_bytes, relative_path)
                if fn.is_public_api
            ]

    return module


def extract_functions_from_file(
    file_path: Path, repo_root: Path
) -> list[FunctionNode]:
    """
    Extract all function definitions from a Python file.

    Separate from analyze_module to allow fine-grained call-graph analysis.
    """
    relative_path = str(file_path.relative_to(repo_root)).replace("\\", "/")
    lang = detect_language(file_path)

    if lang != LangEnum.PYTHON:
        return []

    try:
        source_bytes = file_path.read_bytes()
    except OSError as exc:
        logger.warning("Cannot read %s: %s", relative_path, exc)
        return []

    parser = get_parser(lang)
    if not parser:
        return []

    tree = parser.parse(source_bytes)
    return _extract_functions(tree.root_node, source_bytes, relative_path)
