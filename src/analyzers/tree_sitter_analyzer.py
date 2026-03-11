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
from tree_sitter import Language, Parser, Node, QueryCursor

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
# Python AST Analysis (using modern Query API)
# ---------------------------------------------------------------------------

# Optimized queries for Python structural extraction
PYTHON_QUERIES = {
    "imports": """
        (import_statement (dotted_name) @name)
        (import_from_statement module_name: (dotted_name) @module)
    """,
    "functions": """
        (function_definition name: (identifier) @name)
    """,
    "classes": """
        (class_definition name: (identifier) @name)
    """,
    "complexity": """
        (if_statement) @branch
        (while_statement) @branch
        (for_statement) @branch
        (conditional_expression) @branch
        (boolean_operator) @branch
        (except_clause) @branch
        (list_comprehension) @branch
        (set_comprehension) @branch
        (dictionary_comprehension) @branch
        (generator_expression) @branch
    """,
}


def _extract_imports(root_node: Node, source_code: bytes) -> list[str]:
    """
    Use Tree-sitter Query API to find all import statements.

    cursor.captures() returns {tag_name: [Node, ...]}, so we
    iterate over the dict values to get the matched nodes.
    """
    lang = _get_python_language()
    query = lang.query(PYTHON_QUERIES["imports"])
    cursor = QueryCursor(query)

    imports: list[str] = []
    # captures is a dict: {"name": [Node, ...], "module": [Node, ...]}
    captures: dict = cursor.captures(root_node)
    for nodes in captures.values():
        for node in nodes:
            imports.append(node.text.decode("utf-8"))

    return list(set(imports))  # Deduplicate


def _extract_functions(
    root_node: Node, source_code: bytes, module_path: str
) -> list[FunctionNode]:
    """
    Use Tree-sitter Query API to extract function definitions.
    
    Note: Still uses a recursive walk for class-level functions to 
    maintain qualified naming (Class.method) correctly.
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
    """
    Use Tree-sitter Query API to extract top-level class names.

    Only captures class names whose parent class_definition is a
    direct child of the module root node (i.e. top-level classes).
    """
    lang = _get_python_language()
    query = lang.query(PYTHON_QUERIES["classes"])
    cursor = QueryCursor(query)

    classes: list[str] = []
    captures: dict = cursor.captures(root_node)
    for nodes in captures.values():
        for node in nodes:
            # node is the identifier; its parent is class_definition
            cls_def = node.parent
            if cls_def and cls_def.parent == root_node:
                classes.append(node.text.decode("utf-8"))

    return classes


def _compute_complexity(root_node: Node, lines_of_code: int) -> float:
    """
    Estimate cyclomatic complexity by counting decision points and
    normalizing by file size.

    Raw complexity is approximated as the number of branching
    constructs (if, for, while, boolean operators, comprehensions,
    except clauses). This value is then scaled to a
    "complexity per 100 LOC" score so that large files are
    comparable to small ones.
    """
    lang = _get_python_language()
    query = lang.query(PYTHON_QUERIES["complexity"])
    cursor = QueryCursor(query)
    captures: dict = cursor.captures(root_node)

    # Sum total matched nodes across all capture tags
    total = sum(len(nodes) for nodes in captures.values())
    logger.debug("Complexity captures: %d", total)

    loc = max(lines_of_code, 1)
    raw = 1 + total  # cyclomatic-style base of 1
    # Normalize to "complexity per 100 LOC" for comparability
    normalized = (raw / loc) * 100.0
    return round(normalized, 2)


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
            module.complexity_score = _compute_complexity(root, lines_of_code)

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
