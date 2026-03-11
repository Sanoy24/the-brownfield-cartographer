"""
Surveyor Agent — Static Structure Analyst.

Performs deep static analysis of the entire codebase using tree-sitter
for language-agnostic AST parsing. Builds the structural skeleton:
module import graph, PageRank hub detection, git velocity analysis,
and dead code candidate identification.
"""

from __future__ import annotations

import logging
import subprocess
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from src.analyzers.tree_sitter_analyzer import (
    analyze_module,
    detect_language,
    extract_functions_from_file,
)
from src.graph.knowledge_graph import KnowledgeGraph
from src.models.schemas import Language, ModuleNode

logger = logging.getLogger(__name__)

# File extensions the Surveyor scans
SCANNABLE_EXTENSIONS = {".py", ".sql", ".yml", ".yaml", ".js", ".ts"}

# Directories to always skip
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".cartography", ".eggs", "*.egg-info",
}


def _should_skip_dir(dir_name: str) -> bool:
    """Check if a directory should be excluded from scanning."""
    return dir_name in SKIP_DIRS or dir_name.endswith(".egg-info")


def _collect_source_files(repo_root: Path) -> list[Path]:
    """
    Walk the repository tree and collect all scannable source files.

    Skips hidden directories, virtual environments, and build artifacts.
    Returns absolute paths sorted alphabetically.
    """
    source_files: list[Path] = []

    for item in repo_root.rglob("*"):
        # Skip excluded directories
        if any(_should_skip_dir(part) for part in item.parts):
            continue
        if item.is_file() and item.suffix.lower() in SCANNABLE_EXTENSIONS:
            source_files.append(item)

    return sorted(source_files)


# ---------------------------------------------------------------------------
# Git Velocity Analysis
# ---------------------------------------------------------------------------

def extract_git_velocity(
    repo_root: Path, days: int = 30
) -> dict[str, int]:
    """
    Compute per-file commit frequency over the last N days.

    Uses `git log --name-only` to count how many commits touched
    each file. Files with the highest velocity are likely pain points
    or actively developed features.

    Args:
        repo_root: Path to the git repository root.
        days: Number of days to look back (default: 30).

    Returns:
        Dict mapping relative file paths to commit counts.
    """
    since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        result = subprocess.run(
            [
                "git", "log",
                f"--since={since_date}",
                "--name-only",
                "--pretty=format:",
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("Git velocity extraction failed: %s", exc)
        return {}

    if result.returncode != 0:
        logger.warning("git log failed: %s", result.stderr.strip())
        return {}

    # Count occurrences of each file path
    file_counts: Counter[str] = Counter()
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            file_counts[line.replace("\\", "/")] += 1

    return dict(file_counts)


# ---------------------------------------------------------------------------
# Dead Code Detection
# ---------------------------------------------------------------------------

def _detect_dead_code(
    modules: dict[str, ModuleNode],
) -> set[str]:
    """
    Identify modules whose exported symbols are never imported by anyone.

    A module is a dead code candidate if:
      - It defines public functions or classes
      - No other module in the repo imports from it
    """
    # Collect all imported module names across the codebase
    all_imported: set[str] = set()
    for module in modules.values():
        for imp in module.imports:
            all_imported.add(imp)
            # Also add partial matches for package imports
            parts = imp.split(".")
            for i in range(1, len(parts) + 1):
                all_imported.add(".".join(parts[:i]))

    dead_candidates: set[str] = set()
    for path, module in modules.items():
        if not module.public_functions and not module.classes:
            continue  # Nothing to import from here

        # Convert file path to module-style dotted name
        module_name = path.replace("/", ".").replace("\\", ".")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]

        # Check if anyone imports this module (by any prefix)
        is_imported = any(
            module_name.endswith(imp) or imp.endswith(module_name)
            for imp in all_imported
        )
        if not is_imported:
            dead_candidates.add(path)

    return dead_candidates


# ---------------------------------------------------------------------------
# Surveyor Agent — Public API
# ---------------------------------------------------------------------------

class Surveyor:
    """
    The Surveyor agent analyzes the static structure of a codebase.

    Orchestrates tree-sitter parsing, git velocity extraction, module
    graph construction, PageRank analysis, and dead code detection.

    Usage:
        surveyor = Surveyor()
        results = surveyor.run(Path("/path/to/repo"))
    """

    def run(
        self,
        repo_root: Path,
        knowledge_graph: KnowledgeGraph,
    ) -> dict[str, Any]:
        """
        Execute full structural analysis on the given repository.

        Steps:
          1. Scan all source files
          2. Parse each file with tree-sitter (extract imports, functions, classes)
          3. Compute git change velocity
          4. Build the module import graph (NetworkX DiGraph)
          5. Run PageRank to identify architectural hubs
          6. Detect circular dependencies (strongly connected components)
          7. Flag dead code candidates

        Args:
            repo_root: Absolute path to the repository root.
            knowledge_graph: Shared KnowledgeGraph to populate.

        Returns:
            Dict with analysis summary and key metrics.
        """
        logger.info("Surveyor: scanning %s", repo_root)

        # Step 1: Collect source files
        source_files = _collect_source_files(repo_root)
        logger.info("Found %d source files", len(source_files))

        # Step 2: Parse each file
        modules: dict[str, ModuleNode] = {}
        for file_path in source_files:
            module = analyze_module(file_path, repo_root)
            if module:
                modules[module.path] = module

        logger.info("Parsed %d modules successfully", len(modules))

        # Step 3: Git velocity
        velocity = extract_git_velocity(repo_root)
        for path, commits in velocity.items():
            if path in modules:
                modules[path].change_velocity_30d = commits

        # Step 4: Detect dead code candidates
        dead_code = _detect_dead_code(modules)
        for path in dead_code:
            if path in modules:
                modules[path].is_dead_code_candidate = True

        # Step 5: Register modules and build import graph
        for module in modules.values():
            knowledge_graph.add_module_node(module)

        # Build import edges between modules
        all_module_paths = set(modules.keys())
        for module in modules.values():
            for imp in module.imports:
                # Try to resolve import to a known module file
                target = _resolve_import(imp, all_module_paths)
                if target and target != module.path:
                    knowledge_graph.add_import_edge(module.path, target)

        # Step 6: Extract functions for richer analysis
        for file_path in source_files:
            functions = extract_functions_from_file(file_path, repo_root)
            for fn in functions:
                knowledge_graph.add_function_node(fn)

        # Step 7: Compute structural metrics
        module_subgraph = knowledge_graph.get_module_subgraph()
        pagerank = knowledge_graph.compute_pagerank(module_subgraph)
        cycles = knowledge_graph.find_circular_dependencies(module_subgraph)

        # Sort by PageRank to find architectural hubs
        top_hubs = sorted(
            pagerank.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # High-velocity files (top 20%)
        velocity_sorted = sorted(
            velocity.items(), key=lambda x: x[1], reverse=True
        )
        high_velocity_cutoff = max(1, len(velocity_sorted) // 5)
        high_velocity_files = velocity_sorted[:high_velocity_cutoff]

        summary = {
            "total_files_scanned": len(source_files),
            "modules_parsed": len(modules),
            "top_architectural_hubs": [
                {"path": path, "pagerank": round(score, 4)}
                for path, score in top_hubs
            ],
            "circular_dependencies": cycles,
            "dead_code_candidates": sorted(dead_code),
            "high_velocity_files": [
                {"path": path, "commits_30d": count}
                for path, count in high_velocity_files
            ],
        }

        logger.info("Surveyor complete: %s", json.dumps(summary, indent=2) if len(str(summary)) < 2000 else f"{len(modules)} modules analyzed")
        return summary


def _resolve_import(import_name: str, known_paths: set[str]) -> Optional[str]:
    """
    Attempt to resolve a Python import name to a known module file path.

    Handles dotted imports by converting them to path separators.
    """
    # Convert dotted name to potential file paths
    path_variants = [
        import_name.replace(".", "/") + ".py",
        import_name.replace(".", "/") + "/__init__.py",
    ]

    for variant in path_variants:
        if variant in known_paths:
            return variant
        # Try partial suffix matching for relative imports
        for known_path in known_paths:
            if known_path.endswith(variant):
                return known_path

    return None


# Need json for the logger call
import json
