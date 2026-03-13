"""
Hydrologist Agent — Data Flow & Lineage Analyst.

Specialized for data engineering codebases. Constructs the data lineage DAG
by analyzing data sources, transformations, and sinks across Python, SQL,
and YAML configuration files.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx

from src.analyzers.dag_config_parser import parse_airflow_dag, parse_dbt_project
from src.analyzers.sql_lineage import analyze_sql_file, analyze_sql_string
from src.graph.knowledge_graph import KnowledgeGraph
from src.models.schemas import (
    DatasetNode,
    StorageType,
    TransformationNode,
    TransformationType,
)

logger = logging.getLogger(__name__)

# Regex patterns for Python data I/O operations
PANDAS_READ_PATTERNS = [
    re.compile(r"""pd\.read_csv\(\s*['"]([^'"]+)['"]\s*"""),
    re.compile(r"""pd\.read_sql\(\s*['"]([^'"]+)['"]\s*"""),
    re.compile(r"""pd\.read_parquet\(\s*['"]([^'"]+)['"]\s*"""),
    re.compile(r"""pd\.read_excel\(\s*['"]([^'"]+)['"]\s*"""),
    re.compile(r"""pd\.read_json\(\s*['"]([^'"]+)['"]\s*"""),
    re.compile(r"""pandas\.read_csv\(\s*['"]([^'"]+)['"]\s*"""),
]

PANDAS_WRITE_PATTERNS = [
    re.compile(r"""\.to_csv\(\s*['"]([^'"]+)['"]\s*"""),
    re.compile(r"""\.to_parquet\(\s*['"]([^'"]+)['"]\s*"""),
    re.compile(r"""\.to_sql\(\s*['"]([^'"]+)['"]\s*"""),
    re.compile(r"""\.to_json\(\s*['"]([^'"]+)['"]\s*"""),
    re.compile(r"""\.to_excel\(\s*['"]([^'"]+)['"]\s*"""),
]

SPARK_READ_PATTERNS = [
    re.compile(r"""spark\.read\.[a-z]+\(\s*['"]([^'"]+)['"]\s*"""),
    re.compile(r"""spark\.table\(\s*['"]([^'"]+)['"]\s*"""),
]

SPARK_WRITE_PATTERNS = [
    re.compile(r"""\.write\.[a-z]+\(\s*['"]([^'"]+)['"]\s*"""),
    re.compile(r"""\.saveAsTable\(\s*['"]([^'"]+)['"]\s*"""),
]

SQLALCHEMY_PATTERNS = [
    re.compile(
        r"""execute\(\s*['"]([^'"]*(?:SELECT|INSERT|UPDATE|DELETE)[^'"]*)['"]\s*""",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Python Data Flow Analyzer
# ---------------------------------------------------------------------------


def _analyze_python_data_flow(
    file_path: Path, repo_root: Path
) -> list[TransformationNode]:
    """
    Scan a Python file for data read/write operations.

    Uses regex patterns to detect pandas, PySpark, and SQLAlchemy
    data access calls. Returns TransformationNodes for each detected
    data flow operation.
    """
    relative_path = str(file_path.relative_to(repo_root)).replace("\\", "/")

    try:
        source_text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read %s: %s", relative_path, exc)
        return []

    transformations: list[TransformationNode] = []
    reads: list[str] = []
    writes: list[str] = []

    # Detect pandas read operations
    for pattern in PANDAS_READ_PATTERNS:
        reads.extend(pattern.findall(source_text))

    # Detect pandas write operations
    for pattern in PANDAS_WRITE_PATTERNS:
        writes.extend(pattern.findall(source_text))

    # Detect Spark read/write operations
    for pattern in SPARK_READ_PATTERNS:
        reads.extend(pattern.findall(source_text))
    for pattern in SPARK_WRITE_PATTERNS:
        writes.extend(pattern.findall(source_text))

    # Detect inline SQL executed via SQLAlchemy style calls
    inline_sql_transforms: list[TransformationNode] = []
    for pattern in SQLALCHEMY_PATTERNS:
        for sql_text in pattern.findall(source_text):
            inline_sql_transforms.extend(
                analyze_sql_string(
                    sql_text,
                    source_file=relative_path,
                )
            )

    # If we found any data I/O, create a transformation node
    if reads or writes:
        transformations.append(
            TransformationNode(
                source_datasets=reads if reads else ["<dynamic_source>"],
                target_datasets=writes if writes else [f"<output_of:{relative_path}>"],
                transformation_type=TransformationType.PYTHON_TRANSFORM,
                source_file=relative_path,
                line_range=(1, len(source_text.split("\n"))),
            )
        )
    transformations.extend(inline_sql_transforms)

    return transformations


# ---------------------------------------------------------------------------
# Hydrologist Agent — Public API
# ---------------------------------------------------------------------------


class Hydrologist:
    """
    The Hydrologist agent traces data flow through the codebase.

    Combines analysis from:
      - Python data flow detection (pandas, spark, SQLAlchemy)
      - SQL lineage via sqlglot
      - dbt project / Airflow DAG config parsing

    Populates the knowledge graph with DatasetNodes, TransformationNodes,
    and the CONSUMES/PRODUCES edges that form the data lineage DAG.
    """

    def run(
        self,
        repo_root: Path,
        knowledge_graph: KnowledgeGraph,
        changed_files: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Execute full or incremental data lineage analysis on the given repository.

        Steps (full run):
          1. Detect if repo is a dbt project → parse dbt model DAG
          2. Scan all .sql files → extract SQL lineage via sqlglot
          3. Scan all .py files → detect pandas/spark/SQLAlchemy data flows
          4. Scan for Airflow DAG files → extract task dependencies
          5. Populate the knowledge graph with all discovered lineage
          6. Compute sources (in-degree=0) and sinks (out-degree=0)

        When changed_files is provided (incremental): remove lineage for those files,
        then re-analyze only those files (SQL, Python, DAG) and add new transformations.
        dbt project-level parsing is not re-run in incremental (only file-level SQL/Python).

        Args:
            repo_root: Absolute path to the repository root.
            knowledge_graph: Shared KnowledgeGraph to populate or update.
            changed_files: If set, only these paths are re-analyzed; existing
                transformations for these files are removed first.

        Returns:
            Summary dict with lineage statistics and key data assets.
        """
        if changed_files is not None:
            return self._run_incremental(repo_root, knowledge_graph, changed_files)

        logger.info("Hydrologist: analyzing data flows in %s", repo_root)
        all_transformations, all_datasets, sql_count, py_count = (
            self._collect_lineage_full(repo_root, knowledge_graph)
        )

        for dataset in all_datasets:
            knowledge_graph.add_dataset_node(dataset)
        for transformation in all_transformations:
            knowledge_graph.add_transformation_node(transformation)

        lineage_subgraph = knowledge_graph.get_lineage_subgraph()
        sources = self._find_sources(lineage_subgraph)
        sinks = self._find_sinks(lineage_subgraph)

        summary = {
            "total_transformations": len(all_transformations),
            "total_datasets": len(all_datasets),
            "sql_files_analyzed": sql_count,
            "python_files_scanned": py_count,
            "data_sources": sorted(sources),
            "data_sinks": sorted(sinks),
            "lineage_nodes": lineage_subgraph.number_of_nodes(),
            "lineage_edges": lineage_subgraph.number_of_edges(),
        }
        logger.info(
            "Hydrologist complete: %d transformations, %d sources, %d sinks",
            len(all_transformations),
            len(sources),
            len(sinks),
        )
        return summary

    def _collect_lineage_full(
        self, repo_root: Path, knowledge_graph: KnowledgeGraph
    ) -> tuple[list[TransformationNode], list[DatasetNode], int, int]:
        """Full repo lineage collection (dbt + all SQL + all Python + Airflow)."""
        all_transformations: list[TransformationNode] = []
        all_datasets: list[DatasetNode] = []

        dbt_project_file = repo_root / "dbt_project.yml"
        if dbt_project_file.exists():
            logger.info("Detected dbt project — parsing model DAG")
            dbt_results = parse_dbt_project(repo_root)
            all_transformations.extend(dbt_results["transformations"])
            all_datasets.extend(dbt_results["datasets"])
            for edge in dbt_results["config_edges"]:
                knowledge_graph.add_configures_edge(edge.source, edge.target)

        sql_files = [
            f
            for f in repo_root.rglob("*.sql")
            if not any(
                skip in f.parts
                for skip in {".git", "node_modules", "__pycache__", ".venv"}
            )
        ]
        for sql_file in sql_files:
            all_transformations.extend(analyze_sql_file(sql_file, repo_root))
        logger.info("Analyzed %d SQL files", len(sql_files))

        py_files = [
            f
            for f in repo_root.rglob("*.py")
            if not any(
                skip in f.parts
                for skip in {".git", "node_modules", "__pycache__", ".venv"}
            )
        ]
        for py_file in py_files:
            all_transformations.extend(_analyze_python_data_flow(py_file, repo_root))

        dags_dir = repo_root / "dags"
        if dags_dir.exists():
            for dag_file in dags_dir.rglob("*.py"):
                all_transformations.extend(parse_airflow_dag(dag_file, repo_root))

        return all_transformations, all_datasets, len(sql_files), len(py_files)

    def _run_incremental(
        self,
        repo_root: Path,
        knowledge_graph: KnowledgeGraph,
        changed_files: set[str],
    ) -> dict[str, Any]:
        """Re-analyze lineage only for changed files and merge into the graph."""
        logger.info(
            "Hydrologist: incremental update for %d changed files", len(changed_files)
        )
        knowledge_graph.remove_transformation_nodes_for_files(changed_files)

        all_transformations: list[TransformationNode] = []
        all_datasets: list[DatasetNode] = []
        sql_count = 0
        py_count = 0

        for rel_path in changed_files:
            if not rel_path.strip():
                continue
            path = (repo_root / rel_path).resolve()
            if not path.exists() or not path.is_file():
                continue
            if rel_path.endswith(".sql"):
                all_transformations.extend(analyze_sql_file(path, repo_root))
                sql_count += 1
            elif rel_path.endswith(".py"):
                if "dags" in path.parts:
                    all_transformations.extend(parse_airflow_dag(path, repo_root))
                else:
                    all_transformations.extend(
                        _analyze_python_data_flow(path, repo_root)
                    )
                py_count += 1

        for dataset in all_datasets:
            knowledge_graph.add_dataset_node(dataset)
        for transformation in all_transformations:
            knowledge_graph.add_transformation_node(transformation)

        lineage_subgraph = knowledge_graph.get_lineage_subgraph()
        sources = self._find_sources(lineage_subgraph)
        sinks = self._find_sinks(lineage_subgraph)

        summary = {
            "total_transformations": len(all_transformations),
            "total_datasets": len(all_datasets),
            "sql_files_analyzed": sql_count,
            "python_files_scanned": py_count,
            "data_sources": sorted(sources),
            "data_sinks": sorted(sinks),
            "lineage_nodes": lineage_subgraph.number_of_nodes(),
            "lineage_edges": lineage_subgraph.number_of_edges(),
        }
        logger.info(
            "Hydrologist incremental complete: %d new transformations, %d sources, %d sinks",
            len(all_transformations),
            len(sources),
            len(sinks),
        )
        return summary

    # ------------------------------------------------------------------
    # Lineage Query Methods
    # ------------------------------------------------------------------

    @staticmethod
    def blast_radius(knowledge_graph: KnowledgeGraph, node_name: str) -> list[str]:
        """
        Find all downstream dependents of a given node.

        Uses BFS traversal to discover every node that would be
        affected if the given node's interface or data changed.

        Args:
            knowledge_graph: The populated knowledge graph.
            node_name: The node whose blast radius to compute.

        Returns:
            List of all downstream dependent node names.
        """
        if node_name not in knowledge_graph.graph:
            logger.warning("Node '%s' not found in graph", node_name)
            return []

        visited: set[str] = set()
        queue: deque[str] = deque([node_name])

        while queue:
            current = queue.popleft()
            for successor in knowledge_graph.graph.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)

        return sorted(visited)

    @staticmethod
    def find_sources(knowledge_graph: KnowledgeGraph) -> list[str]:
        """
        Find all data source nodes (in-degree = 0 in the lineage graph).

        These are the entry points of the data system — where data
        enters the pipeline from external sources.
        """
        lineage = knowledge_graph.get_lineage_subgraph()
        return sorted(Hydrologist._find_sources(lineage))

    @staticmethod
    def find_sinks(knowledge_graph: KnowledgeGraph) -> list[str]:
        """
        Find all data sink nodes (out-degree = 0 in the lineage graph).

        These are the terminal outputs — final tables, exports, or APIs.
        """
        lineage = knowledge_graph.get_lineage_subgraph()
        return sorted(Hydrologist._find_sinks(lineage))

    @staticmethod
    def _find_sources(graph: nx.DiGraph) -> set[str]:
        """Nodes with in-degree 0 (data enters here)."""
        return {
            node
            for node in graph.nodes()
            if graph.in_degree(node) == 0
            and graph.nodes[node].get("node_type") == "dataset"
        }

    @staticmethod
    def _find_sinks(graph: nx.DiGraph) -> set[str]:
        """Nodes with out-degree 0 (data leaves here)."""
        return {
            node
            for node in graph.nodes()
            if graph.out_degree(node) == 0
            and graph.nodes[node].get("node_type") == "dataset"
        }

    @staticmethod
    def trace_upstream(knowledge_graph: KnowledgeGraph, dataset_name: str) -> list[str]:
        """
        Trace all upstream sources that feed into a given dataset.

        Follows edges backwards to find every source that contributes
        data to the specified output dataset.
        """
        if dataset_name not in knowledge_graph.graph:
            logger.warning("Dataset '%s' not found in graph", dataset_name)
            return []

        visited: set[str] = set()
        queue: deque[str] = deque([dataset_name])

        while queue:
            current = queue.popleft()
            for predecessor in knowledge_graph.graph.predecessors(current):
                if predecessor not in visited:
                    visited.add(predecessor)
                    queue.append(predecessor)

        # Filter to only dataset nodes (exclude transformation nodes)
        return sorted(
            node
            for node in visited
            if knowledge_graph.graph.nodes[node].get("node_type") == "dataset"
        )
