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

from src.analyzers.dag_config_parser import parse_airflow_dag, parse_dbt_project
from src.analyzers.sql_lineage import analyze_sql_file
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
    re.compile(r"""execute\(\s*['"]([^'"]*(?:SELECT|INSERT|UPDATE|DELETE)[^'"]*)['"]\s*""", re.IGNORECASE),
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
    ) -> dict[str, Any]:
        """
        Execute full data lineage analysis on the given repository.

        Steps:
          1. Detect if repo is a dbt project → parse dbt model DAG
          2. Scan all .sql files → extract SQL lineage via sqlglot
          3. Scan all .py files → detect pandas/spark/SQLAlchemy data flows
          4. Scan for Airflow DAG files → extract task dependencies
          5. Populate the knowledge graph with all discovered lineage
          6. Compute sources (in-degree=0) and sinks (out-degree=0)

        Args:
            repo_root: Absolute path to the repository root.
            knowledge_graph: Shared KnowledgeGraph to populate.

        Returns:
            Summary dict with lineage statistics and key data assets.
        """
        logger.info("Hydrologist: analyzing data flows in %s", repo_root)

        all_transformations: list[TransformationNode] = []
        all_datasets: list[DatasetNode] = []

        # Step 1: Check if this is a dbt project
        dbt_project_file = repo_root / "dbt_project.yml"
        if dbt_project_file.exists():
            logger.info("Detected dbt project — parsing model DAG")
            dbt_results = parse_dbt_project(repo_root)
            all_transformations.extend(dbt_results["transformations"])
            all_datasets.extend(dbt_results["datasets"])

            # Add config edges
            for edge in dbt_results["config_edges"]:
                knowledge_graph.add_configures_edge(edge.source, edge.target)

        # Step 2: Scan SQL files for lineage
        sql_files = list(repo_root.rglob("*.sql"))
        sql_files = [
            f for f in sql_files
            if not any(skip in f.parts for skip in {".git", "node_modules", "__pycache__", ".venv"})
        ]

        for sql_file in sql_files:
            sql_transformations = analyze_sql_file(sql_file, repo_root)
            all_transformations.extend(sql_transformations)

        logger.info("Analyzed %d SQL files", len(sql_files))

        # Step 3: Scan Python files for data flow
        py_files = list(repo_root.rglob("*.py"))
        py_files = [
            f for f in py_files
            if not any(skip in f.parts for skip in {".git", "node_modules", "__pycache__", ".venv"})
        ]

        for py_file in py_files:
            py_transforms = _analyze_python_data_flow(py_file, repo_root)
            all_transformations.extend(py_transforms)

        # Step 4: Check for Airflow DAG files
        dags_dir = repo_root / "dags"
        if dags_dir.exists():
            for dag_file in dags_dir.rglob("*.py"):
                airflow_transforms = parse_airflow_dag(dag_file, repo_root)
                all_transformations.extend(airflow_transforms)

        # Step 5: Populate the knowledge graph
        for dataset in all_datasets:
            knowledge_graph.add_dataset_node(dataset)

        for transformation in all_transformations:
            knowledge_graph.add_transformation_node(transformation)

        # Step 6: Compute sources and sinks
        lineage_subgraph = knowledge_graph.get_lineage_subgraph()
        sources = self._find_sources(lineage_subgraph)
        sinks = self._find_sinks(lineage_subgraph)

        summary = {
            "total_transformations": len(all_transformations),
            "total_datasets": len(all_datasets),
            "sql_files_analyzed": len(sql_files),
            "python_files_scanned": len(py_files),
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

    # ------------------------------------------------------------------
    # Lineage Query Methods
    # ------------------------------------------------------------------

    @staticmethod
    def blast_radius(
        knowledge_graph: KnowledgeGraph, node_name: str
    ) -> list[str]:
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
            node for node in graph.nodes()
            if graph.in_degree(node) == 0
            and graph.nodes[node].get("node_type") == "dataset"
        }

    @staticmethod
    def _find_sinks(graph: nx.DiGraph) -> set[str]:
        """Nodes with out-degree 0 (data leaves here)."""
        return {
            node for node in graph.nodes()
            if graph.out_degree(node) == 0
            and graph.nodes[node].get("node_type") == "dataset"
        }

    @staticmethod
    def trace_upstream(
        knowledge_graph: KnowledgeGraph, dataset_name: str
    ) -> list[str]:
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
            node for node in visited
            if knowledge_graph.graph.nodes[node].get("node_type") == "dataset"
        )
