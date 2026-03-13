"""
Knowledge Graph — centralized graph store for codebase intelligence.

Wraps NetworkX DiGraph with typed node/edge insertion and JSON serialization.
Serves as the single source of truth for all Cartographer analysis results,
consumed by the query interface and artifact generators.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from networkx.readwrite import json_graph

from src.models.schemas import (
    DatasetNode,
    FunctionNode,
    ModuleNode,
    TransformationNode,
)

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Unified knowledge graph for codebase architecture and data lineage.

    Maintains two logical sub-graphs within a single NetworkX DiGraph:
      - Module Graph: file-level imports and structural relationships
      - Lineage Graph: dataset-level data flow (sources → transformations → sinks)
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._module_nodes: dict[str, ModuleNode] = {}
        self._dataset_nodes: dict[str, DatasetNode] = {}
        self._function_nodes: dict[str, FunctionNode] = {}
        self._transformation_nodes: list[TransformationNode] = []

    # ------------------------------------------------------------------
    # Node insertion
    # ------------------------------------------------------------------

    def add_module_node(self, module: ModuleNode) -> None:
        """Register a source file as a module node in the graph."""
        self._module_nodes[module.path] = module
        self.graph.add_node(
            module.path,
            node_type="module",
            language=module.language.value,
            lines_of_code=module.lines_of_code,
            complexity_score=module.complexity_score,
            change_velocity_30d=module.change_velocity_30d,
            is_dead_code_candidate=module.is_dead_code_candidate,
            public_functions=module.public_functions,
            classes=module.classes,
        )

    def add_dataset_node(self, dataset: DatasetNode) -> None:
        """Register a data asset (table, file, stream) in the graph."""
        self._dataset_nodes[dataset.name] = dataset
        self.graph.add_node(
            dataset.name,
            node_type="dataset",
            storage_type=dataset.storage_type.value,
            is_source_of_truth=dataset.is_source_of_truth,
        )

    def add_function_node(self, function: FunctionNode) -> None:
        """Register a function/method in the graph."""
        self._function_nodes[function.qualified_name] = function
        self.graph.add_node(
            function.qualified_name,
            node_type="function",
            parent_module=function.parent_module,
            signature=function.signature,
            is_public_api=function.is_public_api,
        )

    def add_transformation_node(self, transformation: TransformationNode) -> None:
        """Register a data transformation and create lineage edges."""
        self._transformation_nodes.append(transformation)

        # Create a unique ID for this transformation
        tx_id = (
            f"tx:{transformation.source_file}"
            f":{transformation.line_range[0]}-{transformation.line_range[1]}"
        )
        self.graph.add_node(
            tx_id,
            node_type="transformation",
            transformation_type=transformation.transformation_type.value,
            source_file=transformation.source_file,
        )

        # Create CONSUMES edges (source datasets → transformation)
        for src_dataset in transformation.source_datasets:
            if src_dataset not in self.graph:
                self.graph.add_node(src_dataset, node_type="dataset")
            self.graph.add_edge(
                src_dataset, tx_id, edge_type="CONSUMES"
            )

        # Create PRODUCES edges (transformation → target datasets)
        for tgt_dataset in transformation.target_datasets:
            if tgt_dataset not in self.graph:
                self.graph.add_node(tgt_dataset, node_type="dataset")
            self.graph.add_edge(
                tx_id, tgt_dataset, edge_type="PRODUCES"
            )

    # ------------------------------------------------------------------
    # Edge insertion
    # ------------------------------------------------------------------

    def add_import_edge(
        self, source_module: str, target_module: str, weight: int = 1
    ) -> None:
        """Record that source_module imports from target_module."""
        self.graph.add_edge(
            source_module,
            target_module,
            edge_type="IMPORTS",
            weight=weight,
        )

    def add_calls_edge(self, caller: str, callee: str) -> None:
        """Record that one function calls another."""
        self.graph.add_edge(caller, callee, edge_type="CALLS")

    def add_configures_edge(self, config_file: str, target: str) -> None:
        """Record that a config file configures a module or pipeline."""
        self.graph.add_edge(
            config_file, target, edge_type="CONFIGURES"
        )

    # ------------------------------------------------------------------
    # Node/edge removal (for incremental updates)
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear the entire graph and all internal indices. Used before load_replace."""
        self.graph.clear()
        self._module_nodes.clear()
        self._dataset_nodes.clear()
        self._function_nodes.clear()
        self._transformation_nodes.clear()

    def remove_module_node(self, path: str) -> None:
        """Remove a module node and all edges incident to it. Used in incremental updates."""
        if path in self._module_nodes:
            del self._module_nodes[path]
        if self.graph.has_node(path):
            self.graph.remove_node(path)

    def remove_import_edges_for_module(self, path: str) -> None:
        """Remove all IMPORTS edges where path is source or target."""
        to_remove = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("edge_type") == "IMPORTS" and (u == path or v == path)
        ]
        self.graph.remove_edges_from(to_remove)

    def remove_function_nodes_for_module(self, module_path: str) -> None:
        """Remove all function nodes whose parent_module is the given path."""
        to_remove = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "function" and d.get("parent_module") == module_path
        ]
        for n in to_remove:
            self.graph.remove_node(n)
            self._function_nodes.pop(n, None)

    def remove_transformation_nodes_for_files(self, file_paths: set[str]) -> None:
        """Remove all transformation nodes whose source_file is in file_paths (and their edges)."""
        to_remove = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "transformation" and d.get("source_file") in file_paths
        ]
        for n in to_remove:
            self.graph.remove_node(n)
        self._transformation_nodes = [
            t for t in self._transformation_nodes
            if t.source_file not in file_paths
        ]

    def load_from_artifacts_replace(self, cartography_dir: Path) -> None:
        """
        Clear the graph and replace with state from .cartography/ artifacts.

        Used for incremental updates: load previous run, then re-analyze only
        changed files and merge.
        """
        self.clear()
        module_path = cartography_dir / "module_graph.json"
        if module_path.exists():
            self.load_module_graph(module_path)
        lineage_path = cartography_dir / "lineage_graph.json"
        if lineage_path.exists():
            self.load_lineage_graph(lineage_path)
        logger.info(
            "KnowledgeGraph replaced from %s — %d nodes, %d edges",
            cartography_dir,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_module_subgraph(self) -> nx.DiGraph:
        """Return a subgraph containing only module nodes and IMPORTS edges."""
        module_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "module"
        ]
        return self.graph.subgraph(module_nodes).copy()

    def get_lineage_subgraph(self) -> nx.DiGraph:
        """Return a subgraph containing dataset and transformation nodes."""
        lineage_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") in ("dataset", "transformation")
        ]
        return self.graph.subgraph(lineage_nodes).copy()

    def compute_pagerank(self, subgraph: Optional[nx.DiGraph] = None) -> dict[str, float]:
        """Compute PageRank scores to identify architectural hubs."""
        g = subgraph if subgraph is not None else self.graph
        if len(g) == 0:
            return {}
        try:
            return nx.pagerank(g)
        except nx.NetworkXException:
            logger.warning("PageRank failed (likely disconnected graph), using degree centrality")
            return nx.degree_centrality(g)

    def find_circular_dependencies(self, subgraph: Optional[nx.DiGraph] = None) -> list[list[str]]:
        """Detect strongly connected components (circular dependencies)."""
        g = subgraph if subgraph is not None else self.graph
        cycles = [
            list(component)
            for component in nx.strongly_connected_components(g)
            if len(component) > 1
        ]
        return cycles

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize_module_graph(self, output_path: Path) -> None:
        """
        Write the module import graph to JSON.

        Uses NetworkX's node-link format for easy re-loading.
        """
        module_graph = self.get_module_subgraph()
        data = json_graph.node_link_data(module_graph)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        logger.info(
            "Module graph written: %d nodes, %d edges → %s",
            module_graph.number_of_nodes(),
            module_graph.number_of_edges(),
            output_path,
        )

    def serialize_lineage_graph(self, output_path: Path) -> None:
        """
        Write the data lineage graph to JSON.

        Includes dataset nodes, transformation nodes, and all
        CONSUMES/PRODUCES edges with metadata.
        """
        lineage_graph = self.get_lineage_subgraph()
        data = json_graph.node_link_data(lineage_graph)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        logger.info(
            "Lineage graph written: %d nodes, %d edges → %s",
            lineage_graph.number_of_nodes(),
            lineage_graph.number_of_edges(),
            output_path,
        )

    # ------------------------------------------------------------------
    # Deserialization
    # ------------------------------------------------------------------

    def load_module_graph(self, input_path: Path) -> None:
        """
        Load a previously serialized module graph from JSON.

        Merges loaded nodes and edges into the current graph instance,
        restoring node attributes from the node-link JSON format.
        """
        raw = json.loads(input_path.read_text(encoding="utf-8"))
        loaded = json_graph.node_link_graph(raw)
        self.graph.update(loaded)

        # Rebuild internal module node index from loaded attributes
        for node_id, attrs in loaded.nodes(data=True):
            if attrs.get("node_type") == "module":
                self._module_nodes[node_id] = ModuleNode(
                    path=node_id,
                    language=attrs.get("language", "unknown"),
                    purpose_statement=attrs.get("purpose_statement"),
                    domain_cluster=attrs.get("domain_cluster"),
                    lines_of_code=attrs.get("lines_of_code", 0),
                    complexity_score=attrs.get("complexity_score", 0.0),
                    change_velocity_30d=attrs.get("change_velocity_30d", 0),
                    is_dead_code_candidate=attrs.get("is_dead_code_candidate", False),
                    public_functions=attrs.get("public_functions", []),
                    classes=attrs.get("classes", []),
                )

        logger.info(
            "Module graph loaded: %d nodes, %d edges ← %s",
            loaded.number_of_nodes(),
            loaded.number_of_edges(),
            input_path,
        )

    def load_lineage_graph(self, input_path: Path) -> None:
        """
        Load a previously serialized lineage graph from JSON.

        Merges dataset and transformation nodes plus CONSUMES/PRODUCES
        edges into the current graph instance.
        """
        raw = json.loads(input_path.read_text(encoding="utf-8"))
        loaded = json_graph.node_link_graph(raw)
        self.graph.update(loaded)

        # Rebuild internal dataset index from loaded attributes
        for node_id, attrs in loaded.nodes(data=True):
            if attrs.get("node_type") == "dataset":
                self._dataset_nodes[node_id] = DatasetNode(
                    name=node_id,
                    storage_type=attrs.get("storage_type", "table"),
                    is_source_of_truth=attrs.get("is_source_of_truth", False),
                )

        logger.info(
            "Lineage graph loaded: %d nodes, %d edges ← %s",
            loaded.number_of_nodes(),
            loaded.number_of_edges(),
            input_path,
        )

    @classmethod
    def from_artifacts(cls, cartography_dir: Path) -> "KnowledgeGraph":
        """
        Reconstruct a KnowledgeGraph from serialized .cartography/ artifacts.

        Args:
            cartography_dir: Path to the .cartography/ directory containing
                             module_graph.json and lineage_graph.json.

        Returns:
            A KnowledgeGraph instance populated with the saved data.
        """
        kg = cls()

        module_path = cartography_dir / "module_graph.json"
        if module_path.exists():
            kg.load_module_graph(module_path)

        lineage_path = cartography_dir / "lineage_graph.json"
        if lineage_path.exists():
            kg.load_lineage_graph(lineage_path)

        logger.info(
            "KnowledgeGraph restored from %s — %d nodes, %d edges",
            cartography_dir,
            kg.graph.number_of_nodes(),
            kg.graph.number_of_edges(),
        )
        return kg

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a quick summary of graph contents."""
        transformation_count = sum(
            1 for _, d in self.graph.nodes(data=True)
            if d.get("node_type") == "transformation"
        )
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "modules": len(self._module_nodes),
            "datasets": len(self._dataset_nodes),
            "functions": len(self._function_nodes),
            "transformations": transformation_count,
            "circular_dependencies": len(self.find_circular_dependencies()),
        }
