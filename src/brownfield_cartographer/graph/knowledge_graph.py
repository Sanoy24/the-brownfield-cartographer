"""
Knowledge Graph — centralized graph store for codebase intelligence.

Wraps NetworkX DiGraph with typed node/edge insertion and JSON serialization.
Serves as the single source of truth for all Cartographer analysis results,
consumed by the query interface and artifact generators.
"""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from networkx.readwrite import json_graph

from brownfield_cartographer.models.schemas import DatasetNode, FunctionNode, ModuleNode, TransformationNode

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
        self._require_node_type(module.path, "module")
        self._module_nodes[module.path] = module
        self.graph.add_node(module.path, node_type="module", **module.model_dump(mode="json"))

    def add_dataset_node(self, dataset: DatasetNode) -> None:
        """Register a data asset (table, file, stream) in the graph."""
        self._require_node_type(dataset.name, "dataset")
        self._dataset_nodes[dataset.name] = dataset
        self.graph.add_node(dataset.name, node_type="dataset", **dataset.model_dump(mode="json"))

    def add_function_node(self, function: FunctionNode) -> None:
        """Register a function/method in the graph."""
        self._require_node_type(function.qualified_name, "function")
        self._function_nodes[function.qualified_name] = function
        self.graph.add_node(function.qualified_name, node_type="function", **function.model_dump(mode="json"))

    def add_transformation_node(self, transformation: TransformationNode) -> None:
        """Register a data transformation and create lineage edges."""
        # Include operation identity: multiple statements/tasks can share a span.
        identity = json.dumps(transformation.model_dump(mode="json"), sort_keys=True)
        tx_id = (
            f"tx:{transformation.source_file}"
            f":{transformation.line_range[0]}-{transformation.line_range[1]}:"
            f"{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
        )
        if tx_id not in self.graph:
            self._transformation_nodes.append(transformation)
        self.graph.add_node(
            tx_id,
            node_type="transformation",
            **transformation.model_dump(mode="json"),
            line_start=transformation.line_range[0],
            line_end=transformation.line_range[1],
        )
        if transformation.source_file not in self.graph:
            self.add_module_node(ModuleNode(path=transformation.source_file))
        self.graph.add_edge(transformation.source_file, tx_id, edge_type="DEFINES",
                            source_file=transformation.source_file,
                            line_range=list(transformation.line_range), confidence="static")
        heuristic = transformation.transformation_type.value in {"python_transform", "config_pipeline"}
        evidence = {"source_file": transformation.source_file,
                    "line_range": list(transformation.line_range),
                    "confidence": transformation.confidence or ("inferred" if heuristic else "syntax"),
                    "analysis_method": transformation.analysis_method or ("scope_heuristic" if heuristic else "sql_or_dbt_reference")}
        if transformation.artifact_file is not None:
            evidence.update(artifact_file=transformation.artifact_file, resource_id=transformation.resource_id)
        self.graph.nodes[tx_id].update(evidence)

        # Create CONSUMES edges (source datasets → transformation)
        for src_dataset in transformation.source_datasets:
            if src_dataset not in self.graph:
                self.add_dataset_node(DatasetNode(name=src_dataset))
            self.graph.add_edge(src_dataset, tx_id, edge_type="CONSUMES", **evidence)

        # Create PRODUCES edges (transformation → target datasets)
        for tgt_dataset in transformation.target_datasets:
            if tgt_dataset not in self.graph:
                self.add_dataset_node(DatasetNode(name=tgt_dataset))
            self.graph.add_edge(tx_id, tgt_dataset, edge_type="PRODUCES", **evidence)

    # ------------------------------------------------------------------
    # Edge insertion
    # ------------------------------------------------------------------

    def add_import_edge(
        self, source_module: str, target_module: str, weight: int = 1,
        line_range: tuple[int, int] | None = None,
    ) -> None:
        """Record that source_module imports from target_module."""
        self.graph.add_edge(
            source_module,
            target_module,
            edge_type="IMPORTS",
            weight=weight,
            source_file=source_module,
            confidence="static",
            line_range=list(line_range) if line_range else None,
            analysis_method="python_import",
        )

    def add_calls_edge(self, caller: str, callee: str) -> None:
        """Record that one function calls another."""
        self.graph.add_edge(caller, callee, edge_type="CALLS")

    def add_configures_edge(self, config_file: str, target: str) -> None:
        """Record that a config file configures a module or pipeline."""
        if config_file not in self.graph:
            self.graph.add_node(config_file, node_type="config", path=config_file)
        self.graph.add_edge(config_file, target, edge_type="CONFIGURES", source_file=config_file)

    def impact_neighbors(self, node: str):
        """Potential consumers, respecting the semantics of each relationship."""
        for neighbor in self.graph.predecessors(node):
            edge = self.graph.edges[neighbor, node]
            if edge.get("edge_type") in {"IMPORTS", "CALLS"}:
                yield neighbor, edge
        for neighbor in self.graph.successors(node):
            edge = self.graph.edges[node, neighbor]
            if edge.get("edge_type") in {"CONSUMES", "PRODUCES", "DEFINES", "CONFIGURES"}:
                yield neighbor, edge

    def lineage_neighbors(self, node: str, direction: str):
        if direction not in {"upstream", "downstream"}:
            raise ValueError("Direction must be upstream or downstream")
        neighbors = self.graph.predecessors(node) if direction == "upstream" else self.graph.successors(node)
        for neighbor in neighbors:
            edge = self.graph.edges[neighbor, node] if direction == "upstream" else self.graph.edges[node, neighbor]
            if edge.get("edge_type") in {"CONSUMES", "PRODUCES"}:
                yield neighbor

    def clear_lineage(self) -> None:
        """Remove observations owned by lineage extractors before a safe rebuild."""
        nodes = [n for n, d in self.graph.nodes(data=True)
                 if d.get("node_type") in {"dataset", "transformation", "config"}]
        self.graph.remove_nodes_from(nodes)
        self._dataset_nodes.clear()
        self._transformation_nodes.clear()

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
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("edge_type") == "IMPORTS" and (u == path or v == path)
        ]
        self.graph.remove_edges_from(to_remove)

    def remove_function_nodes_for_module(self, module_path: str) -> None:
        """Remove all function nodes whose parent_module is the given path."""
        to_remove = [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "function"
            and d.get("parent_module") == module_path
        ]
        for n in to_remove:
            self.graph.remove_node(n)
            self._function_nodes.pop(n, None)

    def remove_transformation_nodes_for_files(self, file_paths: set[str]) -> None:
        """Remove all transformation nodes whose source_file is in file_paths (and their edges)."""
        to_remove = [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "transformation"
            and d.get("source_file") in file_paths
        ]
        for n in to_remove:
            self.graph.remove_node(n)
        self._transformation_nodes = [
            t for t in self._transformation_nodes if t.source_file not in file_paths
        ]

    def load_from_artifacts_replace(self, cartography_dir: Path) -> None:
        """
        Clear the graph and replace with state from .cartography/ artifacts.

        Used for incremental updates: load previous run, then re-analyze only
        changed files and merge.
        """
        self.clear()
        if (cartography_dir / "index.json").exists():
            self.load_index(cartography_dir / "index.json")
            return
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
        self._restore_indices()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_module_subgraph(self) -> nx.DiGraph:
        """Return a subgraph containing only module nodes and IMPORTS edges."""
        module_nodes = [
            n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "module"
        ]
        return self.graph.subgraph(module_nodes).copy()

    def get_lineage_subgraph(self) -> nx.DiGraph:
        """Return a subgraph containing dataset and transformation nodes."""
        lineage_nodes = [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") in ("dataset", "transformation")
        ]
        return self.graph.subgraph(lineage_nodes).copy()

    def compute_pagerank(
        self, subgraph: Optional[nx.DiGraph] = None
    ) -> dict[str, float]:
        """Compute PageRank scores to identify architectural hubs."""
        g = subgraph if subgraph is not None else self.graph
        if len(g) == 0:
            return {}
        try:
            return nx.pagerank(g)
        except nx.NetworkXException:
            logger.warning(
                "PageRank failed (likely disconnected graph), using degree centrality"
            )
            return nx.degree_centrality(g)

    def find_circular_dependencies(
        self, subgraph: Optional[nx.DiGraph] = None
    ) -> list[list[str]]:
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
        nodes = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") in {"module", "function"}]
        module_graph = self.graph.subgraph(nodes).copy()
        data = json_graph.node_link_data(module_graph)
        from brownfield_cartographer.persistence import write_json
        write_json(output_path, data)
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
        from brownfield_cartographer.persistence import write_json
        write_json(output_path, data)
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
        if (cartography_dir / "index.json").exists():
            kg.load_index(cartography_dir / "index.json")
            return kg

        module_path = cartography_dir / "module_graph.json"
        if module_path.exists():
            kg.load_module_graph(module_path)

        lineage_path = cartography_dir / "lineage_graph.json"
        if lineage_path.exists():
            kg.load_lineage_graph(lineage_path)
        kg._restore_indices()

        logger.info(
            "KnowledgeGraph restored from %s — %d nodes, %d edges",
            cartography_dir,
            kg.graph.number_of_nodes(),
            kg.graph.number_of_edges(),
        )
        return kg

    def serialize_index(self, output_path: Path) -> None:
        """Persist the complete graph with an explicit artifact version."""
        from brownfield_cartographer.persistence import write_json
        write_json(output_path, {"schema_version": 1,
                                 "graph": json_graph.node_link_data(self.graph, edges="edges")})

    def load_index(self, input_path: Path) -> None:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError("Unsupported index schema version; rebuild the index")
        graph = json_graph.node_link_graph(raw["graph"], edges="edges")
        if not graph.is_directed() or graph.is_multigraph():
            raise ValueError("Expected a directed graph index")
        self.clear()
        self.graph = graph
        self._restore_indices()

    def _restore_indices(self) -> None:
        self._module_nodes.clear()
        self._dataset_nodes.clear()
        self._function_nodes.clear()
        self._transformation_nodes.clear()
        for node, attrs in self.graph.nodes(data=True):
            kind = attrs.get("node_type")
            if kind == "module":
                self._module_nodes[node] = ModuleNode.model_validate({"path": node, **attrs})
            elif kind == "dataset":
                self._dataset_nodes[node] = DatasetNode.model_validate({"name": node, **attrs})
            elif kind == "function":
                self._function_nodes[node] = FunctionNode.model_validate({"qualified_name": node, **attrs})
            elif kind == "transformation":
                self._transformation_nodes.append(TransformationNode.model_validate({
                    "line_range": [attrs.get("line_start", 0), attrs.get("line_end", 0)], **attrs,
                }))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a quick summary of graph contents."""
        transformation_count = sum(
            1
            for _, d in self.graph.nodes(data=True)
            if d.get("node_type") == "transformation"
        )
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "modules": len(self._module_nodes),
            "datasets": len(self._dataset_nodes),
            "functions": len(self._function_nodes),
            "transformations": transformation_count,
            "circular_dependencies": len(self.find_circular_dependencies(self.get_module_subgraph())),
        }

    def _require_node_type(self, node: str, kind: str) -> None:
        existing = self.graph.nodes.get(node, {}).get("node_type")
        if existing is not None and existing != kind:
            raise ValueError(f"Node identity collision for {node}: {existing} versus {kind}")
