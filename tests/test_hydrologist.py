"""
Tests for the Hydrologist Agent — Data Lineage Construction.

Validates blast_radius, find_sources, find_sinks behavior on
realistic dbt and Airflow examples.
"""

from __future__ import annotations

import pytest

from src.agents.hydrologist import Hydrologist
from src.graph.knowledge_graph import KnowledgeGraph
from src.models.schemas import (
    DatasetNode,
    StorageType,
    TransformationNode,
    TransformationType,
)


# ---------------------------------------------------------------------------
# Fixtures: Build a realistic dbt-style lineage graph
# ---------------------------------------------------------------------------


@pytest.fixture
def dbt_lineage_graph() -> KnowledgeGraph:
    """
    Construct a KnowledgeGraph mimicking the jaffle-shop dbt project.

    Lineage:
      ecom.raw_customers → stg_customers → customers
      ecom.raw_orders    → stg_orders    → orders   → customers
      ecom.raw_items     → stg_order_items → order_items → orders
      ecom.raw_products  → stg_products   → order_items
                                           → products
      ecom.raw_supplies  → stg_supplies   → order_items
                                           → supplies
      ecom.raw_stores    → stg_locations  → locations
    """
    kg = KnowledgeGraph()

    # --- Source datasets (in-degree = 0) ---
    sources = [
        "ecom.raw_customers",
        "ecom.raw_orders",
        "ecom.raw_items",
        "ecom.raw_products",
        "ecom.raw_supplies",
        "ecom.raw_stores",
    ]
    for src in sources:
        kg.add_dataset_node(
            DatasetNode(name=src, storage_type=StorageType.TABLE, is_source_of_truth=True)
        )

    # --- Staging transformations ---
    staging = [
        ("stg_customers.sql", ["ecom.raw_customers"], ["stg_customers"]),
        ("stg_orders.sql", ["ecom.raw_orders"], ["stg_orders"]),
        ("stg_order_items.sql", ["ecom.raw_items"], ["stg_order_items"]),
        ("stg_products.sql", ["ecom.raw_products"], ["stg_products"]),
        ("stg_supplies.sql", ["ecom.raw_supplies"], ["stg_supplies"]),
        ("stg_locations.sql", ["ecom.raw_stores"], ["stg_locations"]),
    ]
    for source_file, src_datasets, tgt_datasets in staging:
        kg.add_transformation_node(
            TransformationNode(
                source_datasets=src_datasets,
                target_datasets=tgt_datasets,
                transformation_type=TransformationType.SQL_QUERY,
                source_file=f"models/staging/{source_file}",
                line_range=(1, 30),
            )
        )

    # --- Mart transformations ---
    marts = [
        ("order_items.sql", ["stg_orders", "stg_order_items", "stg_products", "stg_supplies"], ["order_items"]),
        ("orders.sql", ["stg_orders", "order_items"], ["orders"]),
        ("customers.sql", ["stg_customers", "orders"], ["customers"]),
        ("products.sql", ["stg_products"], ["products"]),
        ("supplies.sql", ["stg_supplies"], ["supplies"]),
        ("locations.sql", ["stg_locations"], ["locations"]),
    ]
    for source_file, src_datasets, tgt_datasets in marts:
        kg.add_transformation_node(
            TransformationNode(
                source_datasets=src_datasets,
                target_datasets=tgt_datasets,
                transformation_type=TransformationType.SQL_QUERY,
                source_file=f"models/marts/{source_file}",
                line_range=(1, 60),
            )
        )

    return kg


@pytest.fixture
def airflow_lineage_graph() -> KnowledgeGraph:
    """
    Construct a KnowledgeGraph mimicking an Airflow ETL pipeline.

    Lineage:
      s3_raw_events → extract_task → staging_events
      staging_events → transform_task → analytics_events
      analytics_events → load_task → warehouse_events
    """
    kg = KnowledgeGraph()

    kg.add_dataset_node(DatasetNode(name="s3_raw_events", storage_type=StorageType.FILE, is_source_of_truth=True))
    kg.add_dataset_node(DatasetNode(name="staging_events", storage_type=StorageType.TABLE))
    kg.add_dataset_node(DatasetNode(name="analytics_events", storage_type=StorageType.TABLE))
    kg.add_dataset_node(DatasetNode(name="warehouse_events", storage_type=StorageType.TABLE))

    kg.add_transformation_node(
        TransformationNode(
            source_datasets=["s3_raw_events"],
            target_datasets=["staging_events"],
            transformation_type=TransformationType.PYTHON_TRANSFORM,
            source_file="dags/etl_pipeline.py",
            line_range=(10, 40),
        )
    )
    kg.add_transformation_node(
        TransformationNode(
            source_datasets=["staging_events"],
            target_datasets=["analytics_events"],
            transformation_type=TransformationType.SQL_QUERY,
            source_file="dags/etl_pipeline.py",
            line_range=(42, 80),
        )
    )
    kg.add_transformation_node(
        TransformationNode(
            source_datasets=["analytics_events"],
            target_datasets=["warehouse_events"],
            transformation_type=TransformationType.PYTHON_TRANSFORM,
            source_file="dags/etl_pipeline.py",
            line_range=(82, 120),
        )
    )

    return kg


# ---------------------------------------------------------------------------
# Tests: find_sources / find_sinks
# ---------------------------------------------------------------------------


class TestFindSourcesSinks:
    """Validate source/sink detection via in-degree=0 / out-degree=0 queries."""

    def test_dbt_sources_are_raw_tables(self, dbt_lineage_graph: KnowledgeGraph) -> None:
        """All 6 raw ecom tables should be identified as sources."""
        sources = Hydrologist.find_sources(dbt_lineage_graph)
        assert len(sources) == 6
        for src in sources:
            assert src.startswith("ecom.raw_"), f"Unexpected source: {src}"

    def test_dbt_sinks_are_mart_outputs(self, dbt_lineage_graph: KnowledgeGraph) -> None:
        """The 5 terminal mart tables should be identified as sinks."""
        sinks = Hydrologist.find_sinks(dbt_lineage_graph)
        expected_sinks = {"customers", "orders", "order_items", "products", "supplies", "locations"}
        assert set(sinks) == expected_sinks

    def test_airflow_single_source(self, airflow_lineage_graph: KnowledgeGraph) -> None:
        """The single S3 source should be the only data entry point."""
        sources = Hydrologist.find_sources(airflow_lineage_graph)
        assert sources == ["s3_raw_events"]

    def test_airflow_single_sink(self, airflow_lineage_graph: KnowledgeGraph) -> None:
        """The warehouse table should be the only data exit point."""
        sinks = Hydrologist.find_sinks(airflow_lineage_graph)
        assert sinks == ["warehouse_events"]


# ---------------------------------------------------------------------------
# Tests: blast_radius (BFS downstream traversal)
# ---------------------------------------------------------------------------


class TestBlastRadius:
    """Validate downstream dependency traversal for impact analysis."""

    def test_blast_radius_of_stg_orders(self, dbt_lineage_graph: KnowledgeGraph) -> None:
        """
        stg_orders feeds order_items and orders directly.
        orders feeds customers transitively.
        Blast radius should include: transformation nodes + order_items, orders, customers.
        """
        affected = Hydrologist.blast_radius(dbt_lineage_graph, "stg_orders")
        affected_names = set(affected)

        # Must include the critical downstream outputs
        assert "order_items" in affected_names
        assert "orders" in affected_names
        assert "customers" in affected_names
        # Should NOT include unrelated marts
        assert "locations" not in affected_names
        assert "products" not in affected_names

    def test_blast_radius_of_raw_source(self, dbt_lineage_graph: KnowledgeGraph) -> None:
        """Blasting a raw source should cascade through staging → marts."""
        affected = Hydrologist.blast_radius(dbt_lineage_graph, "ecom.raw_orders")
        affected_names = set(affected)
        # Must cascade through stg_orders → order_items/orders → customers
        assert "stg_orders" in affected_names
        assert "customers" in affected_names

    def test_blast_radius_of_sink_is_empty(self, dbt_lineage_graph: KnowledgeGraph) -> None:
        """A sink node (out-degree=0 dataset) should have zero blast radius."""
        affected = Hydrologist.blast_radius(dbt_lineage_graph, "locations")
        assert len(affected) == 0

    def test_blast_radius_airflow_cascade(self, airflow_lineage_graph: KnowledgeGraph) -> None:
        """Blasting the source in a linear pipeline should reach the final sink."""
        affected = Hydrologist.blast_radius(airflow_lineage_graph, "s3_raw_events")
        affected_names = set(affected)
        assert "staging_events" in affected_names
        assert "analytics_events" in affected_names
        assert "warehouse_events" in affected_names

    def test_blast_radius_nonexistent_node(self, dbt_lineage_graph: KnowledgeGraph) -> None:
        """Requesting blast radius for a nonexistent node should return empty."""
        affected = Hydrologist.blast_radius(dbt_lineage_graph, "does_not_exist")
        assert len(affected) == 0


# ---------------------------------------------------------------------------
# Tests: Edge metadata conventions
# ---------------------------------------------------------------------------


class TestEdgeMetadata:
    """Verify that edge metadata follows documented conventions."""

    def test_consumes_edges_have_type(self, dbt_lineage_graph: KnowledgeGraph) -> None:
        """All CONSUMES edges should carry edge_type='CONSUMES'."""
        for u, v, data in dbt_lineage_graph.graph.edges(data=True):
            if data.get("edge_type") == "CONSUMES":
                # Source should be a dataset, target a transformation
                src_type = dbt_lineage_graph.graph.nodes[u].get("node_type")
                tgt_type = dbt_lineage_graph.graph.nodes[v].get("node_type")
                assert src_type == "dataset", f"CONSUMES edge source should be dataset, got {src_type}"
                assert tgt_type == "transformation", f"CONSUMES edge target should be transformation, got {tgt_type}"

    def test_produces_edges_have_type(self, dbt_lineage_graph: KnowledgeGraph) -> None:
        """All PRODUCES edges should carry edge_type='PRODUCES'."""
        for u, v, data in dbt_lineage_graph.graph.edges(data=True):
            if data.get("edge_type") == "PRODUCES":
                src_type = dbt_lineage_graph.graph.nodes[u].get("node_type")
                tgt_type = dbt_lineage_graph.graph.nodes[v].get("node_type")
                assert src_type == "transformation", f"PRODUCES edge source should be transformation, got {src_type}"
                assert tgt_type == "dataset", f"PRODUCES edge target should be dataset, got {tgt_type}"

    def test_transformation_nodes_have_metadata(self, dbt_lineage_graph: KnowledgeGraph) -> None:
        """All transformation nodes should carry source_file and line range."""
        for node, data in dbt_lineage_graph.graph.nodes(data=True):
            if data.get("node_type") == "transformation":
                assert "source_file" in data, f"Missing source_file on {node}"
                assert "line_start" in data, f"Missing line_start on {node}"
                assert "line_end" in data, f"Missing line_end on {node}"
                assert "transformation_type" in data, f"Missing transformation_type on {node}"
                assert data["line_start"] <= data["line_end"], f"Invalid line range on {node}"
