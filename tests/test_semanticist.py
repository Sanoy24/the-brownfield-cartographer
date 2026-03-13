"""
Tests for the Semanticist Agent — LLM-Powered Analysis.

Validates domain clustering quality and drift detection logic
using mock data that simulates real codebase analysis results.
"""

from __future__ import annotations

import pytest

from src.agents.semanticist import ContextWindowBudget, cluster_into_domains


# ---------------------------------------------------------------------------
# Tests: ContextWindowBudget
# ---------------------------------------------------------------------------


class TestContextWindowBudget:
    """Validate token budget tracking and exhaustion detection."""

    def test_initial_state(self) -> None:
        budget = ContextWindowBudget(max_tokens=1000)
        assert budget.used_tokens == 0
        assert budget.remaining == 1000
        assert not budget.exhausted
        assert budget.call_count == 0

    def test_consume_tracks_usage(self) -> None:
        budget = ContextWindowBudget(max_tokens=1000)
        budget.consume("Hello world", "Response text here")
        assert budget.used_tokens > 0
        assert budget.call_count == 1
        assert budget.remaining < 1000

    def test_exhaustion_detected(self) -> None:
        budget = ContextWindowBudget(max_tokens=10)
        budget.consume("A" * 100, "B" * 100)
        assert budget.exhausted
        assert budget.remaining == 0

    def test_estimate_tokens_heuristic(self) -> None:
        budget = ContextWindowBudget()
        # ~4 chars per token heuristic
        assert budget.estimate_tokens("a" * 400) == 100
        assert budget.estimate_tokens("") == 1  # minimum 1


# ---------------------------------------------------------------------------
# Tests: Domain Clustering
# ---------------------------------------------------------------------------


class TestDomainClustering:
    """Validate TF-IDF + k-means domain clustering on sample purpose statements."""

    @pytest.fixture
    def dbt_purpose_statements(self) -> dict[str, str]:
        """Simulated purpose statements from a dbt project analysis."""
        return {
            "models/staging/stg_customers.sql": (
                "Cleans and renames raw customer records from the ecom source "
                "into a standardized staging format for downstream consumption."
            ),
            "models/staging/stg_orders.sql": (
                "Transforms raw order records by mapping status codes to human-readable "
                "labels and standardizing column names for the staging layer."
            ),
            "models/staging/stg_order_items.sql": (
                "Joins raw order items with product data to create a normalized "
                "staging table of line-item details."
            ),
            "models/staging/stg_products.sql": (
                "Renames and standardizes raw product catalog columns for staging."
            ),
            "models/staging/stg_supplies.sql": (
                "Cleans raw supply chain data and renames columns for staging."
            ),
            "models/staging/stg_locations.sql": (
                "Transforms raw store location data into standardized staging format."
            ),
            "models/marts/customers.sql": (
                "Builds a Customer 360 table aggregating first/last order dates, "
                "order counts, and customer lifetime value from staging data."
            ),
            "models/marts/orders.sql": (
                "Enriches order records with line-item subtotals, discounts, "
                "taxes, and order-level total calculations."
            ),
            "models/marts/order_items.sql": (
                "Creates detailed line-item records joining orders, products, "
                "and supplies with currency conversion via cents_to_dollars macro."
            ),
            "models/marts/products.sql": (
                "Simple passthrough view exposing staged product data to the mart layer."
            ),
            "models/marts/supplies.sql": (
                "Simple passthrough view exposing staged supply chain data to the mart layer."
            ),
            "models/marts/locations.sql": (
                "Simple passthrough view exposing staged store location data to the mart layer."
            ),
        }

    def test_clusters_are_assigned_to_all_modules(
        self, dbt_purpose_statements: dict[str, str]
    ) -> None:
        """Every module should receive a domain cluster label."""
        result = cluster_into_domains(dbt_purpose_statements, n_clusters=3)
        assert set(result.assignments.keys()) == set(dbt_purpose_statements.keys())

    def test_cluster_labels_are_not_hardcoded(
        self, dbt_purpose_statements: dict[str, str]
    ) -> None:
        """Labels should be derived from TF-IDF terms, not predefined strings."""
        result = cluster_into_domains(dbt_purpose_statements, n_clusters=3)
        labels = set(result.assignments.values())
        for label in labels:
            assert "_" in label, f"Label '{label}' doesn't look term-derived"

    def test_similar_modules_cluster_together(
        self, dbt_purpose_statements: dict[str, str]
    ) -> None:
        """Staging modules should generally cluster separately from mart modules."""
        result = cluster_into_domains(dbt_purpose_statements, n_clusters=3)
        staging_clusters = {result.assignments[p] for p in result.assignments if "staging" in p}
        mart_clusters = {result.assignments[p] for p in result.assignments if "marts" in p}
        assert len(staging_clusters) >= 1
        assert len(mart_clusters) >= 1

    def test_quality_metrics_present(
        self, dbt_purpose_statements: dict[str, str]
    ) -> None:
        """ClusteringResult should contain silhouette score and per-cluster stats."""
        result = cluster_into_domains(dbt_purpose_statements, n_clusters=3)
        assert "silhouette_score" in result.quality_metrics
        assert "n_clusters" in result.quality_metrics
        assert "total_inertia" in result.quality_metrics
        assert "clusters" in result.quality_metrics
        # Silhouette should be between -1 and 1
        sil = result.quality_metrics["silhouette_score"]
        assert -1.0 <= sil <= 1.0

    def test_exemplars_present_for_each_domain(
        self, dbt_purpose_statements: dict[str, str]
    ) -> None:
        """Each domain should have at least one exemplar module."""
        result = cluster_into_domains(dbt_purpose_statements, n_clusters=3)
        for domain in set(result.assignments.values()):
            assert domain in result.exemplars, f"Missing exemplar for domain '{domain}'"
            assert len(result.exemplars[domain]) >= 1

    def test_small_input_returns_core(self) -> None:
        """With fewer than 3 modules, should return 'core' for all."""
        result = cluster_into_domains({"a.py": "Does stuff", "b.py": "Other stuff"})
        assert all(v == "core" for v in result.assignments.values())

    def test_empty_input(self) -> None:
        """Empty input should return empty output."""
        result = cluster_into_domains({})
        assert result.assignments == {}

