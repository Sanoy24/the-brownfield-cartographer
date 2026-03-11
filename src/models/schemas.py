"""
Pydantic schemas for the Brownfield Cartographer knowledge graph.

Defines all node and edge types that form the central knowledge representation
of a codebase's architecture, data flows, and semantic structure.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class StorageType(str, Enum):
    """Classification of how a dataset is physically stored."""
    TABLE = "table"
    FILE = "file"
    STREAM = "stream"
    API = "api"


class TransformationType(str, Enum):
    """Classification of data transformation operations."""
    SQL_QUERY = "sql_query"
    PYTHON_TRANSFORM = "python_transform"
    SPARK_JOB = "spark_job"
    DBT_MODEL = "dbt_model"
    NOTEBOOK = "notebook"
    CONFIG_PIPELINE = "config_pipeline"


class Language(str, Enum):
    """Supported programming languages for AST parsing."""
    PYTHON = "python"
    SQL = "sql"
    YAML = "yaml"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Node Types
# ---------------------------------------------------------------------------

class ModuleNode(BaseModel):
    """
    Represents a single source file in the codebase.

    Captures structural metadata, semantic purpose, and change velocity
    to identify architectural hotspots and dead code candidates.
    """

    path: str = Field(description="Relative path from repo root")
    language: Language = Field(default=Language.UNKNOWN)
    purpose_statement: Optional[str] = Field(
        default=None,
        description="LLM-generated summary of what this module does (not how)",
    )
    domain_cluster: Optional[str] = Field(
        default=None,
        description="Inferred business domain (e.g. ingestion, transformation)",
    )
    complexity_score: float = Field(
        default=0.0,
        description="Cyclomatic complexity or lines-of-code proxy",
    )
    change_velocity_30d: int = Field(
        default=0,
        description="Number of commits touching this file in the last 30 days",
    )
    is_dead_code_candidate: bool = Field(
        default=False,
        description="True if exported symbols are never imported elsewhere",
    )
    last_modified: Optional[datetime] = Field(default=None)

    # Structural data extracted by the Surveyor
    imports: list[str] = Field(
        default_factory=list,
        description="List of module paths this file imports",
    )
    public_functions: list[str] = Field(
        default_factory=list,
        description="Names of public (non-underscore) functions defined here",
    )
    classes: list[str] = Field(
        default_factory=list,
        description="Names of classes defined here",
    )
    lines_of_code: int = Field(default=0)
    comment_ratio: float = Field(
        default=0.0,
        description="Fraction of lines that are comments (0.0–1.0)",
    )


class DatasetNode(BaseModel):
    """
    Represents a data asset — a table, file, stream, or API endpoint.

    Serves as a vertex in the data lineage graph, connected by
    PRODUCES / CONSUMES edges from TransformationNodes.
    """

    name: str = Field(description="Qualified name (e.g. schema.table or file path)")
    storage_type: StorageType = Field(default=StorageType.TABLE)
    schema_snapshot: Optional[dict] = Field(
        default=None,
        description="Column-level schema if available",
    )
    freshness_sla: Optional[str] = Field(default=None)
    owner: Optional[str] = Field(default=None)
    is_source_of_truth: bool = Field(default=False)


class FunctionNode(BaseModel):
    """
    Represents a single callable (function or method) within a module.

    Used for call-graph analysis and identifying high-impact functions.
    """

    qualified_name: str = Field(
        description="Fully qualified name (e.g. module.ClassName.method)"
    )
    parent_module: str = Field(description="Relative path of the containing module")
    signature: str = Field(default="", description="Function signature as source text")
    purpose_statement: Optional[str] = Field(default=None)
    call_count_within_repo: int = Field(
        default=0,
        description="How many other functions in the repo call this one",
    )
    is_public_api: bool = Field(
        default=True,
        description="False if the name starts with an underscore",
    )
    line_start: int = Field(default=0)
    line_end: int = Field(default=0)


class TransformationNode(BaseModel):
    """
    Represents a data transformation operation that connects datasets.

    Bridges the gap between source datasets and target datasets in the
    data lineage graph with full provenance metadata.
    """

    source_datasets: list[str] = Field(default_factory=list)
    target_datasets: list[str] = Field(default_factory=list)
    transformation_type: TransformationType = Field(
        default=TransformationType.SQL_QUERY
    )
    source_file: str = Field(description="File where this transformation is defined")
    line_range: tuple[int, int] = Field(
        default=(0, 0),
        description="(start_line, end_line) in the source file",
    )
    sql_query_if_applicable: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Edge Types
# ---------------------------------------------------------------------------

class ImportsEdge(BaseModel):
    """Module A imports from module B."""
    source: str = Field(description="Importing module path")
    target: str = Field(description="Imported module path")
    weight: int = Field(default=1, description="Number of symbols imported")


class ProducesEdge(BaseModel):
    """Transformation node writes to a dataset."""
    source: str = Field(description="Transformation identifier")
    target: str = Field(description="Dataset name being produced")


class ConsumesEdge(BaseModel):
    """Transformation node reads from a dataset."""
    source: str = Field(description="Transformation identifier")
    target: str = Field(description="Dataset name being consumed")


class CallsEdge(BaseModel):
    """Function A calls function B."""
    source: str = Field(description="Calling function qualified name")
    target: str = Field(description="Called function qualified name")


class ConfiguresEdge(BaseModel):
    """A config file configures a module or pipeline."""
    source: str = Field(description="Config file path")
    target: str = Field(description="Module or pipeline being configured")
