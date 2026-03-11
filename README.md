# The Brownfield Cartographer

> 🗺️ **A multi-agent codebase intelligence system for rapid FDE onboarding in production environments.**

The Brownfield Cartographer ingests any GitHub repository (or local path) and produces a living, queryable knowledge graph of the system's architecture, data flows, and semantic structure.

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Installation

```bash
# Clone this repository
git clone https://github.com/Sanoy24/the-brownfield-cartographer.git
cd the-brownfield-cartographer

# Install dependencies with uv
uv sync
```

### Usage

#### Analyze a GitHub Repository

```bash
# Point the Cartographer at any GitHub repo
uv run python -m src.cli analyze https://github.com/dbt-labs/jaffle_shop

# With verbose logging
uv run python -m src.cli analyze https://github.com/dbt-labs/jaffle_shop -v

# Custom output directory
uv run python -m src.cli analyze https://github.com/dbt-labs/jaffle_shop -o ./results
```

#### Analyze a Local Repository

```bash
uv run python -m src.cli analyze /path/to/local/repo
```

### Output

The Cartographer generates artifacts in the `.cartography/` directory:

| File                    | Description                                   |
| ----------------------- | --------------------------------------------- |
| `module_graph.json`     | Module import graph (NetworkX JSON format)    |
| `lineage_graph.json`    | Data lineage DAG (datasets + transformations) |
| `analysis_summary.json` | Combined analysis results from all agents     |

## Architecture

The system uses a **multi-agent pipeline**:

1. **Surveyor** — Static structure analysis (tree-sitter AST parsing, PageRank, git velocity, dead code detection)
2. **Hydrologist** — Data lineage analysis (sqlglot SQL parsing, dbt/Airflow config parsing, Python data flow detection)
3. **Semanticist** — LLM-powered purpose analysis _(final submission)_
4. **Archivist** — Living context generation _(final submission)_

All agents write to a shared **KnowledgeGraph** (NetworkX-backed) with typed Pydantic node and edge schemas.

## Project Structure

```
src/
├── cli.py                          # CLI entry point (typer)
├── orchestrator.py                 # Pipeline coordinator
├── models/
│   └── schemas.py                  # Pydantic node/edge schemas
├── analyzers/
│   ├── tree_sitter_analyzer.py     # Multi-language AST parsing
│   ├── sql_lineage.py              # sqlglot SQL dependency extraction
│   └── dag_config_parser.py        # dbt/Airflow config parsing
├── agents/
│   ├── surveyor.py                 # Static structure agent
│   └── hydrologist.py              # Data lineage agent
└── graph/
    └── knowledge_graph.py          # NetworkX wrapper + serialization
```
