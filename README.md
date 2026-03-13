# The Brownfield Cartographer

> 🗺️ **A multi-agent codebase intelligence system for rapid FDE onboarding in production environments.**

The Brownfield Cartographer ingests any GitHub repository (or local path) and produces a living, queryable knowledge graph of the system's architecture, data flows, and semantic structure.

## Quick Start

### Prerequisites

- Python 3.12+
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

### Environment Setup (for LLM-powered analysis)

Create a `.env` file in the project root with one of these providers:

```bash
# Option 1: OpenAI
OPENAI_API_KEY=sk-...

# Option 2: Google Gemini
GEMINI_API_KEY=AIza...

# Option 3: OpenRouter (any model)
OPENROUTER_API_KEY=sk-or-...

# Option 4: Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Override the default model
LLM_MODEL=gpt-4o-mini
```

> **Note:** LLM configuration is optional. The Surveyor and Hydrologist agents work fully without an LLM. The Semanticist (purpose statements, domain clustering) requires an LLM API key.

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

#### Query a Previously Analyzed Codebase

```bash
# Start the Navigator interactive REPL
uv run python -m src.cli query .

# Or specify the path containing .cartography/
uv run python -m src.cli query /path/to/analyzed/repo
```

**Navigator commands:**

| Command                  | Description                                   |
| ------------------------ | --------------------------------------------- |
| `find <concept>`         | Search for implementations matching a concept |
| `lineage <dataset>`      | Trace upstream data lineage                   |
| `lineage-down <dataset>` | Trace downstream data lineage                 |
| `blast <module>`         | Show blast radius of a module                 |
| `explain <path>`         | Explain a module's purpose and metadata       |
| `summary`                | Show graph summary statistics                 |
| Any natural language     | LLM-powered query (requires API key)          |

### Output

The Cartographer generates artifacts in the `.cartography/` directory:

| File                      | Description                                   |
| ------------------------- | --------------------------------------------- |
| `module_graph.json`       | Module import graph (NetworkX JSON format)    |
| `lineage_graph.json`      | Data lineage DAG (datasets + transformations) |
| `analysis_summary.json`   | Combined analysis results from all agents     |
| `CODEBASE.md`             | Living context file for AI agent injection    |
| `onboarding_brief.md`     | FDE Day-One Brief answering 5 key questions   |
| `cartography_trace.jsonl` | Audit log of every analysis action            |
| `state.json`              | Incremental update state (last commit hash)   |

## Architecture

The system uses a **multi-agent pipeline** with four specialized analysis agents:

```
Input (GitHub URL or local path)
  │
  ▼
┌─────────────────────────────────────────────────┐
│  Orchestrator (src/orchestrator.py)              │
│                                                  │
│  1. 🔭 Surveyor    — Static structure analysis   │
│  2. 🌊 Hydrologist — Data lineage analysis       │
│  3. 🧠 Semanticist — LLM-powered analysis        │
│  4. 📜 Archivist   — Living context generation   │
│                                                  │
│  All agents write to shared KnowledgeGraph       │
└─────────────────────────────────────────────────┘
  │
  ▼
.cartography/ artifacts
```

1. **Surveyor** — Static structure analysis (tree-sitter AST parsing, PageRank, git velocity, dead code detection)
2. **Hydrologist** — Data lineage analysis (sqlglot SQL parsing, dbt/Airflow config parsing, Python data flow detection)
3. **Semanticist** — LLM-powered purpose analysis (purpose statements, doc drift detection, domain clustering, Day-One Q&A)
4. **Archivist** — Living context generation (CODEBASE.md, onboarding brief, trace logging)

**Query Interface:**

5. **Navigator** — LangGraph agent with 4 tools for interactive codebase exploration

All agents write to a shared **KnowledgeGraph** (NetworkX-backed) with typed Pydantic node and edge schemas.

## Project Structure

```
src/
├── cli.py                          # CLI entry point (typer): analyze + query
├── orchestrator.py                 # Pipeline coordinator (4-agent sequence)
├── models/
│   └── schemas.py                  # Pydantic node/edge schemas
├── analyzers/
│   ├── tree_sitter_analyzer.py     # Multi-language AST parsing
│   ├── sql_lineage.py              # sqlglot SQL dependency extraction
│   └── dag_config_parser.py        # dbt/Airflow config parsing
├── agents/
│   ├── surveyor.py                 # Static structure agent
│   ├── hydrologist.py              # Data lineage agent
│   ├── semanticist.py              # LLM-powered semantic analysis
│   ├── archivist.py                # Living context artifact generation
│   └── navigator.py               # LangGraph query agent (4 tools)
└── graph/
    └── knowledge_graph.py          # NetworkX wrapper + serialization
```

## Supported LLM Providers

| Provider      | Env Variable         | Default Model                      | Notes               |
| ------------- | -------------------- | ---------------------------------- | ------------------- |
| OpenAI        | `OPENAI_API_KEY`     | `gpt-4o-mini`                      | Best quality        |
| Google Gemini | `GEMINI_API_KEY`     | `gemini-2.0-flash`                 | Free tier available |
| OpenRouter    | `OPENROUTER_API_KEY` | `google/gemini-2.0-flash-exp:free` | Multi-model access  |
| Ollama        | `OLLAMA_BASE_URL`    | `llama3`                           | Local, private      |

Override the model with `LLM_MODEL=<model-name>`.
