# Brownfield Cartographer

A local-first repository analysis engine for evidence-backed code structure,
dependency impact, and data lineage. The Python API, deterministic CLI, and
read-only local MCP server share one query engine; reasoning belongs to the
developer's existing AI assistant.

The current branch implements the local analysis foundation.
The CLI includes `analyze`, `overview`, `find`, `impact`, and `lineage`. See the
[milestone notes](docs/FOUNDATION_MILESTONE.md) for scope and remaining limits.

## Install and analyze

Requires Python 3.12+, Git, and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/Sanoy24/the-brownfield-cartographer.git
cd the-brownfield-cartographer
uv sync --frozen

# Local analysis needs no network, API key, or model service.
uv run cartographer analyze /path/to/repository

# Force a fresh rebuild or choose an output root.
uv run cartographer analyze /path/to/repository --full -o ./results
```

An explicit GitHub URL input still downloads a checkout:

```sh
uv run cartographer analyze https://github.com/owner/repository
```

Local analysis never implicitly uploads source. Environment API keys do not
enable models in the CLI. Legacy model-dependent capabilities remain available
as optional Python code through `uv sync --extra llm`; they are outside the
deterministic default workflow.

## Outputs and queries

Artifacts are written under `<output-root>/.cartography/<repository-name>/`.
For a local checkout, the default output root is that checkout. GitHub URL
inputs use the current directory and an `owner--repository` folder name.

| Artifact | Purpose |
| --- | --- |
| `index.json` | Complete versioned graph, source locations, metadata, coverage |
| `module_graph.json` | Structural graph view, including functions |
| `lineage_graph.json` | Dataset/transformation graph view |
| `analysis_summary.json` | Complete/partial status, diagnostics and stage results |
| `CODEBASE.md` | Static repository context |
| `onboarding_brief.md` | Static metrics; inferred answers are omitted by default |
| `cartography_trace.jsonl` | Retained run history with run IDs |
| `state.json` | Last successful snapshot's content fingerprint |

Start the existing deterministic interactive query interface:

```sh
uv run cartographer query /path/to/repository
```

Or use individual commands with human-readable or JSON output:

```sh
uv run --no-sync cartographer overview /path/to/repository --json
uv run --no-sync cartographer find run --repo /path/to/repository --json
uv run --no-sync cartographer impact producer.py --repo /path/to/repository
uv run --no-sync cartographer lineage output --repo /path/to/repository --direction upstream --json
```

Use `--index-dir` for a custom output location. Python callers use
`brownfield_cartographer.query_api.QueryEngine.open(Path(repo))` and the same
query methods. Responses carry snapshot identity, coverage, capped-output
markers, and available source evidence. Ambiguous targets are rejected.

## Connect your AI assistant

```sh
uv sync --frozen --extra mcp
uv run --no-sync cartographer-mcp --repo /absolute/path/to/repository
```

The stdio server exposes `get_architecture`, `find_symbol`, `trace_dependencies`,
`analyze_impact`, `trace_data_lineage`, and `get_code_context`. It reads a
previously analyzed index inside a fixed repository scope and never calls a
model. See [MCP setup](docs/MCP.md) for Claude Code, Codex, and Cursor.

Commands: `find <text>`, `lineage <dataset>`, `lineage-down <dataset>`,
`blast <module-or-dataset>`, `explain <module>`, `summary`, and `quit`.
`find` is lexical matching. `blast` reports potential dependents, not a guarantee
that every consumer will break. Read the persisted coverage and diagnostics
before relying on results from an incomplete scan.

Analysis returns exit code 0 for complete/unchanged snapshots, 2 for partial
coverage, and 1 for an input or fatal run error. A partial run does not advance
the successful checkpoint. Changed working trees currently use full rebuilds;
unchanged successful snapshots can reuse the complete index.

## Support and safety

| Input | Current support |
| --- | --- |
| Python | Functions/classes, supported absolute/relative imports; conservative literal I/O associations |
| SQL | SELECT/UNION, CTAS and INSERT queries; statement spans and table dependencies |
| dbt SQL/YAML | Basic source `ref`/`source` parsing and configured model paths; manifest adapter planned |
| Airflow | Experimental, limited task dependency regex |
| JavaScript/TypeScript | File metadata only; unsupported structure explicitly listed |

The shared scanner respects nested `.gitignore` and `.cartographerignore`, prunes
build/environment directories, rejects symlinks/junctions, and caps file size,
total bytes, file count, entry count and traversal depth. Conventional secret
filenames are excluded; secrets embedded in normal source are not detected.
Large, binary, unreadable or unsupported files are reported rather than treated
as fully analyzed. No repository source code is executed by local analysis.

## Development and branching

```sh
uv sync --frozen
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
uv run --no-sync mypy -p brownfield_cartographer.scanner -p brownfield_cartographer.persistence -p brownfield_cartographer.query_api --follow-imports=skip
uv build
```

Use short-lived `feat/`, `fix/`, `docs/`, and `chore/` branches and pull requests
into stable `main`. Do not push feature work directly to `main`. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and release conventions.
