# Brownfield Cartographer

[![CI](https://github.com/Sanoy24/the-brownfield-cartographer/actions/workflows/ci.yml/badge.svg)](https://github.com/Sanoy24/the-brownfield-cartographer/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

A local-first repository analysis engine for evidence-backed code structure,
dependency impact, and data lineage. The Python API, deterministic CLI, and
read-only local MCP server share one query engine; reasoning belongs to the
developer's existing AI assistant.

The CLI includes `analyze`, `overview`, `find`, `impact`, and `lineage`.
The support table below describes current capabilities and limitations.

## Install and analyze

Requires Python 3.12+ and Git. Install the command-line tools from PyPI with
[uv](https://docs.astral.sh/uv/) or [pipx](https://pipx.pypa.io/):

```sh
uv tool install "brownfield-cartographer[mcp]"
# or: pipx install "brownfield-cartographer[mcp]"
cartographer --version

# Local analysis needs no network, API key, or model service.
cartographer analyze /path/to/repository

# Force a fresh rebuild or choose an output root.
cartographer analyze /path/to/repository --full -o ./results

# Optional: use a previously generated dbt v12 manifest.
cartographer analyze /path/to/dbt-project --dbt-manifest target/manifest.json
```

Omit `[mcp]` if you only need the CLI. To run from a source checkout instead,
clone the repository, run `uv sync --frozen`, and prefix commands with `uv run`,
as in the examples below.

An explicit GitHub URL input still downloads a checkout:

```sh
cartographer analyze https://github.com/owner/repository
```

Local analysis never implicitly uploads source. Environment API keys do not
enable models in the CLI. Legacy model-dependent capabilities remain available
as optional Python code through `uv sync --extra llm`; they are outside the
deterministic default workflow.

## Outputs and queries

For a local checkout, artifacts default to
`<repository>/.cartography/<repository-name>/`. With an explicit `--output`,
local repositories use `<output-root>/.cartography/<repository-name>--<path-hash>/`
so same-named checkouts do not overwrite one another. The hash identifies the
resolved checkout path; moving a checkout creates a separate index. The JSON
analysis result includes `artifact_directory`, which can be passed to query
commands as `--index-dir`. Existing custom-output indexes are left untouched;
reanalyzing writes to the new directory. GitHub URL inputs use the current
directory by default and an `owner--repository` folder name.

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
cartographer analyze /absolute/path/to/repository
cartographer-mcp --repo /absolute/path/to/repository
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
| dbt | Explicit manifest v12 ingestion for models, seeds, snapshots and sources; basic SQL/YAML fallback without a manifest |
| Airflow | Experimental, limited task dependency regex |
| JavaScript/TypeScript | `.js .mjs .cjs .jsx .ts .mts .cts .tsx`: functions, arrow functions, class methods; ESM, type-only, `require()` and `import()` imports with literal relative specifiers. Package imports are external; tsconfig path aliases are not resolved |

The shared scanner respects nested `.gitignore` and `.cartographerignore`, prunes
build/environment directories, rejects symlinks/junctions, and caps file size,
total bytes, file count, entry count and traversal depth. Conventional secret
filenames are excluded; secrets embedded in normal source are not detected.
Large, binary, unreadable or unsupported files are reported rather than treated
as fully analyzed. No repository source code is executed by local analysis.

See [dbt artifact ingestion](docs/DBT.md) for supported manifest versions,
dataset identifiers, artifact freshness, and scanner limits.

The [accuracy benchmark](benchmarks/README.md) scores extraction against
hand-written expected results for pinned dbt, Python and SQL corpora, including
known gaps. It runs in CI.

## Development and branching

```sh
uv sync --frozen
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts
uv run --no-sync mypy -p brownfield_cartographer.scanner -p brownfield_cartographer.persistence -p brownfield_cartographer.query_api -p brownfield_cartographer.analyzers.javascript_analyzer --follow-imports=skip
uv build
uv run --no-sync python scripts/check_wheel.py
```

Use short-lived `feat/`, `fix/`, `docs/`, and `chore/` branches and pull requests
into stable `main`. Do not push feature work directly to `main`. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and release conventions,
and [CHANGELOG.md](CHANGELOG.md) for release history.

## Community and license

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). Report
vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

Licensed under the [Apache License 2.0](LICENSE).
