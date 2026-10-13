# Brownfield Cartographer

[![CI](https://github.com/Sanoy24/the-brownfield-cartographer/actions/workflows/ci.yml/badge.svg)](https://github.com/Sanoy24/the-brownfield-cartographer/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

**What breaks if I change this?** Cartographer finds the dependencies your
language server can't see: a table written by a Python job, transformed in SQL,
consumed by a dbt model. It maps them statically and locally, with file and line
evidence, for you and your coding agent.

```sh
cartographer analyze .
cartographer impact models/staging/stg_orders.sql   # what is downstream of this file?
cartographer lineage customers --direction upstream # where does this table come from?
```

Language servers and grep answer questions about code symbols. Cartographer
answers questions that cross language boundaries through names in strings and
configuration, starting with data lineage across Python, SQL and dbt. Every
result carries source evidence and coverage, and unsupported cases are reported
rather than guessed. It runs offline, needs no API key, and never executes
repository code.

One query engine serves the CLI, a Python API, and a read-only MCP server for
Claude Code, Codex and Cursor. The support table below lists current
capabilities and known gaps.

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

Local analysis never uploads source and never calls a model.

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
| `onboarding_brief.md` | Key modules, data entry/exit points and change hotspots |
| `cartography_trace.jsonl` | Retained run history with run IDs |
| `state.json` | Last successful snapshot's content fingerprint |

Explore interactively with `cartographer query /path/to/repository`. It accepts
`find`, `impact`, `deps`, `dependents`, `lineage`, `lineage-down`, `context`,
`overview`, `help` and `quit`, using the same engine as the commands below.

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

`find` is lexical matching. `impact` reports potential dependents, not a
guarantee that every consumer will break. Read the persisted coverage and
diagnostics before relying on results from an incomplete scan.

Analysis returns exit code 0 for complete/unchanged snapshots, 2 for partial
coverage, and 1 for an input or fatal run error. A partial run does not advance
the successful checkpoint. Changed working trees currently use full rebuilds;
unchanged successful snapshots can reuse the complete index.

## Support and safety

| Input | Current support |
| --- | --- |
| Python | Functions/classes, supported absolute/relative imports; conservative literal I/O associations (pandas, Spark readers and writers). SQL passed to `execute`/`executemany`/`executescript`, `.sql()` and `read_sql` is parsed from literals and same-module constants, including `text()` and DB-API placeholders. Dynamic SQL is reported, not guessed; constants imported from other modules are not resolved |
| SQL | SELECT/UNION, CTAS, INSERT, UPDATE, DELETE and MERGE; statement spans and table dependencies. DDL and transaction statements without table lineage are skipped; other statements, such as COPY, are reported individually without dropping the rest of the file |
| dbt | Explicit manifest v12 ingestion for models, seeds, snapshots and sources; basic SQL/YAML fallback without a manifest. Projects are found anywhere in the repository, including several per repository |
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
uv run --no-sync mypy -p brownfield_cartographer.scanner -p brownfield_cartographer.persistence -p brownfield_cartographer.query_api -p brownfield_cartographer.analyzers.javascript_analyzer -p brownfield_cartographer.analyzers.embedded_sql -p brownfield_cartographer.analyzers.sql_lineage --follow-imports=skip
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
