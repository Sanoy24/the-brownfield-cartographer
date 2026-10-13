# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, minor
versions may contain breaking changes.

## [Unreleased]

### Added

- Lineage from SQL embedded in Python: `execute`, `executemany`,
  `executescript`, `.sql()` (Spark, DuckDB), `read_sql`/`read_sql_query` and
  `read_sql_table`, including SQLAlchemy `text()` and DB-API placeholders.
  Queries count as reads of the surrounding function; `INSERT`/`CREATE ... AS`
  become exact statement-level transformations. SQL is resolved from literals
  and same-module constants; dynamic SQL is reported as a diagnostic.
- Spark `DataFrameWriter` path writes such as `.write.parquet(path)`.
- `python_etl` benchmark corpus.
- Table lineage for `UPDATE`, `DELETE` and `MERGE`: the written table depends
  on every other table the statement reads.

### Changed

- An unsupported or invalid SQL statement is reported on its own; the other
  statements in the same file or script are still extracted.
- Statements without table lineage (`DROP`, `TRUNCATE`, `ALTER`, transactions,
  `SET`, `GRANT`, `CREATE INDEX`/`SCHEMA`) no longer make coverage partial.

- The interactive `cartographer query` now runs on the shared query engine used
  by the other commands and the MCP server. It takes `REPO` and `--index-dir`
  like the other commands, and supports `find`, `impact`, `deps`, `dependents`,
  `lineage`, `lineage-down`, `context` and `overview`.
- `onboarding_brief.md` lists key modules, data entry and exit points, and
  change hotspots instead of five placeholder questions.
- `analysis_summary.json` no longer contains a `semanticist` section.
- The README describes the project's focus: dependencies that cross language
  boundaries, starting with data lineage.

### Removed

- The `llm` extra and all model-based code: purpose statements, documentation
  drift, domain clustering, day-one answers, and the LangGraph chat agent.
  `Orchestrator(enable_llm=...)` is no longer accepted.
- The duplicate `navigator` query implementation, and the Documentation Drift
  and Module Purpose Index sections of `CODEBASE.md`.
- `.env.example`, since nothing reads environment configuration.

## [0.4.0] - 2026-10-12

Adds JavaScript/TypeScript structure and an accuracy benchmark. The `mcp` extra
now requires MCP Python SDK 2.x.

### Added

- Accuracy benchmark: pinned dbt (jaffle_shop), Python import and SQL lineage
  corpora with hand-written expected results, scored by `scripts/benchmark.py`
  and checked in CI.
- JavaScript and TypeScript structure (`.js .mjs .cjs .jsx .ts .mts .cts .tsx`):
  functions, arrow functions and class methods with line spans; ESM, type-only,
  `require()` and `import()` dependencies resolved for relative specifiers.
  Syntax errors are reported as diagnostics and make coverage partial.

### Changed

- The MCP server uses MCP Python SDK 2.x; the `mcp` extra now requires
  `mcp>=2.3,<3`. Tool names, arguments, and results are unchanged. Tool calls
  are serialized because the SDK runs handlers on worker threads.
- Dead-code candidates for JavaScript/TypeScript modules use resolved imports.

## [0.3.0] - 2026-10-12

First release published to PyPI as `brownfield-cartographer`.

### Added

- `cartographer --version`.
- Tag-triggered release workflow publishing to PyPI through trusted publishing.
- Apache-2.0 license, code of conduct, security policy, issue and pull request
  templates, and Dependabot configuration.
- Package metadata: license, project URLs, keywords, and classifiers.

### Changed

- The distribution is renamed from `the-brownfield-cartographer` to
  `brownfield-cartographer`, matching the `brownfield_cartographer` import name.
- The CLI no longer loads a `.env` file. Core analysis never reads API keys.
- `.env.example` now documents only the optional legacy `llm` extra.
- Ruff checks pyflakes, pycodestyle, import order, bugbear, and pyupgrade rules.

### Removed

- Unused `gitpython` and `python-dotenv` core dependencies.
- The top-level `src` compatibility package and `src/cli.py` shim. Use the
  `cartographer` entry point or `python -m brownfield_cartographer.cli`.
- Committed `.cartography/` output samples, which contained machine-specific
  paths, the standalone `visualize.html` demo, and `RECONNAISSANCE.md`.

## [0.2.0] - untagged

### Added

- Bounded, ignore-aware repository scanner shared by all analysis stages.
- Versioned `index.json` with source spans, coverage, and diagnostics.
- Deterministic `QueryEngine` and `overview`, `find`, `impact`, and `lineage`
  CLI commands with JSON output.
- Read-only MCP stdio server (`cartographer-mcp`) exposing six query tools.
- dbt manifest v12 ingestion with artifact provenance.
- CI across Linux and Windows on Python 3.12 and 3.13, with installed-wheel checks.

### Fixed

- Import impact now returns reverse dependents; lineage traversal follows only
  lineage relationships.
- Distinct transformation IDs and statement-specific SQL sources and targets.
- Failed or partial runs no longer advance the successful checkpoint.

[Unreleased]: https://github.com/Sanoy24/the-brownfield-cartographer/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Sanoy24/the-brownfield-cartographer/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Sanoy24/the-brownfield-cartographer/releases/tag/v0.3.0
[0.2.0]: https://github.com/Sanoy24/the-brownfield-cartographer/tree/6079f70
