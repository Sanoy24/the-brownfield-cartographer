# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, minor
versions may contain breaking changes.

## [Unreleased]

### Added

- tsconfig/jsconfig path aliases for JavaScript/TypeScript imports: `paths`
  wildcards and exact patterns, `baseUrl`, and `extends` chains within the
  repository, using the nearest config above each importer. Config files are
  parsed as JSON with comments and trailing commas, are part of the scanned
  inventory, and invalidate the cache when edited.
- SQL string constants imported from other repository modules
  (`from etl.queries import ORDERS`, relative imports, `import etl.queries as q`
  with `q.ORDERS`) are resolved for embedded SQL lineage. Constants assigned
  more than once remain unresolved.

### Fixed

- MCP tool errors reached assistants only as "Error executing tool <name>"
  since the SDK 2.x migration; the reason (for example `Target not found` or an
  ambiguous-target hint) is returned again so the assistant can correct the call.
- `trace_dependencies` accepts Airflow task ids and follows task order
  (`dependencies` run before the task, `dependents` after). A live Claude Code
  session previously fell back to reading DAG code and missed a downstream task.
- MCP tool descriptions mention Airflow task ids and `dataset::column` lineage,
  which a live session did not discover before.

### Added

- Column lineage from a dbt manifest's compiled SQL. With `--dbt-manifest`,
  each model's `compiled_code` (macros and refs already expanded) is traced;
  tables are mapped back to dbt unique IDs through `relation_name`, and the
  manifest's `adapter_type` selects the SQL dialect. Models without compiled
  SQL are counted in a diagnostic suggesting `dbt compile`.

## [0.6.0] - 2026-10-13

Adds column-level lineage, Airflow task modeling and per-file SQL dialect
selection, finds dbt projects anywhere in a repository, and removes internal
transformation nodes from query results. Validated on four public data
repositories at pinned commits.

### Added

- Column-level lineage. Columns are `dataset::column` nodes linked by
  `DERIVES` edges marked `copy` or `derived`, with statement evidence. Built for
  `CREATE ... AS` and `INSERT ... SELECT` (including column lists) in SQL
  files, SQL embedded in Python, Airflow inline SQL, and dbt models whose only
  Jinja is `ref`/`source`/`config`/comments. `SELECT *` over CTEs and subqueries
  is expanded, and a single-source query's unqualified columns are attributed
  to that source. `SELECT *` over tables, unqualified columns in joins, and
  other templated models are reported as unresolved. Query with
  `cartographer lineage <dataset>::<column>` or the existing MCP tools; impact
  follows derived columns. The summary reports `column_lineage` counts.

- Airflow DAG structure. Tasks are graph nodes (`task:<file>::<dag>.<task>`),
  not datasets, defined from operators and `@task` functions in `with DAG`,
  `@dag` and `dag=` DAGs. `SCHEDULES` edges record order from `>>`/`<<`, lists,
  `set_downstream`/`set_upstream`, `chain()`, `cross_downstream()` and TaskFlow
  data passing, each with its own line. `RUNS` edges link a task to its
  `python_callable`, `@task` function, `sql=` file or inline SQL lineage, and
  `python x.py` bash commands. Impact flows from changed code through the tasks
  that run it to downstream tasks; `find_symbol` returns tasks; the
  architecture overview counts them. DAG files are detected by importing
  `airflow`, anywhere in the repository.
- `airflow_dags` benchmark corpus with `tasks`, `schedules` and `runs` facts.

- SQL dialect selection. Each SQL file or embedded SQL string is parsed with
  the candidate dialect that yields the fewest invalid or unsupported
  statements (DuckDB, BigQuery, Snowflake, Postgres, MySQL, Spark, T-SQL,
  Trino). The dialect used is recorded in evidence as `sql:<dialect>`; it names
  the grammar that parsed the SQL, not the database engine.
  `cartographer analyze --dialect <name>` pins one dialect.

### Changed

- Lineage and impact results no longer list internal transformation nodes
  (`tx:...`). They are followed transparently and do not count toward
  `depth`, so `--depth 1` returns the directly connected datasets; evidence
  still cites the transformation's file and lines. Results report
  `stopped_by: depth|limit` when more results exist, and the CLI says which.
- Commands without table lineage that sqlglot cannot model structurally
  (`GRANT`, `SET`, `CREATE CATALOG`/`DATABASE`/`SCHEMA`, and similar) no longer
  make coverage partial.
- sqlglot's "Falling back to parsing as a 'Command'" warnings are no longer
  printed.
- A query in a multi-statement SQL file is a read only; it no longer creates a
  write to a dataset named after the file. A file that is a single query still
  defines a dataset named after the file, now labeled `confidence: inferred`
  with method `sql:<dialect>+file_name`.

### Removed

- The regex Airflow parser, which stored task order as datasets with
  whole-file evidence and only read files under `dags/`.

### Fixed

- dbt projects in subfolders were not detected, so their models were parsed as
  plain SQL and lost all lineage. Every `dbt_project.yml` is now a project
  root, evidence paths are relative to the repository, and the analysis
  summary lists `dbt_projects`.
- Macros, tests and analyses inside a dbt project were parsed as plain SQL,
  making coverage falsely partial. No `.sql` file inside a dbt project is
  parsed as plain SQL.

## [0.5.0] - 2026-10-13

Focuses the project on dependencies that cross language boundaries. Adds
lineage from SQL embedded in Python and from `UPDATE`, `DELETE` and `MERGE`,
and removes all model-based code. Breaking: the `llm` extra,
`Orchestrator(enable_llm=...)` and artifact-directory paths for
`cartographer query` are gone.

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

[Unreleased]: https://github.com/Sanoy24/the-brownfield-cartographer/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/Sanoy24/the-brownfield-cartographer/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Sanoy24/the-brownfield-cartographer/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Sanoy24/the-brownfield-cartographer/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Sanoy24/the-brownfield-cartographer/releases/tag/v0.3.0
[0.2.0]: https://github.com/Sanoy24/the-brownfield-cartographer/tree/6079f70
