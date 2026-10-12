# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, minor
versions may contain breaking changes.

## [Unreleased]

### Added

- Accuracy benchmark: pinned dbt (jaffle_shop), Python import and SQL lineage
  corpora with hand-written expected results, scored by `scripts/benchmark.py`
  and checked in CI.
- JavaScript and TypeScript structure (`.js .mjs .cjs .jsx .ts .mts .cts .tsx`):
  functions, arrow functions and class methods with line spans; ESM, type-only,
  `require()` and `import()` dependencies resolved for relative specifiers.
  Syntax errors are reported as diagnostics and make coverage partial.

### Changed

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

[Unreleased]: https://github.com/Sanoy24/the-brownfield-cartographer/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Sanoy24/the-brownfield-cartographer/releases/tag/v0.3.0
[0.2.0]: https://github.com/Sanoy24/the-brownfield-cartographer/tree/6079f70
