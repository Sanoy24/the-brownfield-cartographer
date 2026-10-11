# Contributing

Brownfield Cartographer provides local, deterministic repository analysis.
Contributions should prioritize correctness, source evidence, safe scanning,
and consistent behavior across the Python API, CLI, and MCP interfaces.

## Branching and releases

Use `main` as the stable integration branch. Work in short-lived branches based
on the latest `main` and merge through a reviewed pull request after checks pass.
Do not push feature work directly to `main`.

Branch names:

- `feat/<short-description>` for capabilities.
- `fix/<short-description>` for defects.
- `docs/<short-description>` for documentation.
- `chore/<short-description>` for tooling and maintenance.

For example:

```sh
git switch main
git pull --ff-only origin main
git switch -c feat/local-first-foundation
# Make and validate a small change, then commit it.
git push -u origin feat/local-first-foundation
```

Open a pull request targeting `main`. State the problem, behavior change,
validation, and any remaining limitations. Prefer squash merging, then delete
the merged feature branch. Use commit/PR titles such as `feat: ...`, `fix: ...`,
`docs: ...`, or `chore: ...`. Avoid long-lived `develop` branches. Release from
reviewed `main` commits using version tags such as `v0.3.0`; prereleases may use
`v0.3.0rc1`. Creating a branch does not authorize an automatic commit or push.

Repository administrators should configure a GitHub ruleset for `main` to
require pull requests, passing CI, resolved review conversations, and to block
force pushes and deletion. This document defines the convention; it does not
configure remote protection or guarantee enforcement.

## Development

```sh
uv sync --frozen
uv run pytest
uv run ruff check src tests
uv run mypy -p brownfield_cartographer.scanner -p brownfield_cartographer.persistence -p brownfield_cartographer.query_api --follow-imports=skip
```

The default installation runs without model adapters. Legacy LLM enrichment
dependencies are available through `uv sync --extra llm`; installing them does
not opt the core into making model calls. Source fixtures should be small,
deterministic, and runnable without network access or credentials. Test both
extraction and queries, including evidence and incomplete-analysis behavior.
