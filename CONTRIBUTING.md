# Contributing

Brownfield Cartographer provides local, deterministic repository analysis.
Contributions should prioritize correctness, source evidence, safe scanning,
and consistent behavior across the Python API, CLI, and MCP interfaces.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
Report security issues privately as described in [SECURITY.md](SECURITY.md),
not in public issues. Contributions are accepted under the project's
[Apache-2.0 license](LICENSE). Add user-visible changes to the `Unreleased`
section of [CHANGELOG.md](CHANGELOG.md).

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

A GitHub ruleset on `main` requires pull requests, passing CI jobs, resolved
review conversations, and squash merges, and blocks force pushes and deletion.

### Publishing a release

1. In a pull request, set `version` in `pyproject.toml`, run `uv lock`, and move
   the `Unreleased` changelog entries under a new `## [X.Y.Z] - YYYY-MM-DD`
   heading.
2. After it merges, tag the merge commit and push the tag:

   ```sh
   git switch main && git pull --ff-only origin main
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

The `Release` workflow checks that the tag matches the project version and that
the changelog has a matching section, runs the tests and installed-wheel checks,
publishes to PyPI, and creates a GitHub release with the changelog notes.

Publishing uses PyPI trusted publishing, so the repository holds no PyPI token.
A maintainer configures it once on PyPI with owner `Sanoy24`, repository
`the-brownfield-cartographer`, workflow `release.yml`, and environment `pypi`.
PyPI versions cannot be reused; fix a bad release with a new patch version.

## Development

```sh
uv sync --frozen
uv run pytest
uv run ruff check src tests scripts
uv run mypy -p brownfield_cartographer.scanner -p brownfield_cartographer.persistence -p brownfield_cartographer.query_api -p brownfield_cartographer.analyzers.javascript_analyzer -p brownfield_cartographer.analyzers.embedded_sql -p brownfield_cartographer.analyzers.sql_lineage --follow-imports=skip
uv build
uv run --no-sync python scripts/check_wheel.py
uv run --no-sync python scripts/check_wheel.py --mcp
```

Cartographer never calls a model and needs no credentials. Source fixtures should be small,
deterministic, and runnable without network access or credentials. Test both
extraction and queries, including evidence and incomplete-analysis behavior.

The wheel check installs the built distribution and its core dependencies into
a fresh temporary environment, then exercises the API and installed CLI from
outside the source tree. It checks that optional model/MCP packages are absent,
that local analysis works with network calls blocked in the API probe, and that
missing MCP dependencies produce an actionable error. Installation may download
dependencies; add `--offline` when the required packages are already cached.
The `--mcp` check first validates the core without extras, then installs the MCP
extra into the same isolated environment. It exercises all six tools through
the installed stdio entry point, compares structured results with the Python
API, checks rejected requests and status refresh, and verifies no tool writes.
