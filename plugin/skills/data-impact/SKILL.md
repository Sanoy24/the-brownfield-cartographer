---
name: data-impact
description: Answer "what breaks if I change this?" and "where does this data come from?" with the Brownfield Cartographer MCP tools. Use for impact of changing a file, function, SQL/dbt model, table or column; upstream/downstream table or column lineage; which Airflow tasks run before or after a task; or onboarding questions about how data flows through a repository with SQL, dbt, Airflow, Python or TypeScript.
---

# Data impact with Brownfield Cartographer

The `cartographer` MCP server answers from a local, static index of this
repository. Every result carries file and line evidence and a coverage status.
Prefer it over reading many files when the question is about dependencies,
lineage or impact.

## 1. Make sure the repository is indexed

If a tool returns `No index found for this repository; run cartographer analyze ...`,
or the user just changed code and wants fresh answers, build the index with Bash
from the project root (it runs offline and never executes repository code):

```sh
uvx --from brownfield-cartographer cartographer analyze "${CLAUDE_PROJECT_DIR}"
```

Add `--dbt-manifest target/manifest.json` when that file exists (column lineage
for dbt models needs a manifest built with `dbt compile`), and `--dialect <name>`
(for example `bigquery`) only if the user names their SQL engine. Then retry the
tool call; the server picks up the new index automatically.

## 2. Pick the tool for the question

| Question | Tool and arguments |
| --- | --- |
| What does changing this file, function, table, column or task affect? | `analyze_impact` with the path, `module.py::function`, table, `table::column`, or task id |
| Where does this table come from / what consumes it? | `trace_data_lineage` with the table, `direction` `upstream` or `downstream` |
| Which source columns feed this column / what is derived from it? | `trace_data_lineage` with `table::column` |
| Which Airflow tasks run before or after this task? | `find_symbol` to get the task id (`task:<file>::<dag>.<task>`), then `trace_dependencies` with `direction` `dependencies` (before) or `dependents` (after) |
| What imports or calls this module or function? | `trace_dependencies` with `direction` `dependents` |
| Where is this defined? Exact name of a table, column or task? | `find_symbol` with part of the name |
| What is in this file at these lines? | `get_code_context` with a repository-relative path and line range |
| What is in this repository? | `get_architecture` |

Use `depth` to widen or narrow traversals; `depth: 1` gives direct neighbors.
If a result says `stopped_by: depth`, more results exist further away.

## 3. Report honestly

- Cite the `evidence` path and line range for each dependency you mention.
- Results show *potential* impact from static analysis, not proof that something
  will fail at runtime.
- If `status` is `partial`, or `coverage.diagnostics` lists unresolved or dynamic
  items relevant to the question, say what is not covered instead of implying
  completeness.
- If a tool error names an ambiguous target, retry with one of the qualified names
  it lists rather than guessing.
