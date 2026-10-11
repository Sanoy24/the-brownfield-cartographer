# Small local fixture

`sample_repo` demonstrates import impact and table lineage without a running
database, model, or network connection. Analysis reads the source; it does not
execute it. These examples are fixtures, not an accuracy benchmark.

From the Cartographer checkout:

```sh
uv run --no-sync cartographer analyze examples/sample_repo --json
uv run --no-sync cartographer overview examples/sample_repo --json
uv run --no-sync cartographer find normalize_name --repo examples/sample_repo --json
uv run --no-sync cartographer impact producer.py --repo examples/sample_repo --json
uv run --no-sync cartographer lineage analytics.customer_totals --repo examples/sample_repo --json
```

Expected import impact: `producer.py` has `consumer.py` as a reverse dependent.
Expected upstream table lineage: `analytics.customer_totals` consumes
`analytics.orders`, which consumes `raw.orders`, with transformation spans.

Install the MCP extra, analyze the fixture, then configure the stdio server
with `--repo` set to the fixture's absolute directory. See [MCP setup](../docs/MCP.md).
