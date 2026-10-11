# Connect your coding assistant

Brownfield Cartographer exposes six read-only local tools over MCP stdio. The
server returns repository facts and source excerpts; your existing host model
explains them. Cartographer needs no model credential or running model service.

Install and index a local checkout first:

```sh
uv sync --frozen --extra mcp
uv run cartographer analyze /absolute/path/to/target
```

For host configurations, use the absolute Python interpreter in Cartographer's
virtual environment (`/absolute/path/to/cartographer/.venv/bin/python` on Unix,
`C:/path/to/cartographer/.venv/Scripts/python.exe` on Windows). Launch arguments
are `-m brownfield_cartographer.mcp_server --repo /absolute/path/to/target`.
Using that interpreter avoids package downloads or environment resolution when
the host starts the server.

With a custom analysis output root, append `--index-dir` followed by the exact
`<output-root>/.cartography/<repository-name>` directory. The server validates
that the index belongs to the configured repository. Tool arguments cannot
select a different repository. Re-run `analyze` after edits; a running server
observes newly published indexes. Source excerpts reject stale content or
changed ignore rules. Queries otherwise describe their indexed snapshot.

## Claude Code

From your target project, register the server using your absolute interpreter
and repository paths:

```sh
claude mcp add --transport stdio --scope project brownfield -- /absolute/path/to/cartographer/.venv/bin/python -m brownfield_cartographer.mcp_server --repo /absolute/path/to/target
```

Approve the project server in Claude Code and use `/mcp` to inspect its status.
This command is supplied for you to run; Cartographer does not modify host
settings during analysis. See the [official Claude Code MCP guide](https://code.claude.com/docs/en/mcp).

## Codex

Register a local server with the same interpreter:

```sh
codex mcp add brownfield -- /absolute/path/to/cartographer/.venv/bin/python -m brownfield_cartographer.mcp_server --repo /absolute/path/to/target
```

Alternatively, add this to the appropriate Codex configuration:

```toml
[mcp_servers.brownfield]
command = "/absolute/path/to/cartographer/.venv/bin/python"
args = ["-m", "brownfield_cartographer.mcp_server", "--repo", "/absolute/path/to/target"]
startup_timeout_sec = 20
tool_timeout_sec = 60
```

Use `codex mcp list` to inspect registration. These fields and the CLI command
are documented in the [official OpenAI MCP documentation](https://developers.openai.com/codex/mcp/).

## Cursor

Add a server entry to your target project's `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "brownfield": {
      "type": "stdio",
      "command": "/absolute/path/to/cartographer/.venv/bin/python",
      "args": ["-m", "brownfield_cartographer.mcp_server", "--repo", "/absolute/path/to/target"]
    }
  }
}
```

Use forward slashes in Windows JSON paths, or escape each backslash. See the
[official Cursor MCP documentation](https://cursor.com/docs/mcp) for configuration
locations and server status controls.

## Available tools

| Tool | Input | Result |
| --- | --- | --- |
| `get_architecture` | Optional result limit | Counts, module hubs, sources/sinks, coverage |
| `find_symbol` | Literal search pattern | Matching modules/functions with available source locations |
| `trace_dependencies` | Qualified target, direction | Imports/calls toward dependencies or reverse dependents |
| `analyze_impact` | Qualified target | Potential affected consumers; no guarantee of runtime failure |
| `trace_data_lineage` | Dataset, upstream/downstream | Supported producers or consumers |
| `get_code_context` | Relative path, line range | Snapshot-checked source excerpt inside the repository |

Responses include `schema_version`, `snapshot_id`, `status`, `coverage`,
`truncated`, and `data`. Available evidence carries source paths, spans,
extraction methods, and confidence labels. Missing locations remain unknown.
Maximum output is 64 KB; result counts are capped at 200 and traversal depth at
20. Code excerpts contain at most 200 lines/32 KB. Ambiguous names and invalid
arguments produce tool errors, rather than choosing an arbitrary match.

The server uses the maintained v1 SDK line with an explicit upper bound; see
the [official MCP SDK documentation](https://py.sdk.modelcontextprotocol.io/v1/).
The automated test launches a real stdio server, discovers tools, reads source,
checks structured results, rejects an out-of-scope path, and verifies that the
repository/index files remain unchanged. This verifies protocol integration;
an interactive session in Claude Code, Codex, or Cursor has not been exercised
by that test. No host settings are installed automatically.
