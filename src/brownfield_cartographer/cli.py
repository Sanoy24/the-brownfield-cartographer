"""
CLI entry point for The Brownfield Cartographer.

Provides the `cartographer` commands: analyze, overview, find, impact,
lineage, and an interactive query loop, all backed by QueryEngine.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from brownfield_cartographer.orchestrator import Orchestrator
from brownfield_cartographer.query_api import QueryEngine
from brownfield_cartographer.scanner import ScanSettings

console = Console()

# Main app with a default callback that does nothing — this forces typer
# to treat decorated functions as proper subcommands instead of the root.
app = typer.Typer(
    name="cartographer",
    help="Brownfield Cartographer - local codebase intelligence",
    add_completion=False,
)


def _setup_logging(verbose: bool = False) -> None:
    """Configure structured logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                show_time=True,
                show_path=False,
                markup=True,
            )
        ],
        force=True,
    )


def _print_version(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(f"cartographer {version('brownfield-cartographer')}")
        raise typer.Exit()


@app.callback()
def callback(
    _version: bool = typer.Option(False, "--version", callback=_print_version, is_eager=True,
                                  help="Show the installed version and exit."),
) -> None:
    """Brownfield Cartographer - local codebase intelligence."""


@app.command()
def analyze(
    repo: str = typer.Argument(
        help="Local path or GitHub URL (e.g. https://github.com/dbt-labs/jaffle_shop)"
    ),
    output: str | None = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for .cartography/ artifacts (defaults to cwd for URLs, repo root for local)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable debug logging",
    ),
    full: bool = typer.Option(False, "--full", help="Rebuild even when the snapshot is unchanged"),
    as_json: bool = typer.Option(False, "--json", help="Return structured run results on stdout"),
    dialect: str | None = typer.Option(
        None, "--dialect", help="SQL dialect for all SQL (e.g. bigquery, snowflake, postgres); default: best fit per file",
    ),
    dbt_manifest: str | None = typer.Option(
        None, "--dbt-manifest", help="Repository-relative dbt v12 JSON artifact; explicitly overrides ignore rules for this file",
    ),
) -> None:
    """
    Analyze a codebase and generate cartography artifacts.

    Runs static structure, supported lineage extraction, and artifact generation.
    Local repositories need no network or model credentials. Changed snapshots
    use a conservative full rebuild. Partial coverage returns exit code 2.
    """
    _setup_logging(verbose)

    output_dir = Path(output) if output else None
    orchestrator = None
    try:
        orchestrator = Orchestrator(scan_settings=ScanSettings(dbt_manifest=dbt_manifest, sql_dialect=dialect))
        if as_json:
            with redirect_stdout(sys.stderr):
                results = orchestrator.run(repo, output_dir=output_dir, full=full)
            typer.echo(json.dumps(results, indent=2))
        else:
            results = orchestrator.run(repo, output_dir=output_dir, full=full)
        if results.get("status") in {"partial", "failed"}:
            raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except (RuntimeError, ValueError, OSError) as exc:
        if as_json:
            typer.echo(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from None
    finally:
        if orchestrator is not None:
            orchestrator.cleanup()


@app.command()
def query(repo: Path = typer.Argument(Path("."), help="Previously analyzed repository"),
          index_dir: Path | None = typer.Option(None, "--index-dir")) -> None:
    """
    Interactively query a previously analyzed repository.

    Uses the same query engine as the other commands and the MCP server.
    Type `help` for commands and `quit` to exit.
    """
    engine = _query_engine(repo, index_dir)
    if engine.run_status in {"partial", "failed"}:
        console.print(f"Warning: the index status is {engine.run_status}; check coverage in results.", markup=False)
    console.print(_REPL_HELP, markup=False)
    while True:
        try:
            line = input("cartographer> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not line:
            continue
        command, _, argument = line.partition(" ")
        argument = argument.strip()
        if command in {"quit", "exit"}:
            return
        if command == "help":
            console.print(_REPL_HELP, markup=False)
            continue
        if command not in _REPL_COMMANDS:
            console.print(f"Unknown command: {command}. Type help for commands.", markup=False)
            continue
        method, usage, build = _REPL_COMMANDS[command]
        if usage and not argument:
            console.print(f"Usage: {command} {usage}", markup=False)
            continue
        try:
            _show_result(getattr(engine, method)(**build(argument)), as_json=False)
        except (ValueError, OSError) as exc:
            console.print(f"Error: {exc}", markup=False)


def _context_arguments(argument: str) -> dict:
    path, *lines = argument.split()
    numbers = [int(value) for value in lines[:2]]
    return {"path": path, **dict(zip(("start_line", "end_line"), numbers, strict=False))}


_REPL_COMMANDS = {
    "find": ("find_symbol", "<text>", lambda a: {"pattern": a}),
    "impact": ("analyze_impact", "<target>", lambda a: {"target": a}),
    "blast": ("analyze_impact", "<target>", lambda a: {"target": a}),
    "deps": ("trace_dependencies", "<module-or-function>", lambda a: {"target": a}),
    "dependents": ("trace_dependencies", "<module-or-function>", lambda a: {"target": a, "direction": "dependents"}),
    "lineage": ("trace_data_lineage", "<dataset>", lambda a: {"dataset": a}),
    "lineage-down": ("trace_data_lineage", "<dataset>", lambda a: {"dataset": a, "direction": "downstream"}),
    "context": ("get_code_context", "<path> [start] [end]", _context_arguments),
    "overview": ("get_architecture", "", lambda a: {}),
    "summary": ("get_architecture", "", lambda a: {}),
}
_REPL_HELP = (
    "Commands: find <text> | impact <target> | deps <target> | dependents <target> | "
    "lineage <dataset> | lineage-down <dataset> | context <path> [start] [end] | overview | help | quit"
)


def main() -> None:
    """Entry point for `python -m brownfield_cartographer.cli`."""
    app()


def _query_engine(repo: Path, index_dir: Path | None, as_json: bool = False) -> QueryEngine:
    try:
        return QueryEngine.open(repo, index_dir)
    except (ValueError, OSError) as exc:
        if as_json:
            typer.echo(json.dumps({"error": str(exc)}))
        else:
            console.print(f"Error: {exc}", markup=False)
        raise typer.Exit(code=1) from exc


def _show_result(result: dict, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(result, indent=2))
        return
    console.print(f"{result['query']} | {result['status']} | snapshot {result['snapshot_id'][:12]}", markup=False)
    if result["truncated"]:
        reason = result["data"].get("stopped_by")
        message = ("More results exist beyond this depth; increase --depth to see them." if reason == "depth"
                   else "Results were capped; narrow the query or increase --limit.")
        console.print(message, markup=False)
    data = result["data"]
    rows = data.get("results", data.get("matches", data.get("modules")))
    if rows is not None:
        table = Table("Symbol / node", "Relationship", "Evidence")
        for row in rows:
            evidence = row.get("evidence", {})
            location = evidence.get("path") or "unknown"
            if evidence.get("line_range"):
                location += ":" + "-".join(map(str, evidence["line_range"]))
            table.add_row(str(row.get("node", row.get("id", row.get("path", "")))),
                          str(row.get("edge_type", row.get("type", row.get("language", "")))), location)
        console.print(table)
        if "counts" in data:
            console.print(json.dumps(data["counts"], indent=2), markup=False)
    else:
        console.print(json.dumps(data, indent=2), markup=False)


def _run_query(engine: QueryEngine, method: str, as_json: bool, **arguments) -> None:
    try:
        _show_result(getattr(engine, method)(**arguments), as_json)
    except (ValueError, OSError) as exc:
        if as_json:
            typer.echo(json.dumps({"error": str(exc)}))
        else:
            console.print(f"Error: {exc}", markup=False)
        raise typer.Exit(code=1) from exc


@app.command()
def overview(repo: Path = typer.Argument(Path(".")),
             index_dir: Path | None = typer.Option(None, "--index-dir"),
             limit: int = typer.Option(50, min=1, max=200),
             as_json: bool = typer.Option(False, "--json")) -> None:
    """Show indexed architecture, dataset endpoints and coverage."""
    _run_query(_query_engine(repo, index_dir, as_json), "get_architecture", as_json, limit=limit)


@app.command("find")
def find_command(pattern: str, repo: Path = typer.Option(Path("."), "--repo"),
                 index_dir: Path | None = typer.Option(None, "--index-dir"),
                 limit: int = typer.Option(50, min=1, max=200),
                 as_json: bool = typer.Option(False, "--json")) -> None:
    """Find symbols and modules by literal text."""
    _run_query(_query_engine(repo, index_dir, as_json), "find_symbol", as_json, pattern=pattern, limit=limit)


@app.command()
def impact(target: str, repo: Path = typer.Option(Path("."), "--repo"),
           index_dir: Path | None = typer.Option(None, "--index-dir"),
           depth: int = typer.Option(10, min=1, max=20),
           limit: int = typer.Option(100, min=1, max=200),
           as_json: bool = typer.Option(False, "--json")) -> None:
    """Trace potential affected consumers of a module or dataset."""
    _run_query(_query_engine(repo, index_dir, as_json), "analyze_impact", as_json, target=target, depth=depth, limit=limit)


@app.command()
def lineage(dataset: str, repo: Path = typer.Option(Path("."), "--repo"),
            direction: str = typer.Option("upstream", "--direction"),
            index_dir: Path | None = typer.Option(None, "--index-dir"),
            depth: int = typer.Option(10, min=1, max=20),
            limit: int = typer.Option(100, min=1, max=200),
            as_json: bool = typer.Option(False, "--json")) -> None:
    """Trace supported upstream or downstream dataset lineage."""
    _run_query(_query_engine(repo, index_dir, as_json), "trace_data_lineage", as_json,
               dataset=dataset, direction=direction, depth=depth, limit=limit)


if __name__ == "__main__":
    main()
