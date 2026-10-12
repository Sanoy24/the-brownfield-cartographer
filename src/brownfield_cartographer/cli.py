"""
CLI entry point for The Brownfield Cartographer.

Provides the `cartographer` commands:
  - analyze: Ingest a local or remote repo and generate cartography artifacts
  - query:   Interactively query a previously analyzed codebase
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
from rich.prompt import Prompt
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


@app.callback()
def callback() -> None:
    """Brownfield Cartographer - local codebase intelligence."""
    pass


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
        orchestrator = Orchestrator(scan_settings=ScanSettings(dbt_manifest=dbt_manifest))
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
def query(
    path: str = typer.Argument(
        ".",
        help="Path to a directory containing .cartography/ artifacts (default: current directory)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable debug logging",
    ),
) -> None:
    """
    Interactively query a previously analyzed codebase.

    Opens a deterministic REPL for exploring a previously saved graph.

    Commands:
      find <concept>        — Search for implementations
      lineage <dataset>     — Trace upstream data lineage
      lineage-down <dataset>— Trace downstream lineage
      blast <module>        — Show blast radius
      explain <path>        — Explain a module
      summary               — Show graph summary

    Requires .cartography/ artifacts from a prior `analyze` run.
    Model credentials are not used by this command.
    """
    _setup_logging(verbose)

    from brownfield_cartographer.agents.navigator import run_interactive
    from brownfield_cartographer.graph.knowledge_graph import KnowledgeGraph

    base_path = Path(path).resolve()
    cartography_dir = base_path / ".cartography"

    if not cartography_dir.exists():
        # Maybe the path provided IS the artifact directory?
        if (base_path / "module_graph.json").exists():
            cartography_dir = base_path
        else:
            console.print(
                f"[bold red]Error:[/bold red] No .cartography/ directory found at {base_path}\n"
                f"[dim]Run `cartographer analyze <repo>` first to generate artifacts.[/dim]"
            )
            raise typer.Exit(code=1)

    # Multi-repo support: check for subdirectories (repo folders)
    repos = [d for d in cartography_dir.iterdir() if d.is_dir() and (d / "module_graph.json").exists()]

    target_dir = cartography_dir
    if repos:
        if len(repos) == 1:
            target_dir = repos[0]
            console.print(f"Selecting only repo found: [bold]{target_dir.name}[/bold]")
        else:
            console.print("\n[bold]Select a repository to query:[/bold]")
            for i, repo_dir in enumerate(repos, 1):
                console.print(f"  {i}. [cyan]{repo_dir.name}[/cyan]")

            choice = Prompt.ask(
                "\nEnter number",
                choices=[str(i) for i in range(1, len(repos) + 1)],
                default="1"
            )
            target_dir = repos[int(choice) - 1]

    console.print(f"Loading knowledge graph from [bold cyan]{target_dir}[/bold cyan]…")
    summary_path = target_dir / "analysis_summary.json"
    if summary_path.exists():
        try:
            run_status = json.loads(summary_path.read_text(encoding="utf-8")).get("status")
            if run_status in {"partial", "failed"}:
                console.print("Warning: this index has incomplete coverage; inspect analysis_summary.json.", markup=False)
        except (OSError, ValueError):
            console.print("Warning: run diagnostics could not be read.", markup=False)

    try:
        kg = KnowledgeGraph.from_artifacts(target_dir)
    except Exception as exc:
        console.print(f"[bold red]Failed to load graph:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    summary = kg.summary()
    if summary['total_nodes'] == 0:
        console.print(
            "[bold yellow]Warning:[/bold yellow] Loaded graph is empty. "
            "The analysis might have failed or found no source files."
        )
    else:
        console.print(
            f"  Loaded: {summary['modules']} modules, "
            f"{summary['datasets']} datasets, "
            f"{summary['total_nodes']} nodes, "
            f"{summary['total_edges']} edges"
        )

    run_interactive(kg)


def main() -> None:
    """Entry point for `python -m src.cli`."""
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
        console.print("Results were capped; narrow the query or increase the limit.", markup=False)
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
