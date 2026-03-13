"""
CLI entry point for The Brownfield Cartographer.

Provides the `cartographer` commands:
  - analyze: Ingest a local or remote repo and generate cartography artifacts
  - query:   Interactively query a previously analyzed codebase
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Prompt

from src.orchestrator import Orchestrator

# Load environment variables from .env file
load_dotenv()

console = Console()

# Main app with a default callback that does nothing — this forces typer
# to treat decorated functions as proper subcommands instead of the root.
app = typer.Typer(
    name="cartographer",
    help="🗺️  The Brownfield Cartographer — Codebase Intelligence System",
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
    """🗺️  The Brownfield Cartographer — Codebase Intelligence System"""
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
) -> None:
    """
    Analyze a codebase and generate cartography artifacts.

    Runs the full four-agent pipeline:
      Surveyor (static structure) → Hydrologist (data lineage) →
      Semanticist (LLM analysis) → Archivist (living context)

    Supports any project with Python, SQL, YAML, or mixed stacks — including
    dbt projects, Airflow DAGs, Django/Flask/FastAPI apps, data pipelines, etc.

    Set one of these env vars for LLM-powered analysis:
      OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, OLLAMA_BASE_URL
    """
    _setup_logging(verbose)

    output_dir = Path(output) if output else None
    orchestrator = Orchestrator()

    try:
        results = orchestrator.run(repo, output_dir=output_dir)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    finally:
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

    Opens the Navigator — an interactive REPL for exploring the
    knowledge graph with both direct commands and natural language queries.

    Commands:
      find <concept>        — Search for implementations
      lineage <dataset>     — Trace upstream data lineage
      lineage-down <dataset>— Trace downstream lineage
      blast <module>        — Show blast radius
      explain <path>        — Explain a module
      summary               — Show graph summary

    Requires .cartography/ artifacts from a prior `analyze` run.
    Set an LLM env var (OPENAI_API_KEY, etc.) for natural language queries.
    """
    _setup_logging(verbose)

    from src.agents.navigator import run_interactive
    from src.graph.knowledge_graph import KnowledgeGraph

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
    
    try:
        kg = KnowledgeGraph.from_artifacts(target_dir)
    except Exception as exc:
        console.print(f"[bold red]Failed to load graph:[/bold red] {exc}")
        raise typer.Exit(code=1)
        
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


if __name__ == "__main__":
    main()
