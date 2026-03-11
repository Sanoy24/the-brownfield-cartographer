"""
CLI entry point for The Brownfield Cartographer.

Provides the `cartographer` commands:
  - analyze: Ingest a local or remote repo and generate cartography artifacts
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from src.orchestrator import Orchestrator

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
    output: str = typer.Option(
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

    Runs the Surveyor (static structure) and Hydrologist (data lineage)
    agents on any codebase containing Python, SQL, and/or YAML files.

    Supports any project with Python, SQL, YAML, or mixed stacks — including
    dbt projects, Airflow DAGs, Django/Flask/FastAPI apps, data pipelines, etc.
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


def main() -> None:
    """Entry point for `python -m src.cli`."""
    app()


if __name__ == "__main__":
    main()
