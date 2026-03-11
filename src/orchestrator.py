"""
Orchestrator — wires the Cartographer analysis pipeline.

Coordinates the Surveyor and Hydrologist agents in sequence, manages
the shared KnowledgeGraph, and serializes all outputs to the
.cartography/ directory.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.agents.hydrologist import Hydrologist
from src.agents.surveyor import Surveyor
from src.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)
console = Console()


def _is_github_url(repo_input: str) -> bool:
    """Check if the input looks like a GitHub URL."""
    parsed = urlparse(repo_input)
    return parsed.scheme in ("http", "https") and "github" in parsed.netloc


def _clone_repo(url: str, target_dir: Path) -> Path:
    """
    Clone a GitHub repository to a local directory.

    Args:
        url: GitHub repository URL.
        target_dir: Directory to clone into.

    Returns:
        Path to the cloned repository root.

    Raises:
        RuntimeError: If git clone fails or times out.
    """
    console.print(f"  Cloning [bold cyan]{url}[/bold cyan] …")
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target_dir)],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"git clone timed out after 600s. Check your network connection "
            f"or clone the repo manually and pass a local path instead."
        )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    console.print(f"  ✓ Cloned to {target_dir}")
    return target_dir


class Orchestrator:
    """
    Coordinates the full Cartographer analysis pipeline.

    Pipeline: Surveyor → Hydrologist → serialize .cartography/ artifacts

    Handles both local paths and GitHub URLs as input,
    cloning remote repos to a temporary directory when needed.
    """

    def __init__(self) -> None:
        self.knowledge_graph = KnowledgeGraph()
        self.surveyor = Surveyor()
        self.hydrologist = Hydrologist()
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def run(self, repo_input: str, output_dir: Optional[Path] = None) -> dict[str, Any]:
        """
        Execute the full analysis pipeline.

        Args:
            repo_input: Local path to a repo OR a GitHub URL.
            output_dir: Where to write .cartography/ artifacts.
                        Defaults to cwd for GitHub URLs, repo root for local paths.

        Returns:
            Combined summary from all agents.
        """
        is_remote = _is_github_url(repo_input)
        repo_root = self._resolve_repo(repo_input)

        # For remote repos, write artifacts to cwd (not the temp clone dir).
        # For local repos, write artifacts alongside the repo itself.
        if output_dir:
            artifact_root = output_dir
        elif is_remote:
            artifact_root = Path.cwd()
        else:
            artifact_root = repo_root

        cartography_dir = artifact_root / ".cartography"
        cartography_dir.mkdir(parents=True, exist_ok=True)

        console.print(
            Panel(
                f"[bold]Analyzing:[/bold] {repo_root}\n"
                f"[bold]Output:[/bold]   {cartography_dir}",
                title="🗺️  Brownfield Cartographer",
                border_style="blue",
            )
        )

        results: dict[str, Any] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # --- Phase 1: Surveyor ---
            task = progress.add_task("Running Surveyor (static structure)…", total=None)
            surveyor_results = self.surveyor.run(repo_root, self.knowledge_graph)
            results["surveyor"] = surveyor_results
            progress.update(task, completed=True, description="✓ Surveyor complete")

            # --- Phase 2: Hydrologist ---
            task = progress.add_task("Running Hydrologist (data lineage)…", total=None)
            hydrologist_results = self.hydrologist.run(repo_root, self.knowledge_graph)
            results["hydrologist"] = hydrologist_results
            progress.update(task, completed=True, description="✓ Hydrologist complete")

            # --- Serialize outputs ---
            task = progress.add_task("Serializing knowledge graph…", total=None)
            self.knowledge_graph.serialize_module_graph(
                cartography_dir / "module_graph.json"
            )
            self.knowledge_graph.serialize_lineage_graph(
                cartography_dir / "lineage_graph.json"
            )

            # Write combined summary
            summary_path = cartography_dir / "analysis_summary.json"
            summary_path.write_text(
                json.dumps(results, indent=2, default=str), encoding="utf-8"
            )
            progress.update(task, completed=True, description="✓ Artifacts written")

        # Print summary table
        graph_summary = self.knowledge_graph.summary()
        console.print()
        console.print(Panel(
            f"[bold green]Analysis Complete[/bold green]\n\n"
            f"  Modules parsed:      {graph_summary['modules']}\n"
            f"  Functions extracted:  {graph_summary['functions']}\n"
            f"  Datasets discovered: {graph_summary['datasets']}\n"
            f"  Transformations:     {graph_summary['transformations']}\n"
            f"  Circular deps:       {graph_summary['circular_dependencies']}\n"
            f"  Total graph nodes:   {graph_summary['total_nodes']}\n"
            f"  Total graph edges:   {graph_summary['total_edges']}\n\n"
            f"  Artifacts → {cartography_dir}",
            title="📊 Summary",
            border_style="green",
        ))

        results["graph_summary"] = graph_summary
        return results

    def _resolve_repo(self, repo_input: str) -> Path:
        """
        Resolve repo_input to a local directory path.

        If the input is a GitHub URL, clone it to a temporary directory.
        If it's a local path, validate it exists.
        """
        if _is_github_url(repo_input):
            self._temp_dir = tempfile.TemporaryDirectory(prefix="cartographer_")
            clone_path = Path(self._temp_dir.name) / "repo"
            return _clone_repo(repo_input, clone_path)
        else:
            local_path = Path(repo_input).resolve()
            if not local_path.exists():
                raise FileNotFoundError(f"Repository path not found: {local_path}")
            if not local_path.is_dir():
                raise NotADirectoryError(f"Not a directory: {local_path}")
            return local_path

    def cleanup(self) -> None:
        """Clean up any temporary directories created for cloned repos."""
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None
