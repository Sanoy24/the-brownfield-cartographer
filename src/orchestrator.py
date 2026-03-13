"""
Orchestrator — wires the Cartographer analysis pipeline.

Coordinates all four analysis agents in sequence, manages the shared
KnowledgeGraph, and serializes all outputs to the .cartography/ directory.

Pipeline: Surveyor → Hydrologist → Semanticist → Archivist
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from urllib.parse import urlparse

from src.agents.archivist import Archivist
from src.agents.hydrologist import Hydrologist
from src.agents.semanticist import Semanticist
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

    Uses --depth 100 to preserve enough git history for velocity analysis
    while keeping clone time reasonable.

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
            ["git", "clone", "--depth", "100", url, str(target_dir)],
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


# ---------------------------------------------------------------------------
# Incremental Update
# ---------------------------------------------------------------------------

def _check_incremental(cartography_dir: Path, repo_root: Path) -> set[str] | None:
    """
    Check if an incremental update is possible.

    Reads .cartography/state.json for the last run timestamp,
    then uses git diff to find changed files since that time.

    Returns:
        Set of changed file paths (relative to repo_root), or None
        if a full run is needed (no previous state or not a git repo).
    """
    state_path = cartography_dir / "state.json"
    if not state_path.exists():
        return None

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        last_commit = state.get("last_commit_hash")
        if not last_commit:
            return None
    except (json.JSONDecodeError, OSError):
        return None

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", last_commit, "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None

        changed = {
            line.strip().replace("\\", "/")
            for line in result.stdout.strip().split("\n")
            if line.strip()
        }
        return changed if changed else set()

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _save_state(cartography_dir: Path, repo_root: Path) -> None:
    """Save the current git state for incremental updates."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            state = {
                "last_commit_hash": result.stdout.strip(),
                "last_run": datetime.now(timezone.utc).isoformat(),
            }
            state_path = cartography_dir / "state.json"
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Non-critical


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Coordinates the full Cartographer analysis pipeline.

    Pipeline: Surveyor → Hydrologist → Semanticist → Archivist

    Handles both local paths and GitHub URLs as input,
    cloning remote repos to a temporary directory when needed.
    Supports incremental updates via git diff.
    """

    def __init__(self) -> None:
        self.knowledge_graph = KnowledgeGraph()
        self.surveyor = Surveyor()
        self.hydrologist = Hydrologist()
        self.semanticist = Semanticist()
        self.archivist = Archivist()
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    def run(self, repo_input: str, output_dir: Path | None = None) -> dict[str, Any]:
        """
        Execute the full analysis pipeline.

        Args:
            repo_input: Local path to a repo OR a GitHub URL.
            output_dir: Where to write .cartography/ artifacts.
                        Defaults to cwd for GitHub URLs, repo root for local paths.

        Returns:
            Combined summary from all agents.
        """
        start_time = time.monotonic()
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

        # Initialize trace logger
        trace_logger = self.archivist.init_trace(cartography_dir)

        trace_logger.log(
            agent="orchestrator",
            action="pipeline_started",
            details={
                "repo_input": repo_input,
                "repo_root": str(repo_root),
                "output_dir": str(cartography_dir),
            },
        )

        console.print(
            Panel(
                f"[bold]Analyzing:[/bold] {repo_root}\n"
                f"[bold]Output:[/bold]   {cartography_dir}\n"
                f"[bold]LLM:[/bold]      {self.semanticist.llm_config.provider or 'none (set OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, or OLLAMA_BASE_URL)'}",
                title="🗺️  Brownfield Cartographer",
                border_style="blue",
            )
        )

        # Check for incremental update
        changed_files = _check_incremental(cartography_dir, repo_root)
        if changed_files is not None and len(changed_files) == 0:
            console.print("[green]No changes detected since last run — skipping analysis.[/green]")
            trace_logger.log(agent="orchestrator", action="incremental_skip")
            trace_logger.flush()
            return {"status": "no_changes"}

        # When we have changed files and previous artifacts exist, load and re-analyze only those files
        incremental_run = (
            changed_files is not None
            and len(changed_files) > 0
            and (cartography_dir / "module_graph.json").exists()
            and (cartography_dir / "lineage_graph.json").exists()
        )
        if incremental_run:
            console.print(
                f"[cyan]Incremental mode: loading previous state, re-analyzing {len(changed_files)} changed files[/cyan]"
            )
            self.knowledge_graph.load_from_artifacts_replace(cartography_dir)
            trace_logger.log(
                agent="orchestrator",
                action="incremental_update",
                details={"changed_files": sorted(changed_files)[:50]},
            )
        elif changed_files:
            console.print(
                f"[cyan]Changed files detected but no previous run — doing full analysis[/cyan]"
            )

        surveyor_changed = changed_files if incremental_run else None
        hydrologist_changed = changed_files if incremental_run else None
        semanticist_changed = changed_files if incremental_run else None

        results: dict[str, Any] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # --- Phase 1: Surveyor ---
            task = progress.add_task("Running Surveyor (static structure)…", total=None)
            surveyor_results = self.surveyor.run(
                repo_root, self.knowledge_graph, changed_files=surveyor_changed
            )
            results["surveyor"] = surveyor_results
            trace_logger.log(
                agent="surveyor",
                action="analysis_complete",
                details={"modules_parsed": surveyor_results.get("modules_parsed", 0)},
            )
            progress.update(task, completed=True, description="✓ Surveyor complete")

            # --- Phase 2: Hydrologist ---
            task = progress.add_task("Running Hydrologist (data lineage)…", total=None)
            hydrologist_results = self.hydrologist.run(
                repo_root, self.knowledge_graph, changed_files=hydrologist_changed
            )
            results["hydrologist"] = hydrologist_results
            trace_logger.log(
                agent="hydrologist",
                action="analysis_complete",
                details={
                    "transformations": hydrologist_results.get("total_transformations", 0),
                    "datasets": hydrologist_results.get("total_datasets", 0),
                },
            )
            progress.update(task, completed=True, description="✓ Hydrologist complete")

            # --- Phase 3: Semanticist ---
            task = progress.add_task("Running Semanticist (LLM analysis)…", total=None)
            semanticist_results = self.semanticist.run(
                repo_root,
                self.knowledge_graph,
                surveyor_results=surveyor_results,
                hydrologist_results=hydrologist_results,
                changed_modules=semanticist_changed,
            )
            results["semanticist"] = semanticist_results
            trace_logger.log(
                agent="semanticist",
                action="analysis_complete",
                evidence_source="llm_inference",
                details={
                    "provider": semanticist_results.get("llm_provider", "none"),
                    "purpose_statements": len(semanticist_results.get("purpose_statements", {})),
                    "token_budget": semanticist_results.get("token_budget", {}),
                },
            )
            if self.semanticist.available:
                progress.update(task, completed=True, description="✓ Semanticist complete")
            else:
                progress.update(task, completed=True, description="⊘ Semanticist skipped (no LLM)")

            # --- Phase 4: Serialize core graphs ---
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
            progress.update(task, completed=True, description="✓ Graphs serialized")

            # --- Phase 5: Archivist ---
            task = progress.add_task("Running Archivist (living context)…", total=None)
            archivist_results = self.archivist.run(
                repo_root,
                self.knowledge_graph,
                surveyor_results,
                hydrologist_results,
                semanticist_results,
                cartography_dir,
            )
            results["archivist"] = archivist_results
            progress.update(task, completed=True, description="✓ Archivist complete")

        # Save incremental state
        _save_state(cartography_dir, repo_root)

        # Print summary table
        elapsed = time.monotonic() - start_time
        graph_summary = self.knowledge_graph.summary()
        console.print()
        console.print(Panel(
            f"[bold green]Analysis Complete[/bold green] ({elapsed:.1f}s)\n\n"
            f"  Modules parsed:      {graph_summary['modules']}\n"
            f"  Functions extracted:  {graph_summary['functions']}\n"
            f"  Datasets discovered: {graph_summary['datasets']}\n"
            f"  Transformations:     {graph_summary['transformations']}\n"
            f"  Circular deps:       {graph_summary['circular_dependencies']}\n"
            f"  Total graph nodes:   {graph_summary['total_nodes']}\n"
            f"  Total graph edges:   {graph_summary['total_edges']}\n\n"
            f"  LLM provider:        {semanticist_results.get('llm_provider', 'none')}\n"
            f"  Purpose statements:  {len(semanticist_results.get('purpose_statements', {}))}\n\n"
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
