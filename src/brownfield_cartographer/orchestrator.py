"""Local-first pipeline and successful-snapshot checkpoint management."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rich.console import Console

from brownfield_cartographer.agents.archivist import Archivist
from brownfield_cartographer.agents.hydrologist import Hydrologist
from brownfield_cartographer.agents.surveyor import Surveyor
from brownfield_cartographer.graph.knowledge_graph import KnowledgeGraph
from brownfield_cartographer.persistence import write_json
from brownfield_cartographer.scanner import ScanResult, ScanSettings, scan_repository

logger = logging.getLogger(__name__)
console = Console()
ANALYSIS_VERSION = 4


def _is_github_url(value: str) -> bool:
    parsed = urlparse(value)
    return (parsed.scheme == "https" and parsed.hostname == "github.com"
            and parsed.username is None and parsed.port is None)


def _derive_repo_name(value: str, *, isolate_local: bool = False) -> str:
    if _is_github_url(value):
        parts = urlparse(value).path.strip("/").removesuffix(".git").split("/")
        if len(parts) != 2 or any(not p or p in {".", ".."} for p in parts):
            raise ValueError("Use a canonical https://github.com/owner/repository URL")
        return "--".join(parts)
    path = Path(value).resolve()
    if isolate_local:
        identity = os.path.normcase(str(path))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return f"{path.name}--{digest}"
    return path.name


def _clone_repo(url: str, target_dir: Path) -> Path:
    _derive_repo_name(url)
    try:
        result = subprocess.run(["git", "clone", "--depth", "100", "--", url, str(target_dir)],
                                capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise RuntimeError("Could not clone repository; pass an existing local checkout") from exc
    if result.returncode:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    return target_dir


def _checkpoint_key(repo_root: Path, inventory: ScanResult, settings: ScanSettings) -> dict:
    return {"analysis_version": ANALYSIS_VERSION, "repository": str(repo_root),
            "settings": vars(settings), "fingerprints": inventory.fingerprints,
            "velocity_date": datetime.now(UTC).date().isoformat()}


def _check_incremental(cartography_dir: Path, repo_root: Path,
                       inventory: ScanResult | None = None,
                       settings: ScanSettings | None = None) -> set[str] | None:
    """Content detection includes dirty/untracked/deleted files.

    Changed snapshots use a full rebuild until selective invalidation can
    satisfy the complete rebuild contract.
    """
    settings = settings or ScanSettings()
    inventory = inventory or scan_repository(repo_root, settings)
    try:
        state = json.loads((cartography_dir / "state.json").read_text(encoding="utf-8"))
        key = _checkpoint_key(repo_root, inventory, settings)
        if any(state.get(k) != v for k, v in key.items() if k != "fingerprints"):
            return None
        previous = state.get("fingerprints", {})
        return {p for p in previous.keys() | inventory.fingerprints.keys()
                if previous.get(p) != inventory.fingerprints.get(p)}
    except (OSError, ValueError, TypeError):
        return None


def _artifact_directory(root: Path, name: str) -> Path:
    path = root.absolute() / ".cartography" / name
    for candidate in (path, *path.parents):
        if candidate.is_symlink() or candidate.is_junction():
            raise ValueError("Artifact paths must not pass through symlinks or junctions")
    path.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() or p.is_junction() for p in path.iterdir()):
        raise ValueError("Artifact directory contains a symlink or junction")
    return path


class Orchestrator:
    """Run static analysis offline."""

    def __init__(self, *, scan_settings: ScanSettings | None = None):
        self.knowledge_graph = KnowledgeGraph()
        self.surveyor = Surveyor()
        self.hydrologist = Hydrologist()
        self.archivist = Archivist()
        self.scan_settings = scan_settings or ScanSettings()
        self._temp_dir = None

    def run(self, repo_input: str, output_dir: Path | None = None, *, full: bool = False) -> dict[str, Any]:
        start = time.monotonic()
        repo_root = self._resolve_repo(repo_input)
        inventory = scan_repository(repo_root, self.scan_settings)
        root = output_dir if output_dir is not None else (Path.cwd() if _is_github_url(repo_input) else repo_root)
        artifacts = _artifact_directory(root, _derive_repo_name(
            repo_input, isolate_local=output_dir is not None,
        ))
        trace = self.archivist.init_trace(artifacts)
        trace.log(agent="orchestrator", action="pipeline_started",
                  details={"repo_root": str(repo_root), "scan": inventory.report()})
        changed = None if full else _check_incremental(artifacts, repo_root, inventory, self.scan_settings)
        if changed == set() and inventory.complete:
            try:
                summary = json.loads((artifacts / "analysis_summary.json").read_text(encoding="utf-8"))
                if summary.get("status") == "complete":
                    self.knowledge_graph.load_index(artifacts / "index.json")
                    trace.log(agent="orchestrator", action="unchanged_snapshot")
                    trace.flush()
                    return {**summary, "status": "no_changes"}
            except (OSError, ValueError, KeyError, TypeError):
                pass

        self.knowledge_graph.clear()
        results: dict[str, Any] = {"scan": inventory.report(), "run_mode": "full_rebuild",
                                   "artifact_directory": str(artifacts.resolve()),
                                   "changed_files": sorted(changed) if changed is not None else None}
        errors = {}
        try:
            for name, stage in (("surveyor", self.surveyor), ("hydrologist", self.hydrologist)):
                try:
                    results[name] = stage.run(repo_root, self.knowledge_graph, inventory=inventory)
                    trace.log(agent=name, action="analysis_complete", details=results[name])
                except Exception as exc:
                    errors[name] = str(exc)
                    results[name] = {}
                    logger.error("%s failed: %s", name, exc)
                    trace.log(agent=name, action="failed", confidence=0, details={"error": str(exc)})
            final_inventory = scan_repository(repo_root, self.scan_settings)
            if final_inventory.fingerprints != inventory.fingerprints:
                errors["snapshot"] = "Repository changed during analysis; retry on a stable working tree"
            results["scan"] = final_inventory.report()
            self.knowledge_graph.graph.graph["repository"] = str(repo_root)
            self.knowledge_graph.graph.graph["snapshot"] = inventory.fingerprints
            self.knowledge_graph.graph.graph["scan_settings"] = vars(self.scan_settings)
            incomplete_stages = any(d.get("incomplete") for name in ("surveyor", "hydrologist")
                                    for d in results[name].get("diagnostics", []))
            self.knowledge_graph.graph.graph["coverage"] = {
                "scan": final_inventory.report(),
                "diagnostics": {name: results[name].get("diagnostics", [])
                                for name in ("surveyor", "hydrologist")},
                "unsupported_structure": results["surveyor"].get("unsupported_structure", []),
            }
            self.knowledge_graph.serialize_module_graph(artifacts / "module_graph.json")
            self.knowledge_graph.serialize_lineage_graph(artifacts / "lineage_graph.json")
            try:
                results["archivist"] = self.archivist.run(repo_root, self.knowledge_graph,
                    results["surveyor"], results["hydrologist"], artifacts)
            except Exception as exc:
                errors["archivist"] = str(exc)
            results["graph_summary"] = self.knowledge_graph.summary()
            results["agent_errors"] = errors
            results["elapsed_seconds"] = round(time.monotonic() - start, 3)
            results["status"] = "complete" if not errors and not incomplete_stages and inventory.complete and final_inventory.complete else "partial"
            self.knowledge_graph.graph.graph["run_status"] = results["status"]
            write_json(artifacts / "analysis_summary.json", results)
            self.knowledge_graph.serialize_index(artifacts / "index.json")
            trace.log(agent="orchestrator", action="pipeline_finished", details={"status": results["status"]})
            trace.flush()
            if results["status"] == "complete":
                write_json(artifacts / "state.json", _checkpoint_key(repo_root, inventory, self.scan_settings))
            console.print(f"Analysis {results['status']}: {artifacts}", markup=False)
            return results
        except Exception as exc:
            errors["artifacts"] = str(exc)
            results.update(status="failed", agent_errors=errors)
            write_json(artifacts / "analysis_summary.json", results)
            trace.log(agent="orchestrator", action="failed", confidence=0, details=errors)
            raise
        finally:
            if trace._entries:
                trace.flush()

    def _resolve_repo(self, repo_input: str) -> Path:
        if _is_github_url(repo_input):
            self._temp_dir = tempfile.TemporaryDirectory(prefix="cartographer_")
            return _clone_repo(repo_input, Path(self._temp_dir.name) / "repo")
        if urlparse(repo_input).scheme in {"http", "https"}:
            raise ValueError("Only canonical HTTPS GitHub repository URLs are supported")
        path = Path(repo_input)
        if path.is_symlink() or path.is_junction():
            raise ValueError("Repository root must not be a symlink or junction")
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise NotADirectoryError(str(path))
        return path

    def cleanup(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
