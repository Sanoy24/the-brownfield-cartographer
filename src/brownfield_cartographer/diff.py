"""Map a git diff onto the graph: which nodes changed, and what they can affect."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from collections import deque
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from brownfield_cartographer.orchestrator import Orchestrator
from brownfield_cartographer.query_api import QueryEngine, QueryError
from brownfield_cartographer.scanner import ScanSettings

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# Modules whose own identity matters to importers. SQL files are represented by
# the datasets their statements produce; YAML files by the models they configure.
_CODE_LANGUAGES = {"python", "javascript", "typescript"}


def _git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", "-c", "core.quotePath=false", *args], cwd=repo,
                                capture_output=True, text=True, encoding="utf-8", timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise QueryError("git is required for cartographer diff") from exc
    if result.returncode:
        raise QueryError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout


Ranges = dict[str, list[tuple[int, int]]]


def changed_lines(repo: Path, base: str) -> tuple[str, Ranges, Ranges, list[str]]:
    """Changes since the merge base with `base`, including uncommitted and untracked files.

    Returns the merge base, changed line ranges per file in the current version
    and in the merge base, and deleted files. In the current version a pure
    deletion marks the lines on both sides of it; in the merge base a pure
    insertion removes nothing and is not marked.
    """
    merge_base = _git(repo, "merge-base", base, "HEAD").strip()
    deleted = []
    fields = _git(repo, "diff", "--name-status", "--no-renames", "--relative", "-z", merge_base).split("\0")
    for status, path in zip(fields[::2], fields[1::2], strict=False):
        if status == "D":
            deleted.append(path)
    ranges: Ranges = {}
    old_ranges: Ranges = {}
    current = old = None
    for line in _git(repo, "diff", "-U0", "--no-renames", "--relative", "--no-ext-diff", merge_base).splitlines():
        if line.startswith("--- "):
            old = line[6:] if line.startswith("--- a/") else None
        elif line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else None
        elif match := _HUNK.match(line):
            old_start, old_count = int(match[1]), int(match[2] if match[2] is not None else 1)
            start, count = int(match[3]), int(match[4] if match[4] is not None else 1)
            if current:
                span = (start, start + count - 1) if count else (max(start, 1), start + 1)
                ranges.setdefault(current, []).append(span)
            if old and old_count:
                old_ranges.setdefault(old, []).append((old_start, old_start + old_count - 1))
    for path in _git(repo, "ls-files", "--others", "--exclude-standard").splitlines():
        if not path.startswith(".cartography/"):
            ranges[path] = [(1, 10**9)]
    return merge_base, ranges, old_ranges, sorted(deleted)


def _span(attrs: dict) -> tuple[int, int] | None:
    span = attrs.get("line_range")
    if span is None and attrs.get("line_start"):
        span = (attrs["line_start"], attrs.get("line_end") or attrs["line_start"])
    return (span[0], span[1]) if span and span[0] > 0 else None


def _overlaps(span: tuple[int, int] | None, ranges: list[tuple[int, int]]) -> bool:
    return span is not None and any(start <= span[1] and span[0] <= end for start, end in ranges)


def _changed_nodes(engine: QueryEngine, ranges: dict[str, list[tuple[int, int]]]) -> dict[str, str]:
    """Changed node -> changed file."""
    graph = engine.graph.graph
    changed: dict[str, str] = {}
    for node, attrs in graph.nodes(data=True):
        kind = attrs.get("node_type")
        path = attrs.get("parent_module") if kind == "function" else attrs.get("source_file")
        if kind == "module" and node in ranges and (
                str(attrs.get("language", "")).lower() in _CODE_LANGUAGES
                or any(e.get("edge_type") == "CONFIGURES" for _, _, e in graph.out_edges(node, data=True))):
            changed[node] = node
        elif kind in {"function", "task"} and path in ranges and _overlaps(_span(attrs), ranges[path]):
            changed[node] = path
        elif kind == "transformation" and path in ranges and _overlaps(_span(attrs), ranges[path]):
            # A changed statement or model changes the datasets it writes.
            for target in graph.successors(node):
                if graph.edges[node, target].get("edge_type") == "PRODUCES":
                    changed.setdefault(target, path)
    return changed


def _affected(engine: QueryEngine, changed: dict[str, str], depth: int, limit: int,
              report=lambda node: True, skip: set[str] | frozenset[str] = frozenset()) -> tuple[list[dict], str | None]:
    """Dependents of the changed nodes; only nodes passing `report` are listed, but all are traversed."""
    graph = engine.graph.graph
    queue = deque((node, 0, node) for node in sorted(changed))
    seen = set(changed) | set(skip)
    rows: list[dict] = []
    stopped_by = None
    while queue:
        current, distance, origin = queue.popleft()
        candidates = sorted(engine.graph.impact_neighbors(current), key=lambda pair: pair[0])
        for neighbor, edge in candidates:
            # A changed module's own definitions are reported only if their lines changed.
            if neighbor in seen or (current in changed and edge.get("edge_type") == "DEFINES"
                                    and graph.nodes[current].get("node_type") == "module"):
                continue
            if distance >= depth:
                stopped_by = "depth"
                continue
            seen.add(neighbor)
            attrs = graph.nodes[neighbor]
            if attrs.get("node_type") == "transformation":
                queue.appendleft((neighbor, distance, origin))
                continue
            if report(neighbor):
                rows.append({"node": neighbor, "node_type": attrs.get("node_type"), "distance": distance + 1,
                             "via": origin, "edge_type": edge.get("edge_type"), "evidence": engine._evidence(neighbor, edge)})
            if len(rows) >= limit or len(seen) >= engine.MAX_VISITED:
                return rows, "limit"
            # Configuration (tests, docs, metrics) does not change the data downstream consumers read.
            if edge.get("edge_type") != "CONFIGURES":
                queue.append((neighbor, distance + 1, neighbor))
    return rows, stopped_by


def _located(engine: QueryEngine, node: str, path: str) -> dict[str, Any]:
    """Evidence for a changed node; datasets have no span of their own, so cite the changed file."""
    evidence = engine._evidence(node)
    if not evidence["path"]:
        evidence["path"] = path
    return evidence


def change_impact(repo: Path, base: str = "main", *, depth: int = 10, limit: int = 100,
                  scan_settings: ScanSettings | None = None) -> dict[str, Any]:
    """Analyze the current checkout and report the nodes a diff against `base` changes and affects."""
    repo = Path(repo).resolve(strict=True)
    if not 1 <= depth <= QueryEngine.MAX_DEPTH or not 1 <= limit <= QueryEngine.MAX_RESULTS:
        raise QueryError("Depth or limit is out of range")
    merge_base, ranges, old_ranges, deleted = changed_lines(repo, base)
    with redirect_stdout(sys.stderr):
        run = Orchestrator(scan_settings=scan_settings).run(str(repo))
    engine = QueryEngine.open(repo)
    graph = engine.graph.graph
    changed = _changed_nodes(engine, ranges)
    affected, stopped_by = _affected(engine, changed, depth, limit)
    rows = [{"node": node, "node_type": graph.nodes[node].get("node_type"), "path": path, "removed": False,
             "evidence": _located(engine, node, path)} for node, path in sorted(changed.items())]

    statuses = [run.get("status")]
    if old_ranges:
        # Removed lines can only be mapped in the merge base: deleted files and functions,
        # and statements whose tables are no longer written. Their dependents that still
        # exist are affected.
        with tempfile.TemporaryDirectory(prefix="cartographer_base_") as temp:
            worktree = Path(temp) / "base"
            prefix = _git(repo, "rev-parse", "--show-prefix").strip()
            _git(repo, "worktree", "add", "--quiet", "--detach", str(worktree), merge_base)
            try:
                base_root = worktree / prefix
                with redirect_stdout(sys.stderr):
                    base_run = Orchestrator(scan_settings=scan_settings).run(str(base_root), output_dir=Path(temp) / "out")
                statuses.append(base_run.get("status"))
                base_engine = QueryEngine.open(base_root, Path(base_run["artifact_directory"]))
                base_changed = {node: path for node, path in _changed_nodes(base_engine, old_ranges).items()
                                if node not in changed}
                reported = {row["node"] for row in affected}
                more, more_stopped = _affected(base_engine, base_changed, depth, limit - len(affected),
                                               report=lambda node: node in graph and node not in reported,
                                               skip=set(changed)) if len(affected) < limit else ([], "limit")
                affected += more
                stopped_by = stopped_by or more_stopped
                base_graph = base_engine.graph.graph
                rows += [{"node": node, "node_type": base_graph.nodes[node].get("node_type"), "path": path,
                          "removed": node not in graph, "evidence": _located(base_engine, node, path)}
                         for node, path in sorted(base_changed.items())]
            finally:
                _git(repo, "worktree", "remove", "--force", str(worktree))
    status = next((s for s in ("failed", "partial") if s in statuses), "complete")
    covered = {row["path"] for row in rows}
    return {
        "schema_version": 1, "query": "change_impact", "repository": str(repo), "base": base,
        "merge_base": merge_base, "status": status, "snapshot_id": engine.snapshot_id,
        "changed_files": sorted(ranges.keys() | set(deleted)), "deleted_files": deleted,
        "unmapped_files": sorted((ranges.keys() | set(deleted)) - covered),
        "changed": rows, "affected": affected, "stopped_by": stopped_by, "truncated": stopped_by is not None,
    }


def to_markdown(report: dict[str, Any]) -> str:
    """A pull-request comment: what the changes can affect, then the changed nodes, with evidence."""
    def location(evidence: dict) -> str:
        text = evidence.get("path") or "unknown"
        return text + (":" + "-".join(map(str, evidence["line_range"])) if evidence.get("line_range") else "")

    lines = ["## Cartographer change impact", ""]
    if not report["changed"] and not report["deleted_files"]:
        lines.append(f"No indexed code, models, tasks or datasets changed since `{report['base']}`.")
    if report["affected"]:
        lines += [f"**{len(report['affected'])} potentially affected** by {len(report['changed'])} changed nodes "
                  f"since `{report['base']}`:", "",
                  "| Affected | Type | Via | Evidence |", "| --- | --- | --- | --- |"]
        lines += [f"| `{row['node']}` | {row['node_type']} | `{row['via']}` ({row['edge_type']}) | {location(row['evidence'])} |"
                  for row in report["affected"]]
        lines.append("")
    elif report["changed"]:
        lines += [f"Nothing indexed depends on the {len(report['changed'])} changed nodes since `{report['base']}`.", ""]
    if report["changed"]:
        table = ["| Changed | Type | Location |", "| --- | --- | --- |"]
        table += [f"| `{row['node']}` | {row['node_type']}{' (removed)' if row.get('removed') else ''} "
                  f"| {location(row['evidence'])} |" for row in report["changed"]]
        if len(report["changed"]) > 10:
            # Keep long lists out of the way; GitHub needs blank lines around a table inside <details>.
            table = ["<details><summary>Changed nodes</summary>", "", *table, "", "</details>"]
        lines += [*table, ""]
    if report["deleted_files"]:
        lines += ["Deleted files: " + ", ".join(f"`{path}`" for path in report["deleted_files"]), ""]
    if report["truncated"]:
        lines += ["Results were capped; run `cartographer diff` with a larger `--depth` or `--limit`.", ""]
    if report["status"] != "complete":
        lines += [f"Analysis status is **{report['status']}**; check coverage before relying on this list.", ""]
    lines.append("<sub>Static analysis by Brownfield Cartographer: potential dependents, not a guarantee of breakage.</sub>")
    return "\n".join(lines) + "\n"
