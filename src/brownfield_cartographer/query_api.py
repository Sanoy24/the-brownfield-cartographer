"""One bounded, deterministic query service for Python, CLI, and MCP."""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Literal

from brownfield_cartographer.graph.knowledge_graph import KnowledgeGraph


class QueryError(ValueError):
    """Invalid scope, unsupported argument, ambiguous target, or stale source."""


def _artifact_stamp(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.lstat()
        return stat.st_ino, stat.st_mtime_ns, stat.st_size, stat.st_mode
    except FileNotFoundError:
        return None


class QueryEngine:
    MAX_RESULTS = 200
    MAX_DEPTH = 20
    MAX_RESPONSE_BYTES = 64_000
    MAX_VISITED = 5000

    def __init__(self, graph: KnowledgeGraph, repo_root: Path, *, run_status: str = "unknown"):
        self.graph = graph
        self.repo_root = repo_root.resolve(strict=True)
        indexed_root = graph.graph.graph.get("repository")
        if indexed_root is None or Path(indexed_root).resolve() != self.repo_root:
            raise QueryError("Index repository scope does not match the requested repository; analyze it again")
        self.run_status = run_status
        self.snapshot = graph.graph.graph.get("snapshot", {})
        self.snapshot_id = hashlib.sha256(json.dumps(self.snapshot, sort_keys=True).encode()).hexdigest()
        self.index_dir: Path | None = None
        self.index_stamp: tuple[int, int, int, int] | None = None
        self.summary_stamp: tuple[int, int, int, int] | None = None

    @classmethod
    def open(cls, repo_root: Path, index_dir: Path | None = None) -> QueryEngine:
        try:
            return cls._open(repo_root, index_dir)
        except QueryError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise QueryError("Cannot read analysis artifacts; run cartographer analyze again") from exc

    @classmethod
    def _open(cls, repo_root: Path, index_dir: Path | None = None) -> QueryEngine:
        if repo_root.is_symlink() or repo_root.is_junction():
            raise QueryError("Repository root must not be a symlink or junction")
        repo_root = repo_root.resolve(strict=True)
        index_dir = index_dir or repo_root / ".cartography" / repo_root.name
        for path in (index_dir, *index_dir.parents):
            if path.is_symlink() or path.is_junction():
                raise QueryError("Index paths must not pass through symlinks or junctions")
        index_path = index_dir / "index.json"
        if index_path.is_symlink() or not index_path.is_file():
            raise QueryError(
                f'No index found for this repository. Build it (offline; repository code is not executed) with '
                f'`cartographer analyze "{repo_root}"`, or without installing: '
                f'`uvx --from brownfield-cartographer cartographer analyze "{repo_root}"`. Then retry this tool.')
        if index_path.stat().st_size > 100 * 1024 * 1024:
            raise QueryError("Index exceeds the local query size limit")
        summary_path = index_dir / "analysis_summary.json"
        status = "unknown"
        if summary_path.exists():
            if summary_path.is_symlink() or summary_path.stat().st_size > 2 * 1024 * 1024:
                raise QueryError("Invalid run summary")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise QueryError("Invalid run summary; run cartographer analyze again")
            status = summary.get("status", "unknown")
            if status not in {"unknown", "complete", "partial", "failed", "no_changes"}:
                raise QueryError("Invalid run summary status; run cartographer analyze again")
        graph = KnowledgeGraph.from_artifacts(index_dir)
        graph_status = graph.graph.graph.get("run_status", "unknown")
        for severity in ("failed", "partial"):
            if severity in (status, graph_status):
                status = severity
                break
        else:
            if status == "unknown":
                status = graph_status
        engine = cls(graph, repo_root, run_status=status)
        engine.index_dir = index_dir
        engine.index_stamp = _artifact_stamp(index_path)
        engine.summary_stamp = _artifact_stamp(summary_path)
        return engine

    @classmethod
    def open_lazy(cls, repo_root: Path, index_dir: Path | None = None) -> QueryEngine:
        """Open the index, or return an engine that opens it on first use once analysis has run."""
        try:
            return cls.open(repo_root, index_dir)
        except QueryError:
            if repo_root.is_symlink() or repo_root.is_junction():
                raise
            engine = cls.__new__(cls)
            engine.graph = None  # type: ignore[assignment]
            engine.repo_root = repo_root.resolve(strict=True)
            engine.index_dir = index_dir
            engine.index_stamp = engine.summary_stamp = None
            return engine

    def _refresh(self) -> None:
        """Observe graph and run-status changes without writing or analyzing."""
        if self.graph is None:
            # Not analyzed when the engine was created; open as soon as an index exists.
            self.__dict__.update(type(self).open(self.repo_root, self.index_dir).__dict__)
            return
        if self.index_dir is None:
            return
        try:
            changed = (
                _artifact_stamp(self.index_dir / "index.json") != self.index_stamp
                or _artifact_stamp(self.index_dir / "analysis_summary.json") != self.summary_stamp
            )
            if changed:
                fresh = type(self).open(self.repo_root, self.index_dir)
                self.__dict__.update(fresh.__dict__)
        except OSError as exc:
            raise QueryError("Cannot read analysis artifacts; run cartographer analyze again") from exc

    def _limit(self, value: int) -> int:
        if not 1 <= value <= self.MAX_RESULTS:
            raise QueryError(f"Limit must be between 1 and {self.MAX_RESULTS}")
        return value

    def _response(self, query: str, data: dict, *, truncated: bool = False) -> dict[str, Any]:
        coverage = self.graph.graph.graph.get("coverage", {})
        response = {"schema_version": 1, "query": query, "repository": str(self.repo_root),
                    "snapshot_id": self.snapshot_id, "status": self.run_status,
                    "coverage": {"scan_complete": coverage.get("scan", {}).get("complete", False),
                                 "unsupported_structure": coverage.get("unsupported_structure", [])[:50],
                                 "diagnostics": {name: rows[:20] for name, rows in coverage.get("diagnostics", {}).items()}},
                    "truncated": truncated, "data": data}
        while len(json.dumps(response, default=str).encode()) > self.MAX_RESPONSE_BYTES:
            lists = [value for value in data.values() if isinstance(value, list) and value]
            if not lists:
                raise QueryError("Response exceeds the output budget; request a smaller range")
            max(lists, key=len).pop()
            response["truncated"] = True
        return response

    def _node(self, query: str, types: set[str] | None = None) -> str:
        if not query.strip() or len(query) > 1000:
            raise QueryError("Target must contain 1-1000 characters")
        nodes = [n for n, a in self.graph.graph.nodes(data=True) if types is None or a.get("node_type") in types]
        if query in nodes:
            return query
        candidates = sorted(n for n in nodes if n.lower().endswith(query.lower()))
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            raise QueryError(f"Ambiguous target; use a qualified name: {candidates[:10]}")
        raise QueryError(f"Target not found: {query}")

    def _evidence(self, node: str, attrs: dict | None = None) -> dict[str, Any]:
        attrs = attrs if attrs is not None else self.graph.graph.nodes[node]
        path = attrs.get("source_file") or attrs.get("parent_module") or attrs.get("path")
        span = attrs.get("line_range")
        if span is None and attrs.get("line_start", 0):
            span = [attrs["line_start"], attrs.get("line_end", attrs["line_start"])]
        located = bool(path and isinstance(span, (list, tuple)) and len(span) == 2
                       and span[0] > 0 and span[1] >= span[0])
        evidence = {"path": path, "line_range": span if located else None, "location_available": located,
                "method": attrs.get("analysis_method", "static_structure"),
                "confidence": attrs.get("confidence", "syntax"), "snapshot_id": self.snapshot_id}
        if attrs.get("artifact_file"):
            evidence.update(artifact_file=attrs["artifact_file"], resource_id=attrs.get("resource_id"))
        return evidence

    def get_architecture(self, limit: int = 50) -> dict[str, Any]:
        self._refresh()
        limit = self._limit(limit)
        graph = self.graph.get_module_subgraph()
        ranks = self.graph.compute_pagerank(graph)
        modules = sorted(graph, key=lambda n: (-ranks.get(n, 0), n))
        lineage = self.graph.get_lineage_subgraph()
        return self._response("get_architecture", {
            "counts": self.graph.summary(),
            "modules": [{"path": n, "language": graph.nodes[n].get("language"),
                         "pagerank": round(ranks.get(n, 0), 6), "evidence": self._evidence(n)} for n in modules[:limit]],
            "sources": sorted(n for n, a in lineage.nodes(data=True) if a.get("node_type") == "dataset" and lineage.in_degree(n) == 0)[:limit],
            "sinks": sorted(n for n, a in lineage.nodes(data=True) if a.get("node_type") == "dataset" and lineage.out_degree(n) == 0)[:limit],
            "circular_imports": sorted(sorted(c) for c in self.graph.find_circular_dependencies(graph))[:limit],
        }, truncated=len(modules) > limit)

    def find_symbol(self, pattern: str, limit: int = 50) -> dict[str, Any]:
        self._refresh()
        limit = self._limit(limit)
        if not pattern.strip() or len(pattern) > 1000:
            raise QueryError("Search text must contain 1-1000 characters")
        matches = []
        for node, attrs in sorted(self.graph.graph.nodes(data=True)):
            if attrs.get("node_type") not in {"module", "function", "task", "column"}:
                continue
            if pattern.lower() in node.lower() or any(pattern.lower() in name.lower() for name in attrs.get("classes", [])):
                matches.append({"id": node, "type": attrs["node_type"],
                                "signature": attrs.get("signature", "")[:1000], "evidence": self._evidence(node)})
        return self._response("find_symbol", {"total": len(matches), "matches": matches[:limit]}, truncated=len(matches) > limit)

    def _walk(self, query: str, start: str, neighbors, depth: int, limit: int) -> dict[str, Any]:
        limit = self._limit(limit)
        if not 1 <= depth <= self.MAX_DEPTH:
            raise QueryError(f"Depth must be between 1 and {self.MAX_DEPTH}")
        # Transformation nodes connect datasets and code; they are traversed without
        # being reported or counted as a hop. Edge evidence already cites them.
        queue = deque([(start, 0, start)])
        seen = {start}
        paths = []
        stopped_by = None
        while queue:
            current, distance, origin = queue.popleft()
            candidates = sorted(neighbors(current), key=lambda pair: pair[0])
            if distance >= depth:
                if any(n not in seen for n, _ in candidates):
                    stopped_by = "depth"
                continue
            for neighbor, edge in candidates:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                attrs = self.graph.graph.nodes[neighbor]
                if attrs.get("node_type") == "transformation":
                    queue.appendleft((neighbor, distance, origin))
                    continue
                paths.append({"node": neighbor, "node_type": attrs.get("node_type"), "distance": distance + 1,
                              "via": origin, "edge_type": edge.get("edge_type"), "evidence": self._evidence(neighbor, edge)})
                if len(paths) >= limit or len(seen) >= self.MAX_VISITED:
                    # Conservative marker: reaching a bound may leave unexplored edges.
                    return self._response(query, {"target": start, "results": paths, "stopped_by": "limit"},
                                          truncated=True)
                queue.append((neighbor, distance + 1, neighbor))
        return self._response(query, {"target": start, "results": paths, "stopped_by": stopped_by},
                              truncated=stopped_by is not None)

    def trace_dependencies(self, target: str, direction: Literal["dependencies", "dependents"] = "dependencies",
                           depth: int = 5, limit: int = 100) -> dict[str, Any]:
        self._refresh()
        if direction not in {"dependencies", "dependents"}:
            raise QueryError("Direction must be dependencies or dependents")
        start = self._node(target, {"module", "function", "task"})
        def neighbors(node):
            g = self.graph.graph
            if g.nodes[node].get("node_type") == "task":
                # Task order: dependencies run before this task, dependents after it.
                adjacent = g.predecessors(node) if direction == "dependencies" else g.successors(node)
                for neighbor in adjacent:
                    edge = g.edges[neighbor, node] if direction == "dependencies" else g.edges[node, neighbor]
                    if edge.get("edge_type") == "SCHEDULES":
                        yield neighbor, edge
                return
            adjacent = g.successors(node) if direction == "dependencies" else g.predecessors(node)
            for neighbor in adjacent:
                edge = g.edges[node, neighbor] if direction == "dependencies" else g.edges[neighbor, node]
                if edge.get("edge_type") in {"IMPORTS", "CALLS"}:
                    yield neighbor, edge
        return self._walk("trace_dependencies", start, neighbors, depth, limit)

    def analyze_impact(self, target: str, depth: int = 10, limit: int = 100) -> dict[str, Any]:
        self._refresh()
        return self._walk("analyze_impact", self._node(target), self.graph.impact_neighbors, depth, limit)

    def trace_data_lineage(self, dataset: str, direction: Literal["upstream", "downstream"] = "upstream",
                           depth: int = 10, limit: int = 100) -> dict[str, Any]:
        self._refresh()
        if direction not in {"upstream", "downstream"}:
            raise QueryError("Direction must be upstream or downstream")
        start = self._node(dataset, {"dataset", "column"})
        def neighbors(node):
            for neighbor in self.graph.lineage_neighbors(node, direction):
                edge = self.graph.graph.edges[neighbor, node] if direction == "upstream" else self.graph.graph.edges[node, neighbor]
                yield neighbor, edge
        return self._walk("trace_data_lineage", start, neighbors, depth, limit)

    def get_code_context(self, path: str, start_line: int = 1, end_line: int = 80) -> dict[str, Any]:
        self._refresh()
        if not path or len(path) > 1000:
            raise QueryError("Path must contain 1-1000 characters")
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in path:
            raise QueryError("Use a repository-relative path with forward slashes")
        if not 1 <= start_line <= end_line or end_line - start_line >= 200:
            raise QueryError("Request 1-200 lines with positive line numbers")
        candidate = self.repo_root / relative
        for part in (candidate, *candidate.parents):
            if part == self.repo_root:
                break
            if part.is_symlink() or part.is_junction():
                raise QueryError("Code context cannot follow symlinks or junctions")
        if not candidate.resolve().is_relative_to(self.repo_root):
            raise QueryError("Path is outside the repository")
        expected = self.snapshot.get(relative.as_posix())
        if expected is None or self.graph.graph.nodes.get(relative.as_posix(), {}).get("node_type") != "module":
            raise QueryError("Path is not an indexed source file")
        from brownfield_cartographer.scanner import ScanSettings, scan_repository
        try:
            settings = ScanSettings(**self.graph.graph.graph.get("scan_settings", {}))
        except (ValueError, TypeError) as exc:
            raise QueryError("Invalid indexed scanner settings; analyze it again") from exc
        current = scan_repository(self.repo_root, settings)
        if current.fingerprints != self.snapshot:
            raise QueryError("Repository or ignore policy changed since indexing; analyze it again")
        if candidate.stat().st_size > 2 * 1024 * 1024:
            raise QueryError("Source exceeds the code context byte limit")
        with candidate.open("rb") as stream:
            content = stream.read(2 * 1024 * 1024 + 1)
        if len(content) > 2 * 1024 * 1024:
            raise QueryError("Source exceeds the code context byte limit")
        if hashlib.sha256(content).hexdigest() != expected:
            raise QueryError("Source changed since indexing; analyze the repository again")
        lines = content.decode("utf-8").splitlines()
        if start_line > len(lines):
            raise QueryError("Start line is beyond the end of the file")
        end_line = min(end_line, len(lines))
        excerpt = "\n".join(lines[start_line - 1:end_line])
        if len(excerpt.encode()) > 32_000:
            raise QueryError("Source excerpt exceeds the output budget; request fewer lines")
        return self._response("get_code_context", {"path": relative.as_posix(), "line_range": [start_line, end_line],
                                                   "content": excerpt, "method": "source_snapshot"})
