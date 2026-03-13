"""
Navigator Agent — LangGraph-powered Query Interface.

Provides an interactive query interface for exploring the codebase
knowledge graph. Implements four tools that allow both exploratory
investigation and precise structured querying:

  - find_implementation(concept)  — Semantic search for code matching a concept
  - trace_lineage(dataset, direction) — Graph traversal for data lineage
  - blast_radius(module_path)     — Downstream dependency impact analysis
  - explain_module(path)          — Module purpose and metadata retrieval
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Annotated, Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Tool Implementations (framework-independent logic)
# ---------------------------------------------------------------------------


def tool_find_implementation(
    knowledge_graph: KnowledgeGraph,
    concept: str,
) -> list[dict[str, Any]]:
    """
    Search the knowledge graph for modules whose purpose statement,
    function names, or class names match the given concept.

    Returns matches sorted by relevance with file paths and metadata.
    """
    concept_lower = concept.lower()
    results: list[dict[str, Any]] = []

    for node_id, attrs in knowledge_graph.graph.nodes(data=True):
        if attrs.get("node_type") != "module":
            continue

        score = 0.0
        match_reasons: list[str] = []

        # Check purpose statement
        purpose = attrs.get("purpose_statement", "")
        if purpose and concept_lower in purpose.lower():
            score += 3.0
            match_reasons.append("purpose_statement")

        # Check domain cluster
        domain = attrs.get("domain_cluster", "")
        if domain and concept_lower in domain.lower():
            score += 2.0
            match_reasons.append("domain_cluster")

        # Check function names
        functions = attrs.get("public_functions", [])
        for fn in functions:
            if concept_lower in fn.lower():
                score += 1.5
                match_reasons.append(f"function:{fn}")

        # Check class names
        classes = attrs.get("classes", [])
        for cls in classes:
            if concept_lower in cls.lower():
                score += 1.5
                match_reasons.append(f"class:{cls}")

        # Check file path
        if concept_lower in node_id.lower():
            score += 1.0
            match_reasons.append("file_path")

        if score > 0:
            loc = attrs.get("lines_of_code", 0) or 0
            analysis_method = "static"
            if purpose:
                analysis_method = "llm+static"
            results.append(
                {
                    "path": node_id,
                    "score": round(score, 2),
                    "match_reasons": match_reasons,
                    "language": attrs.get("language", "unknown"),
                    "purpose": purpose or "N/A",
                    "lines_of_code": loc,
                    "evidence": {
                        "source_file": node_id,
                        "line_range": [1, loc] if loc else None,
                        "analysis_method": analysis_method,
                    },
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]


def tool_trace_lineage(
    knowledge_graph: KnowledgeGraph,
    dataset: str,
    direction: str = "upstream",
) -> list[dict[str, Any]]:
    """
    Trace data lineage upstream or downstream from a dataset node.

    Args:
        dataset: Name of the dataset to trace from.
        direction: "upstream" (what feeds this) or "downstream" (what depends on this).

    Returns:
        List of lineage path entries with node types and edge metadata.
    """
    graph = knowledge_graph.graph

    # Find the best matching node
    target_node = _fuzzy_find_node(knowledge_graph, dataset, "dataset")
    if not target_node:
        target_node = _fuzzy_find_node(knowledge_graph, dataset)
    if not target_node:
        available = _list_available_nodes(knowledge_graph, "dataset", limit=10)
        return [{
            "error": f"Dataset '{dataset}' not found in graph",
            "suggestion": "Try one of these known datasets" if available else "No datasets in graph",
            "available_datasets": available,
        }]

    visited: set[str] = set()
    lineage: list[dict[str, Any]] = []
    queue: deque[tuple[str, int]] = deque([(target_node, 0)])

    while queue:
        current, depth = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        node_attrs = dict(graph.nodes.get(current, {}))
        # Derive line range for evidence
        line_start = node_attrs.get("line_start")
        line_end = node_attrs.get("line_end")
        if line_start is None or line_end is None:
            # For module nodes, approximate as full file span when LOC is known
            if node_attrs.get("node_type") == "module":
                loc = node_attrs.get("lines_of_code", 0) or 0
                if loc:
                    line_start, line_end = 1, loc
        analysis_method = "static"
        if node_attrs.get("node_type") == "module" and node_attrs.get(
            "purpose_statement"
        ):
            analysis_method = "llm+static"
        lineage.append(
            {
                "node": current,
                "node_type": node_attrs.get("node_type", "unknown"),
                "depth": depth,
                "transformation_type": node_attrs.get("transformation_type"),
                "source_file": node_attrs.get("source_file"),
                "evidence": {
                    "source_file": node_attrs.get("source_file") or current,
                    "line_range": (
                        [line_start, line_end]
                        if line_start is not None and line_end is not None
                        else None
                    ),
                    "analysis_method": analysis_method,
                },
            }
        )

        # Traverse in the requested direction
        if direction == "upstream":
            neighbors = graph.predecessors(current)
        else:
            neighbors = graph.successors(current)

        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))

    return lineage


def tool_blast_radius(
    knowledge_graph: KnowledgeGraph,
    module_path: str,
) -> list[dict[str, Any]]:
    """
    Find all downstream dependents of a given module.

    Uses BFS to discover every node affected if the given module's
    interface or data changed. Works across both module and lineage graphs.

    Returns list of affected nodes with edge type and distance.
    """
    graph = knowledge_graph.graph

    # Fuzzy-find the node
    target_node = _fuzzy_find_node(knowledge_graph, module_path)
    if not target_node:
        available = _list_available_nodes(knowledge_graph, "module", limit=10)
        return [{
            "error": f"Module '{module_path}' not found in graph",
            "suggestion": "Try one of these known modules" if available else "No modules in graph",
            "available_modules": available,
        }]

    visited: set[str] = set()
    affected: list[dict[str, Any]] = []
    queue: deque[tuple[str, int]] = deque([(target_node, 0)])

    while queue:
        current, depth = queue.popleft()
        for successor in graph.successors(current):
            if successor not in visited:
                visited.add(successor)
                edge_data = graph.edges.get((current, successor), {})
                succ_attrs = graph.nodes[successor]

                # Derive line range for evidence
                line_start = succ_attrs.get("line_start")
                line_end = succ_attrs.get("line_end")
                if line_start is None or line_end is None:
                    if succ_attrs.get("node_type") == "module":
                        loc = succ_attrs.get("lines_of_code", 0) or 0
                        if loc:
                            line_start, line_end = 1, loc
                analysis_method = "static"
                if succ_attrs.get("node_type") == "module" and succ_attrs.get(
                    "purpose_statement"
                ):
                    analysis_method = "llm+static"
                affected.append(
                    {
                        "node": successor,
                        "node_type": succ_attrs.get("node_type", "unknown"),
                        "distance": depth + 1,
                        "edge_type": edge_data.get("edge_type", "unknown"),
                        "via": current,
                        "evidence": {
                            "source_file": succ_attrs.get("source_file") or successor,
                            "line_range": (
                                [line_start, line_end]
                                if line_start is not None and line_end is not None
                                else None
                            ),
                            "analysis_method": analysis_method,
                        },
                    }
                )
                queue.append((successor, depth + 1))

    affected.sort(key=lambda x: x["distance"])
    return affected


def tool_explain_module(
    knowledge_graph: KnowledgeGraph,
    path: str,
) -> dict[str, Any]:
    """
    Return a comprehensive explanation of a module.

    Includes purpose statement, domain cluster, complexity score,
    imports, functions, and structural metadata.
    """
    target_node = _fuzzy_find_node(knowledge_graph, path, "module")
    if not target_node:
        available = _list_available_nodes(knowledge_graph, "module", limit=10)
        return {
            "error": f"Module '{path}' not found in graph",
            "suggestion": "Try one of these known modules" if available else "No modules in graph",
            "available_modules": available,
        }

    attrs = dict(knowledge_graph.graph.nodes.get(target_node, {}))

    # Appropriate line range as entire file when LOC is known
    loc = attrs.get("lines_of_code", 0) or 0
    line_start = 1 if loc else None
    line_end = loc if loc else None

    analysis_method = "static"
    if attrs.get("purpose_statement"):
        analysis_method = "llm+static"

    # Get import relationships
    imports_from: list[str] = []
    imported_by: list[str] = []
    for pred in knowledge_graph.graph.predecessors(target_node):
        edge = knowledge_graph.graph.edges.get((pred, target_node), {})
        if edge.get("edge_type") == "IMPORTS":
            imported_by.append(pred)
    for succ in knowledge_graph.graph.successors(target_node):
        edge = knowledge_graph.graph.edges.get((target_node, succ), {})
        if edge.get("edge_type") == "IMPORTS":
            imports_from.append(succ)

    return {
        "path": target_node,
        "language": attrs.get("language", "unknown"),
        "purpose_statement": attrs.get(
            "purpose_statement", "Not generated — run with LLM configured"
        ),
        "domain_cluster": attrs.get("domain_cluster", "unclassified"),
        "complexity_score": attrs.get("complexity_score", 0.0),
        "lines_of_code": attrs.get("lines_of_code", 0),
        "is_dead_code_candidate": attrs.get("is_dead_code_candidate", False),
        "public_functions": attrs.get("public_functions", []),
        "classes": attrs.get("classes", []),
        "imports_from": imports_from,
        "imported_by": imported_by,
        "change_velocity_30d": attrs.get("change_velocity_30d", 0),
        "evidence": {
            "source_file": target_node,
            "line_range": (
                [line_start, line_end]
                if line_start is not None and line_end is not None
                else None
            ),
            "analysis_method": analysis_method,
        },
    }


def _fuzzy_find_node(
    knowledge_graph: KnowledgeGraph,
    query: str,
    node_type: str | None = None,
) -> str | None:
    """Find the best matching node in the graph by name fuzzy matching."""
    graph = knowledge_graph.graph
    query_lower = query.lower().strip()

    # Exact match first
    if query in graph:
        return query

    # Try suffix/prefix matching
    candidates: list[tuple[str, int]] = []
    for node_id in graph.nodes():
        if node_type:
            if graph.nodes[node_id].get("node_type") != node_type:
                continue

        node_lower = node_id.lower()
        if query_lower == node_lower:
            return node_id
        if node_lower.endswith(query_lower) or query_lower in node_lower:
            priority = 0 if node_lower.endswith(query_lower) else 1
            candidates.append((node_id, priority))

    if candidates:
        candidates.sort(key=lambda x: (x[1], len(x[0])))
        return candidates[0][0]

    return None


def _list_available_nodes(
    knowledge_graph: KnowledgeGraph,
    node_type: str,
    limit: int = 10,
) -> list[str]:
    """Return a sample of available node IDs of the given type for error suggestions."""
    nodes = [
        n for n, d in knowledge_graph.graph.nodes(data=True)
        if d.get("node_type") == node_type
    ]
    nodes.sort(key=len)  # shorter names first for readability
    return nodes[:limit]


# ---------------------------------------------------------------------------
# LangGraph Agent
# ---------------------------------------------------------------------------


def build_navigator_agent(knowledge_graph: KnowledgeGraph) -> Any:
    """
    Build the LangGraph Navigator agent with four tools.

    Uses a ReAct architecture with LangGraph's prebuilt agent.
    Falls back to a simple tool-dispatch loop if LangGraph is unavailable.
    """
    from src.agents.semanticist import LLMConfig, _create_chat_model

    config = LLMConfig.from_env()
    llm = _create_chat_model(config)

    if not llm:
        logger.warning("No LLM configured — Navigator will use direct tool dispatch")
        return None

    try:
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent

        @tool
        def find_implementation(concept: str) -> str:
            """Search for code implementing a concept. Returns matching modules with scores."""
            results = tool_find_implementation(knowledge_graph, concept)
            return json.dumps(results, indent=2)

        @tool
        def trace_lineage(dataset: str, direction: str = "upstream") -> str:
            """Trace data lineage upstream or downstream from a dataset. Direction: 'upstream' or 'downstream'."""
            results = tool_trace_lineage(knowledge_graph, dataset, direction)
            return json.dumps(results, indent=2)

        @tool
        def blast_radius_tool(module_path: str) -> str:
            """Find all downstream dependents that would break if a module changed."""
            results = tool_blast_radius(knowledge_graph, module_path)
            return json.dumps(results, indent=2)

        @tool
        def explain_module(path: str) -> str:
            """Get comprehensive explanation of a module including purpose, complexity, and dependencies."""
            result = tool_explain_module(knowledge_graph, path)
            return json.dumps(result, indent=2)

        tools = [find_implementation, trace_lineage, blast_radius_tool, explain_module]

        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt="You are a codebase intelligence assistant. Use the available tools to answer questions about the codebase architecture, data lineage, and module purposes. Always cite specific file paths and evidence.",
        )

        return agent

    except ImportError as exc:
        logger.warning(
            "LangGraph not available: %s — falling back to direct dispatch", exc
        )
        return None


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


def run_interactive(knowledge_graph: KnowledgeGraph) -> None:
    """
    Start an interactive query REPL for exploring the knowledge graph.

    Supports both natural language queries (via LangGraph agent) and
    direct tool commands (fallback when no LLM is available).
    """
    agent = build_navigator_agent(knowledge_graph)

    console.print(
        Panel(
            "[bold]🧭 Brownfield Cartographer — Navigator[/bold]\n\n"
            "Query the codebase knowledge graph interactively.\n\n"
            "[dim]Commands:[/dim]\n"
            "  [cyan]find[/cyan] <concept>        — Search for implementations\n"
            "  [cyan]lineage[/cyan] <dataset>      — Trace data lineage (upstream)\n"
            "  [cyan]lineage-down[/cyan] <dataset> — Trace data lineage (downstream)\n"
            "  [cyan]blast[/cyan] <module>         — Show blast radius\n"
            "  [cyan]explain[/cyan] <path>         — Explain a module\n"
            "  [cyan]summary[/cyan]               — Show graph summary\n"
            "  [cyan]quit[/cyan]                  — Exit\n\n"
            "[dim]Or type any natural language question (requires LLM).[/dim]",
            title="Navigator",
            border_style="cyan",
        )
    )

    while True:
        try:
            query = console.input("\n[bold cyan]navigator>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        # Direct tool dispatch
        if query.lower() == "summary":
            summary = knowledge_graph.summary()
            console.print_json(json.dumps(summary, indent=2))
            continue

        if query.lower().startswith("find "):
            concept = query[5:].strip()
            results = tool_find_implementation(knowledge_graph, concept)
            _display_results("Implementation Search", results)
            continue

        if query.lower().startswith("lineage-down "):
            dataset = query[13:].strip()
            results = tool_trace_lineage(knowledge_graph, dataset, "downstream")
            _display_results("Downstream Lineage", results)
            continue

        if query.lower().startswith("lineage "):
            dataset = query[8:].strip()
            results = tool_trace_lineage(knowledge_graph, dataset, "upstream")
            _display_results("Upstream Lineage", results)
            continue

        if query.lower().startswith("blast "):
            module = query[6:].strip()
            results = tool_blast_radius(knowledge_graph, module)
            _display_results("Blast Radius", results)
            continue

        if query.lower().startswith("explain "):
            path = query[8:].strip()
            result = tool_explain_module(knowledge_graph, path)
            console.print_json(json.dumps(result, indent=2))
            continue

        # Natural language via LangGraph agent
        if agent:
            try:
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": query}]}
                )
                # Extract the final message
                messages = response.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    content = (
                        last_msg.content
                        if hasattr(last_msg, "content")
                        else str(last_msg)
                    )
                    console.print(
                        Panel(
                            Markdown(content),
                            title="Navigator Response",
                            border_style="green",
                        )
                    )
                else:
                    console.print("[yellow]No response generated.[/yellow]")
            except Exception as exc:
                console.print(f"[red]Agent error:[/red] {exc}")
                console.print(
                    "[dim]Try using direct commands (find, lineage, blast, explain).[/dim]"
                )
        else:
            console.print(
                "[yellow]No LLM configured for natural language queries.[/yellow]\n"
                "[dim]Use direct commands: find, lineage, blast, explain, summary[/dim]"
            )


def _display_results(title: str, results: list[dict[str, Any]]) -> None:
    """Pretty-print tool results using Rich."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    if isinstance(results, list) and results and "error" in results[0]:
        console.print(f"[red]{results[0]['error']}[/red]")
        return

    console.print(
        Panel(
            json.dumps(results, indent=2),
            title=f"📊 {title} ({len(results)} results)",
            border_style="blue",
        )
    )
