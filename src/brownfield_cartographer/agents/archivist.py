"""
Archivist Agent — Living Context Maintainer.

Produces and maintains the system's output artifacts as living documents
that can be re-used, updated, and injected into AI coding agents for
instant architectural awareness.

Artifacts produced:
  - CODEBASE.md   — Living context file for AI agent injection
  - onboarding_brief.md — Key modules, data entry/exit points and change hotspots
  - cartography_trace.jsonl — Audit log of every analysis action
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brownfield_cartographer.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace Logger
# ---------------------------------------------------------------------------

class TraceLogger:
    """
    Structured audit logger for cartography analysis actions.

    Writes JSON-Lines to cartography_trace.jsonl with timestamp,
    agent name, action description, evidence source, and confidence.
    """

    def __init__(self, output_path: Path) -> None:
        self._path = output_path
        self._entries: list[dict[str, Any]] = []
        from uuid import uuid4
        self.run_id = uuid4().hex

    def log(
        self,
        agent: str,
        action: str,
        evidence_source: str = "static_analysis",
        confidence: float = 1.0,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record an analysis action."""
        entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "agent": agent,
            "action": action,
            "evidence_source": evidence_source,
            "confidence": confidence,
            "details": details or {},
        }
        self._entries.append(entry)

    def flush(self) -> None:
        """Write all buffered entries to the JSONL file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(entry, default=str) + "\n")
        count = len(self._entries)
        self._entries.clear()
        logger.info("Trace log flushed: %d entries → %s", count, self._path)


# ---------------------------------------------------------------------------
# CODEBASE.md Schema & Validation
# ---------------------------------------------------------------------------

# Document the CODEBASE.md schema so downstream AI agents can reliably
# parse and consume the living context file.
CODEBASE_MD_SCHEMA = {
    "required_sections": [
        {
            "heading": "Architecture Overview",
            "description": "Module/function/edge counts, dataset/transformation stats, circular dependency warnings.",
            "populated_when": "Always — derived from KnowledgeGraph.summary().",
        },
        {
            "heading": "Critical Path",
            "description": "Top 10 modules ranked by PageRank score in a markdown table.",
            "populated_when": "Surveyor has computed PageRank (top_architectural_hubs).",
        },
        {
            "heading": "Data Sources & Sinks",
            "description": "Two sub-lists: Sources (in-degree=0 datasets) and Sinks (out-degree=0 datasets).",
            "populated_when": "Hydrologist has discovered data lineage.",
        },
        {
            "heading": "Known Debt",
            "description": "Sub-sections for Circular Dependencies and Dead Code Candidates.",
            "populated_when": "Circular dependencies or dead code candidates are detected.",
        },
        {
            "heading": "High-Velocity Files",
            "description": "Markdown table of files with highest git commit count in last 30 days.",
            "populated_when": "Git history is available (not a bare/shallow clone).",
        },
    ],
    "format": "GitHub-flavored Markdown with tables, code fences, and emoji indicators.",
    "update_frequency": "Re-generated on every `cartographer analyze` run.",
}


def validate_codebase_md(content: str) -> dict[str, Any]:
    """
    Validate that a generated CODEBASE.md has all required sections populated.

    Returns a dict with:
      - valid: bool — True if all critical sections have content
      - sections: dict mapping section name → {present: bool, populated: bool}
      - warnings: list of human-readable issues
    """
    section_headings = [
        "Architecture Overview",
        "Critical Path",
        "Data Sources & Sinks",
        "Known Debt",
        "High-Velocity Files",
    ]

    results: dict[str, dict[str, bool]] = {}
    warnings: list[str] = []
    lines = content.split("\n")

    for heading in section_headings:
        # Find the section header
        header_idx = None
        for i, line in enumerate(lines):
            if heading.lower() in line.lower() and line.strip().startswith("#"):
                header_idx = i
                break

        if header_idx is None:
            results[heading] = {"present": False, "populated": False}
            warnings.append(f"Missing section: '{heading}'")
            continue

        # Check if the section has content (not just a header + placeholder)
        section_content = []
        for j in range(header_idx + 1, len(lines)):
            if lines[j].strip().startswith("## "):
                break
            stripped = lines[j].strip()
            if stripped and not stripped.startswith(">"):
                section_content.append(stripped)

        has_content = len(section_content) > 0
        is_placeholder = all(
            c.startswith("_") and c.endswith("_") for c in section_content
        ) if section_content else True

        populated = has_content and not is_placeholder
        results[heading] = {"present": True, "populated": populated}

        if not populated:
            warnings.append(
                f"Section '{heading}' is present but contains only placeholder text"
            )

    all_valid = all(r["populated"] for r in results.values())
    return {"valid": all_valid, "sections": results, "warnings": warnings}


# ---------------------------------------------------------------------------
# CODEBASE.md Generator
# ---------------------------------------------------------------------------

def generate_codebase_md(
    knowledge_graph: KnowledgeGraph,
    surveyor_results: dict[str, Any],
    hydrologist_results: dict[str, Any],
) -> str:
    """
    Generate the CODEBASE.md living context file.

    Structured for direct injection into AI coding agents to give
    instant architectural awareness. Sections:
      - Architecture Overview
      - Critical Path (top modules by PageRank)
      - Data Sources & Sinks
      - Known Debt
      - High-Velocity Files
    """
    sections: list[str] = []

    # Header
    sections.append("# CODEBASE.md — Living Architecture Context\n")
    sections.append(f"> Auto-generated by The Brownfield Cartographer on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}\n")
    sections.append("> Inject this file into any AI coding agent for instant architectural awareness.\n")

    # --- Architecture Overview ---
    graph_summary = knowledge_graph.summary()
    sections.append("## Architecture Overview\n")
    sections.append(
        f"This codebase contains **{graph_summary['modules']} source modules** "
        f"with **{graph_summary['functions']} functions** across "
        f"**{graph_summary['total_edges']} dependency edges**. "
        f"The data layer tracks **{graph_summary['datasets']} datasets** "
        f"connected by **{graph_summary['transformations']} transformations**."
    )
    if graph_summary["circular_dependencies"] > 0:
        sections.append(
            f"\n⚠️ **{graph_summary['circular_dependencies']} circular dependency group(s)** detected."
        )
    sections.append("")

    # --- Critical Path ---
    sections.append("## Critical Path (Top Modules by PageRank)\n")
    top_hubs = surveyor_results.get("top_architectural_hubs", [])
    if top_hubs:
        sections.append("| Rank | Module | PageRank Score |")
        sections.append("|------|--------|----------------|")
        for i, hub in enumerate(top_hubs[:10], 1):
            sections.append(f"| {i} | `{hub['path']}` | {hub['pagerank']:.4f} |")
    else:
        sections.append("_No PageRank data available._")
    sections.append("")

    # --- Data Sources & Sinks ---
    sections.append("## Data Sources & Sinks\n")
    sources = hydrologist_results.get("data_sources", [])
    sinks = hydrologist_results.get("data_sinks", [])

    if sources:
        sections.append("### Sources (data enters here)")
        for src in sources:
            sections.append(f"- `{src}`")
        sections.append("")

    if sinks:
        sections.append("### Sinks (data exits here)")
        for sink in sinks:
            sections.append(f"- `{sink}`")
        sections.append("")

    if not sources and not sinks:
        sections.append("_No data lineage detected._\n")

    # --- Known Debt ---
    sections.append("## Known Debt\n")
    circular_deps = surveyor_results.get("circular_dependencies", [])
    dead_code = surveyor_results.get("dead_code_candidates", [])

    debt_items = 0
    if circular_deps:
        sections.append("### Circular Dependencies")
        for cycle in circular_deps:
            sections.append(f"- {' → '.join(f'`{m}`' for m in cycle)}")
            debt_items += 1
        sections.append("")

    if dead_code:
        sections.append("### Dead Code Candidates")
        sections.append("_Modules with public symbols never imported elsewhere:_")
        for path in dead_code[:20]:
            sections.append(f"- `{path}`")
        debt_items += len(dead_code)
        sections.append("")

    if debt_items == 0:
        sections.append("_No findings from the supported checks; this is not a complete debt assessment._\n")

    # --- High-Velocity Files ---
    sections.append("## High-Velocity Files (Recent Change Hotspots)\n")
    velocity_files = surveyor_results.get("high_velocity_files", [])
    if velocity_files:
        sections.append("| File | Commits (30d) |")
        sections.append("|------|---------------|")
        for entry in velocity_files[:15]:
            sections.append(f"| `{entry['path']}` | {entry['commits_30d']} |")
    else:
        sections.append("_No recent git history available (likely a shallow clone)._")
    sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Onboarding Brief Generator
# ---------------------------------------------------------------------------

def generate_onboarding_brief(
    knowledge_graph: KnowledgeGraph,
    surveyor_results: dict[str, Any],
    hydrologist_results: dict[str, Any],
) -> str:
    """Generate onboarding_brief.md from facts in the graph; nothing is inferred."""
    sections: list[str] = []

    sections.append("# Onboarding Brief\n")
    sections.append(f"> Generated by The Brownfield Cartographer on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}.")
    sections.append("> Static facts only. Use `cartographer impact` and `cartographer lineage` for details.\n")

    graph_summary = knowledge_graph.summary()
    sections.append("## At a Glance\n")
    sections.append("| Metric | Value |")
    sections.append("|--------|-------|")
    sections.append(f"| Total modules | {graph_summary['modules']} |")
    sections.append(f"| Total functions | {graph_summary['functions']} |")
    sections.append(f"| Total datasets | {graph_summary['datasets']} |")
    sections.append(f"| Total transformations | {graph_summary['transformations']} |")
    sections.append(f"| Circular dependencies | {graph_summary['circular_dependencies']} |")
    sections.append("")

    sections.append("## Most Depended-On Modules\n")
    hubs = surveyor_results.get("top_architectural_hubs", [])[:5]
    sections.extend(f"- `{hub['path']}`" for hub in hubs)
    if not hubs:
        sections.append("_No module dependencies detected._")
    sections.append("")

    sections.append("## Data Entry and Exit Points\n")
    sources = hydrologist_results.get("data_sources", [])
    sinks = hydrologist_results.get("data_sinks", [])
    if sources:
        sections.append("### Entry points (no upstream producer found)")
        sections.extend(f"- `{src}`" for src in sources)
        sections.append("")
    if sinks:
        sections.append("### Exit points (no downstream consumer found)")
        sections.extend(f"- `{sink}`" for sink in sinks)
        sections.append("")
    if not sources and not sinks:
        sections.append("_No data lineage detected._\n")

    sections.append("## Recent Change Hotspots\n")
    velocity_files = surveyor_results.get("high_velocity_files", [])[:5]
    sections.extend(f"- `{entry['path']}` ({entry['commits_30d']} commits in 30 days)" for entry in velocity_files)
    if not velocity_files:
        sections.append("_No recent git history available._")
    sections.append("")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Archivist Agent — Public API
# ---------------------------------------------------------------------------

class Archivist:
    """
    The Archivist agent produces and maintains living context artifacts.

    Generates CODEBASE.md, onboarding_brief.md, and manages the
    cartography_trace.jsonl audit log across all analysis phases.
    """

    def __init__(self) -> None:
        self.trace_logger: TraceLogger | None = None

    def init_trace(self, cartography_dir: Path) -> TraceLogger:
        """Initialize the trace logger for a new analysis run."""
        trace_path = cartography_dir / "cartography_trace.jsonl"
        # Preserve previous run history even when a snapshot is unchanged.
        self.trace_logger = TraceLogger(trace_path)
        return self.trace_logger

    def run(
        self,
        repo_root: Path,
        knowledge_graph: KnowledgeGraph,
        surveyor_results: dict[str, Any],
        hydrologist_results: dict[str, Any],
        cartography_dir: Path,
    ) -> dict[str, Any]:
        """
        Generate all living context artifacts.

        Writes to the .cartography/ directory:
          - CODEBASE.md
          - onboarding_brief.md
          - cartography_trace.jsonl (flushed)
        """
        logger.info("Archivist: generating living context artifacts")
        start_time = time.monotonic()

        # Log the archivist's own start
        if self.trace_logger:
            self.trace_logger.log(
                agent="archivist",
                action="artifact_generation_started",
                details={"output_dir": str(cartography_dir)},
            )

        cartography_dir.mkdir(parents=True, exist_ok=True)

        # --- Generate CODEBASE.md ---
        codebase_md = generate_codebase_md(
            knowledge_graph,
            surveyor_results,
            hydrologist_results,
        )

        # Empty sections are legitimate results; record them without warning.
        validation = validate_codebase_md(codebase_md)
        for warning in validation["warnings"]:
            logger.debug("CODEBASE.md validation: %s", warning)

        codebase_path = cartography_dir / "CODEBASE.md"
        codebase_path.write_text(codebase_md, encoding="utf-8")
        logger.info("CODEBASE.md written → %s", codebase_path)

        if self.trace_logger:
            self.trace_logger.log(
                agent="archivist",
                action="codebase_md_generated",
                details={
                    "path": str(codebase_path),
                    "size_bytes": len(codebase_md),
                    "validation": validation,
                },
            )

        # --- Generate onboarding_brief.md ---
        brief_md = generate_onboarding_brief(
            knowledge_graph,
            surveyor_results,
            hydrologist_results,
        )
        brief_path = cartography_dir / "onboarding_brief.md"
        brief_path.write_text(brief_md, encoding="utf-8")
        logger.info("onboarding_brief.md written → %s", brief_path)

        if self.trace_logger:
            self.trace_logger.log(
                agent="archivist",
                action="onboarding_brief_generated",
                details={"path": str(brief_path), "size_bytes": len(brief_md)},
            )

        # --- Flush trace log ---
        elapsed = time.monotonic() - start_time
        if self.trace_logger:
            self.trace_logger.log(
                agent="archivist",
                action="artifact_generation_complete",
                details={"elapsed_seconds": round(elapsed, 2)},
            )
            self.trace_logger.flush()

        return {
            "artifacts_written": [
                str(codebase_path),
                str(brief_path),
            ],
            "codebase_md_size": len(codebase_md),
            "brief_md_size": len(brief_md),
            "elapsed_seconds": round(elapsed, 2),
        }
