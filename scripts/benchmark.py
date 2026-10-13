"""Score analysis results against hand-written expectations for pinned corpora.

Each corpus in benchmarks/corpora/ is analyzed in a temporary copy and compared
with benchmarks/expected/<corpus>.json. Precision and recall are reported per
fact type (module imports, function definitions, dataset lineage).

--check fails on any false positive, on any miss not listed under known_gaps,
on a listed gap that is now found, or on an unexpected run status.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from brownfield_cartographer.graph.knowledge_graph import KnowledgeGraph
from brownfield_cartographer.orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parent.parent / "benchmarks"
KINDS = ("imports", "functions", "lineage", "tasks", "schedules", "runs")


def _key(fact) -> str:
    return " -> ".join(fact) if isinstance(fact, list | tuple) else fact


def _task(node: str) -> str:
    """`task:<file>::<dag>.<task>` -> `<dag>.<task>`."""
    return node.split("::", 1)[1] if node.startswith("task:") else node


def extract(index: Path) -> dict[str, set[str]]:
    kg = KnowledgeGraph()
    kg.load_index(index)
    graph = kg.graph
    facts: dict[str, set[str]] = {kind: set() for kind in KINDS}
    for source, target, data in graph.edges(data=True):
        if data.get("edge_type") == "IMPORTS":
            facts["imports"].add(_key((source, target)))
        elif data.get("edge_type") == "SCHEDULES":
            facts["schedules"].add(_key((_task(source), _task(target))))
        elif data.get("edge_type") == "RUNS":
            # A task running inline SQL points at a transformation; name it by the tables it writes.
            written = sorted(v for _, v, d in graph.out_edges(target, data=True) if d.get("edge_type") == "PRODUCES")
            facts["runs"].add(_key((_task(source), "writes " + ", ".join(written) if written else target)))
    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "task":
            facts["tasks"].add(_task(node))
    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "function":
            facts["functions"].add(node)
        if data.get("node_type") == "transformation":
            consumed = [u for u, _, d in graph.in_edges(node, data=True) if d.get("edge_type") == "CONSUMES"]
            produced = [v for _, v, d in graph.out_edges(node, data=True) if d.get("edge_type") == "PRODUCES"]
            facts["lineage"].update(_key((u, v)) for u in consumed for v in produced)
    return facts


def analyze(corpus: Path) -> tuple[str, dict[str, set[str]]]:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / corpus.name
        shutil.copytree(corpus, repo)
        with redirect_stdout(StringIO()):
            result = Orchestrator().run(str(repo))
        index = Path(result["artifact_directory"]) / "index.json"
        return result["status"], extract(index)


def score(expected: dict, status: str, actual: dict[str, set[str]]) -> dict:
    report: dict = {"status": status, "expected_status": expected["status"], "kinds": {}}
    for kind in KINDS:
        if kind not in expected:
            continue
        truth = {_key(fact) for fact in expected[kind]}
        found = set(actual[kind])
        if kind == "imports":
            ignored = set(expected.get("ignore_import_targets", []))
            found = {edge for edge in found if edge.split(" -> ")[1] not in ignored}
        found -= set(expected.get(f"ignore_{kind}", []))
        gaps = {_key(gap["edge"] if isinstance(gap, dict) else gap)
                for gap in expected.get("known_gaps", {}).get(kind, [])}
        true_positive = found & truth
        report["kinds"][kind] = {
            "expected": len(truth),
            "found": len(true_positive),
            "precision": len(true_positive) / len(found) if found else 1.0,
            "recall": len(true_positive) / len(truth) if truth else 1.0,
            "false_positives": sorted(found - truth),
            "misses": sorted(truth - found),
            "known_gaps": sorted(gaps),
        }
    return report


def problems(name: str, report: dict) -> list[str]:
    issues = []
    if report["status"] != report["expected_status"]:
        issues.append(f"{name}: status {report['status']}, expected {report['expected_status']}")
    for kind, result in report["kinds"].items():
        issues += [f"{name} {kind}: false positive {fact}" for fact in result["false_positives"]]
        misses, gaps = set(result["misses"]), set(result["known_gaps"])
        issues += [f"{name} {kind}: missed {fact}" for fact in sorted(misses - gaps)]
        issues += [f"{name} {kind}: known gap now found, remove it from known_gaps: {fact}"
                   for fact in sorted(gaps - misses)]
    return issues


def markdown(reports: dict[str, dict]) -> str:
    lines = ["| Corpus | Facts | Expected | Found | Precision | Recall | Status |",
             "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for name, report in reports.items():
        for kind, result in report["kinds"].items():
            lines.append(f"| {name} | {kind} | {result['expected']} | {result['found']} | "
                         f"{result['precision']:.2f} | {result['recall']:.2f} | {report['status']} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON")
    parser.add_argument("--check", action="store_true", help="Fail on regressions or stale known gaps")
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    reports = {}
    for expected_file in sorted((ROOT / "expected").glob("*.json")):
        expected = json.loads(expected_file.read_text(encoding="utf-8"))
        status, actual = analyze(ROOT / "corpora" / expected_file.stem)
        reports[expected_file.stem] = score(expected, status, actual)

    print(json.dumps(reports, indent=2) if args.json else markdown(reports))
    issues = [issue for name, report in reports.items() for issue in problems(name, report)]
    if args.check and issues:
        print("\n".join(issues), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
