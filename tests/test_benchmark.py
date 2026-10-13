"""The accuracy benchmark must pass, and its checker must catch regressions."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

EXPECTED = {
    "status": "partial",
    "lineage": [["raw.a", "out.a"], ["raw.b", "out.b"]],
    "known_gaps": {"lineage": [{"edge": ["raw.b", "out.b"], "reason": "unsupported"}]},
}


def check(status, lineage):
    report = benchmark.score(EXPECTED, status, {"imports": set(), "functions": set(), "lineage": lineage})
    return report, benchmark.problems("fixture", report)


def test_known_gap_is_accepted_and_scored():
    report, issues = check("partial", {"raw.a -> out.a"})
    assert issues == []
    assert report["kinds"]["lineage"]["precision"] == 1.0
    assert report["kinds"]["lineage"]["recall"] == 0.5


def test_regressions_are_reported():
    _, issues = check("complete", {"raw.a -> out.b"})
    assert "fixture: status complete, expected partial" in issues
    assert "fixture lineage: false positive raw.a -> out.b" in issues
    assert "fixture lineage: missed raw.a -> out.a" in issues


def test_fixed_gap_must_be_removed_from_known_gaps():
    _, issues = check("partial", {"raw.a -> out.a", "raw.b -> out.b"})
    assert issues == ["fixture lineage: known gap now found, remove it from known_gaps: raw.b -> out.b"]


def test_import_targets_can_be_ignored():
    expected = {"status": "complete", "imports": [["a.py", "b.py"]], "ignore_import_targets": ["pkg/__init__.py"]}
    actual = {"imports": {"a.py -> b.py", "a.py -> pkg/__init__.py"}, "functions": set(), "lineage": set()}
    assert benchmark.problems("fixture", benchmark.score(expected, "complete", actual)) == []


def test_pinned_corpora_match_expectations():
    reports = {}
    for expected_file in sorted((benchmark.ROOT / "expected").glob("*.json")):
        expected = json.loads(expected_file.read_text(encoding="utf-8"))
        status, actual = benchmark.analyze(benchmark.ROOT / "corpora" / expected_file.stem)
        reports[expected_file.stem] = benchmark.score(expected, status, actual)
    assert set(reports) == {"jaffle_shop_classic", "js_ts_app", "python_etl", "python_imports", "sql_warehouse"}
    assert [issue for name, report in reports.items() for issue in benchmark.problems(name, report)] == []
