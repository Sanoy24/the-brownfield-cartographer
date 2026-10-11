"""Shared output roots must preserve each checkout's graph and checkpoint."""
from pathlib import Path

from brownfield_cartographer.orchestrator import Orchestrator, _derive_repo_name
from brownfield_cartographer.query_api import QueryEngine


def test_same_named_repositories_keep_independent_indexes(tmp_path):
    output = tmp_path / "results"
    repos = [tmp_path / owner / "repo" for owner in ("first", "second")]
    for repo, symbol in zip(repos, ("alpha", "beta")):
        repo.mkdir(parents=True)
        (repo / "source.py").write_text(f"def {symbol}():\n    pass\n")

    first = Orchestrator().run(str(repos[0]), output_dir=output)
    first_index = Path(first["artifact_directory"])
    original = (first_index / "index.json").read_bytes()
    checkpoint = (first_index / "state.json").read_bytes()

    second = Orchestrator().run(str(repos[1]), output_dir=output)
    second_index = Path(second["artifact_directory"])
    assert first_index != second_index
    assert (first_index / "index.json").read_bytes() == original
    assert (first_index / "state.json").read_bytes() == checkpoint
    for repo, index, symbol in zip(repos, (first_index, second_index), ("alpha", "beta")):
        assert QueryEngine.open(repo, index).find_symbol(symbol)["data"]["total"] == 1
        cached = Orchestrator().run(str(repo), output_dir=output)
        assert cached["status"] == "no_changes"
        assert Path(cached["artifact_directory"]) == index


def test_equivalent_checkout_paths_use_the_same_output(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(tmp_path)
    assert _derive_repo_name("repo", isolate_local=True) == _derive_repo_name(
        str(repo), isolate_local=True,
    )
    assert _derive_repo_name(str(repo)) == "repo"
    assert _derive_repo_name("https://github.com/team/repo.git", isolate_local=True) == "team--repo"


def test_new_output_preserves_legacy_directory(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "source.py").write_text("VALUE = 1\n")
    output = tmp_path / "results"
    legacy = output / ".cartography" / "repo"
    legacy.mkdir(parents=True)
    (legacy / "index.json").write_text("legacy index sentinel")
    result = Orchestrator().run(str(repo), output_dir=output)
    assert result["status"] == "complete"
    assert Path(result["artifact_directory"]) != legacy
    assert (legacy / "index.json").read_text() == "legacy index sentinel"
