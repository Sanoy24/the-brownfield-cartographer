"""Python call edges: resolved statically to functions defined in the repository."""
from brownfield_cartographer.agents.surveyor import Surveyor
from brownfield_cartographer.analyzers.python_calls import extract_calls
from brownfield_cartographer.graph.knowledge_graph import KnowledgeGraph


def _calls(tmp_path, files):
    for path, text in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    graph = KnowledgeGraph()
    Surveyor().run(tmp_path, graph)
    return {(u, v): d for u, v, d in graph.graph.edges(data=True) if d.get("edge_type") == "CALLS"}


def test_extract_calls_records_caller_reference_and_line():
    source = (
        "import pkg.tools as t\n"
        "from pkg.core import load as fetch\n"
        "def run():\n"
        "    fetch()\n"
        "    t.clean()\n"
        "    print('x')\n"
    )
    assert extract_calls(source) == [
        ("run", "pkg.core", "load", 4),
        ("run", "pkg.tools", "clean", 5),
    ]


def test_calls_resolve_local_imported_aliased_module_and_method_calls(tmp_path):
    edges = _calls(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/core.py": "def load():\n    return 1\n",
        "pkg/tools.py": "def clean():\n    return 2\n",
        "app.py": (
            "import pkg.tools\n"
            "from pkg import core\n"
            "from pkg.core import load as fetch\n"
            "\n"
            "def helper():\n"
            "    return fetch()\n"
            "\n"
            "def main():\n"
            "    helper()\n"
            "    core.load()\n"
            "    pkg.tools.clean()\n"
            "\n"
            "class Job:\n"
            "    def run(self):\n"
            "        return self.step()\n"
            "\n"
            "    def step(self):\n"
            "        return main()\n"
        ),
    })
    assert set(edges) == {
        ("app.py::helper", "pkg/core.py::load"),
        ("app.py::main", "app.py::helper"),
        ("app.py::main", "pkg/core.py::load"),
        ("app.py::main", "pkg/tools.py::clean"),
        ("app.py::Job.run", "app.py::Job.step"),
        ("app.py::Job.step", "app.py::main"),
    }
    evidence = edges["app.py::main", "app.py::helper"]
    assert (evidence["source_file"], evidence["line_range"]) == ("app.py", [9, 9])


def test_unresolvable_and_shadowed_calls_are_not_guessed(tmp_path):
    edges = _calls(tmp_path, {
        "lib.py": "def go():\n    return 1\n",
        "app.py": (
            "import os\n"
            "from lib import go\n"
            "\n"
            "def run(go=None):\n"
            "    return go()\n"            # parameter shadows the import
            "\n"
            "def other(obj):\n"
            "    obj.go()\n"               # unknown receiver
            "    getattr(obj, 'go')()\n"   # dynamic
            "    os.path.join('a')\n"      # external module
            "    missing()\n"              # undefined name
            "\n"
            "def later():\n"
            "    go = make()\n"            # local assignment shadows
            "    return go()\n"
        ),
    })
    assert edges == {}
