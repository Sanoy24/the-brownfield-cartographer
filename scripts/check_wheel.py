"""Install a built wheel into a fresh environment and exercise its public CLI.

Run after `uv build`. No editable installation or source-tree imports are used.
Use --offline to resolve dependencies only from the existing uv cache.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


PROBE = r'''
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

import brownfield_cartographer
from brownfield_cartographer.orchestrator import Orchestrator
from brownfield_cartographer.query_api import QueryEngine

package = Path(brownfield_cartographer.__file__).resolve().parent
assert package.is_relative_to(Path(sys.prefix).resolve()), package
assert (package / "py.typed").is_file()
assert importlib.metadata.version("the-brownfield-cartographer") == sys.argv[1]
for optional in ("langchain_core", "langgraph", "sklearn", "mcp"):
    assert importlib.util.find_spec(optional) is None, optional

def deny_network(*args, **kwargs):
    raise AssertionError("Installed core attempted network access")
socket.create_connection = deny_network
socket.socket.connect = deny_network
socket.getaddrinfo = deny_network

repo = Path.cwd() / "fixture"
repo.mkdir()
(repo / "producer.py").write_text("VALUE = 1\n", encoding="utf-8")
(repo / "consumer.py").write_text("import producer\ndef run():\n    return producer.VALUE\n", encoding="utf-8")
(repo / "orders.sql").write_text("CREATE TABLE analytics.orders AS SELECT * FROM raw.orders;\n", encoding="utf-8")

result = Orchestrator().run(str(repo))
assert result["status"] == "complete", result
engine = QueryEngine.open(repo)
assert engine.analyze_impact("producer.py")["data"]["results"][0]["node"] == "consumer.py"
assert engine.find_symbol("run")["data"]["matches"][0]["id"] == "consumer.py::run"
assert "raw.orders" in {row["node"] for row in engine.trace_data_lineage("analytics.orders")["data"]["results"]}
assert engine.get_code_context("consumer.py", 1, 1)["data"]["content"] == "import producer"

scripts = Path(sys.executable).parent
cli = scripts / ("cartographer.exe" if sys.platform == "win32" else "cartographer")
mcp = scripts / ("cartographer-mcp.exe" if sys.platform == "win32" else "cartographer-mcp")
def command(*args, expected=0):
    result = subprocess.run([str(cli), *args], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == expected, (result.stdout, result.stderr)
    return result.stdout

assert "analyze" in command("--help")
assert json.loads(command("analyze", str(repo), "--json"))["status"] == "no_changes"
assert json.loads(command("find", "run", "--repo", str(repo), "--json"))["data"]["total"] == 1
assert json.loads(command("impact", "producer.py", "--repo", str(repo), "--json"))["data"]["results"][0]["node"] == "consumer.py"
assert json.loads(command("overview", str(repo), "--json"))["status"] == "complete"
assert "raw.orders" in {row["node"] for row in json.loads(command("lineage", "analytics.orders", "--repo", str(repo), "--json"))["data"]["results"]}
assert json.loads(command("overview", str(Path.cwd()), "--json", expected=1))["error"]

dbt = Path.cwd() / "dbt-fixture"
(dbt / "models").mkdir(parents=True)
(dbt / "target").mkdir()
(dbt / ".gitignore").write_text("target/\n", encoding="utf-8")
(dbt / "models/orders.sql").write_text("select * from {{ source('raw', 'orders') }}\n", encoding="utf-8")
manifest = {"metadata": {"dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v12.json"},
    "nodes": {"model.shop.orders": {"unique_id": "model.shop.orders", "resource_type": "model",
        "original_file_path": "models/orders.sql", "depends_on": {"nodes": ["source.shop.raw.orders"]}}},
    "sources": {"source.shop.raw.orders": {"unique_id": "source.shop.raw.orders", "resource_type": "source",
        "original_file_path": "models/sources.yml"}}}
(dbt / "target/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
assert json.loads(command("analyze", str(dbt), "--dbt-manifest", "target/manifest.json", "--json"))["status"] == "complete"
lineage = json.loads(command("lineage", "model.shop.orders", "--repo", str(dbt), "--json"))["data"]["results"]
assert "source.shop.raw.orders" in {row["node"] for row in lineage}
assert all(row["evidence"]["method"] == "dbt_manifest" and not row["evidence"]["location_available"] for row in lineage)

missing_extra = subprocess.run([str(mcp), "--repo", str(repo)], capture_output=True, text=True, encoding="utf-8", timeout=30)
assert missing_extra.returncode == 1
assert "Install the mcp extra" in missing_extra.stderr
print("Installed wheel: core API, CLI, dbt adapter, entry points, typing metadata and optional dependency boundaries passed.")
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="Wheel to install; defaults to the current project version in dist/")
    parser.add_argument("--offline", action="store_true", help="Use only cached dependencies")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    wheel = (args.wheel or root / "dist" / (
        f"{project['name'].replace('-', '_')}-{project['version']}-py3-none-any.whl"
    )).resolve()
    if not wheel.is_file():
        parser.error("Wheel not found; run uv build first or specify --wheel")

    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_SYSTEM_PYTHON",
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        env.pop(key, None)
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    scratch = root / ".cartography"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wheel-check-", dir=scratch) as temporary:
        work = Path(temporary)
        uv = ["uv", "--no-config", *(["--offline"] if args.offline else [])]
        def run(command):
            subprocess.run(command, cwd=work, env=env, check=True, timeout=300)
        run([*uv, "venv", "--no-project", "--python", sys.executable, str(work / "env")])
        python = work / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run([*uv, "pip", "install", "--strict", "--python", str(python), str(wheel)])
        probe = work / "probe.py"
        probe.write_text(PROBE, encoding="utf-8")
        run([str(python), "-I", "-X", "utf8", str(probe), project["version"]])


if __name__ == "__main__":
    main()
