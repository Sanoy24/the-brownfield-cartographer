"""Install a built wheel into a fresh environment and exercise its public CLI.

Run after `uv build`. No editable installation or source-tree imports are used.
Use --offline to resolve dependencies only from the existing uv cache.
Use --mcp to additionally install the MCP extra and check its stdio entry point.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

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
assert importlib.metadata.version("brownfield-cartographer") == sys.argv[1]
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


MCP_PROBE = r'''
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from brownfield_cartographer.query_api import QueryEngine

for optional in ("langchain_core", "langgraph", "sklearn"):
    assert importlib.util.find_spec(optional) is None, optional

repo = Path.cwd() / "fixture"
engine = QueryEngine.open(repo)
entry = Path(sys.executable).parent / ("cartographer-mcp.exe" if sys.platform == "win32" else "cartographer-mcp")
requests = {
    "get_architecture": ({}, engine.get_architecture()),
    "find_symbol": ({"pattern": "run"}, engine.find_symbol("run")),
    "trace_dependencies": ({"target": "consumer.py"}, engine.trace_dependencies("consumer.py")),
    "analyze_impact": ({"target": "producer.py"}, engine.analyze_impact("producer.py")),
    "trace_data_lineage": ({"dataset": "analytics.orders"}, engine.trace_data_lineage("analytics.orders")),
    "get_code_context": ({"path": "consumer.py", "start_line": 1, "end_line": 1}, engine.get_code_context("consumer.py", 1, 1)),
}

def files():
    return {path.relative_to(repo): path.read_bytes() for path in repo.rglob("*") if path.is_file()}

async def check():
    before = files()
    parameters = StdioServerParameters(command=str(entry), args=["--repo", str(repo)],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"})
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            listing = await session.list_tools()
            assert {tool.name for tool in listing.tools} == set(requests)
            for tool in listing.tools:
                assert tool.annotations.readOnlyHint
                assert not tool.annotations.destructiveHint
                assert not tool.annotations.openWorldHint
                assert tool.outputSchema is not None
            for name, (arguments, expected) in requests.items():
                print(f"Checking installed MCP tool: {name}", flush=True)
                result = await session.call_tool(name, arguments)
                if result.isError or result.structuredContent != expected:
                    print(json.dumps({"tool": name, "actual": result.structuredContent,
                                      "expected": expected, "errors": [item.model_dump() for item in result.content]},
                                     default=str), file=sys.stderr, flush=True)
                assert not result.isError, (name, result)
                assert result.structuredContent == expected, name
            for name, arguments in (
                ("get_code_context", {"path": "../private.py"}),
                ("get_architecture", {"limit": 0}),
                ("trace_data_lineage", {"dataset": "analytics.orders", "direction": "sideways"}),
            ):
                assert (await session.call_tool(name, arguments)).isError, name
            assert files() == before, "Read-only MCP tools modified repository/index files"

            # Simulate an external failed run without replacing the graph.
            summary = engine.index_dir / "analysis_summary.json"
            original = summary.read_bytes()
            try:
                summary.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
                changed = files()
                response = await session.call_tool("get_architecture", {})
                assert not response.isError
                assert response.structuredContent["status"] == "failed"
                assert files() == changed, "Status refresh modified artifacts"
            finally:
                summary.write_bytes(original)
            response = await session.call_tool("get_architecture", {})
            assert response.structuredContent["status"] == "complete"
    assert files() == before

asyncio.run(asyncio.wait_for(check(), timeout=60))
print("Installed MCP extra: six stdio tools match the API, reject invalid requests, refresh status and leave files unchanged.")
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="Wheel to install; defaults to the current project version in dist/")
    parser.add_argument("--offline", action="store_true", help="Use only cached dependencies")
    parser.add_argument("--mcp", action="store_true", help="Also validate the installed MCP extra over stdio")
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
        if args.mcp:
            requirement = f"{project['name']}[mcp] @ {wheel.as_uri()}"
            run([*uv, "pip", "install", "--strict", "--python", str(python), requirement])
            probe.write_text(MCP_PROBE, encoding="utf-8")
            run([str(python), "-I", "-X", "utf8", str(probe)])


if __name__ == "__main__":
    main()
