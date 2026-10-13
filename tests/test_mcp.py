"""Real stdio protocol smoke test; no model or remote host is involved."""
import asyncio
import importlib.util
import os
import sys

import pytest

from brownfield_cartographer.orchestrator import Orchestrator


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="Optional mcp extra is not installed")
def test_read_only_stdio_tools(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    (tmp_path / "source.py").write_text("def run():\n    return 1\n")
    Orchestrator().run(str(tmp_path))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    async def check():
        parameters = StdioServerParameters(command=sys.executable,
            args=["-m", "brownfield_cartographer.mcp_server", "--repo", str(tmp_path)],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                listing = await session.list_tools()
                assert {t.name for t in listing.tools} == {
                    "get_architecture", "find_symbol", "trace_dependencies", "analyze_impact",
                    "trace_data_lineage", "get_code_context"}
                assert all(t.annotations.read_only_hint and not t.annotations.open_world_hint for t in listing.tools)
                architecture = await session.call_tool("get_architecture", {})
                assert not architecture.is_error
                assert architecture.structured_content["data"]["counts"]["modules"] == 1
                response = await session.call_tool("find_symbol", {"pattern": "run"})
                assert not response.is_error
                assert response.structured_content["data"]["matches"][0]["id"] == "source.py::run"
                context = await session.call_tool("get_code_context", {"path": "source.py", "start_line": 1, "end_line": 2})
                assert context.structured_content["data"]["content"] == "def run():\n    return 1"
                refused = await session.call_tool("get_code_context", {"path": "../secret.py"})
                assert refused.is_error
                # The reason must reach the assistant so it can correct the call.
                assert "repository-relative" in refused.content[0].text
                missing = await session.call_tool("analyze_impact", {"target": "nowhere.py"})
                assert missing.is_error and "Target not found: nowhere.py" in missing.content[0].text
                # SDK v2 runs sync handlers on worker threads; concurrent calls must stay consistent.
                calls = [("find_symbol", {"pattern": "run"}), ("get_architecture", {}),
                         ("analyze_impact", {"target": "source.py"})] * 4
                results = await asyncio.gather(*(session.call_tool(name, args) for name, args in calls))
                assert not any(result.is_error for result in results)
                assert {r.structured_content["snapshot_id"] for r in results} == {response.structured_content["snapshot_id"]}
    asyncio.run(asyncio.wait_for(check(), timeout=30))
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize("installed,expected", [
    (None, "Install the mcp extra"),
    ("1.30.0", "mcp 1.30.0 is installed, but Cartographer requires mcp>=2.3,<3"),
])
def test_unusable_mcp_sdk_is_reported(monkeypatch, tmp_path, installed, expected):
    from brownfield_cartographer import mcp_server

    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)
    monkeypatch.setattr(mcp_server, "_installed_mcp_version", lambda: installed)
    with pytest.raises(RuntimeError, match=expected):
        mcp_server.create_server(tmp_path)


def test_server_starts_before_analysis_and_picks_up_the_index(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("Optional mcp extra is not installed")
    (tmp_path / "source.py").write_text("def run():\n    return 1\n")

    async def check():
        parameters = StdioServerParameters(command=sys.executable,
            args=["-m", "brownfield_cartographer.mcp_server", "--repo", str(tmp_path)],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                assert len((await session.list_tools()).tools) == 6
                before = await session.call_tool("find_symbol", {"pattern": "run"})
                assert before.is_error and "cartographer analyze" in before.content[0].text
                Orchestrator().run(str(tmp_path))
                after = await session.call_tool("find_symbol", {"pattern": "run"})
                assert not after.is_error
                assert after.structured_content["data"]["matches"][0]["id"] == "source.py::run"
    asyncio.run(asyncio.wait_for(check(), timeout=60))
