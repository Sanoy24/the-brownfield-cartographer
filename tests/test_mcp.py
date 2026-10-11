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
                assert all(t.annotations.readOnlyHint and not t.annotations.openWorldHint for t in listing.tools)
                architecture = await session.call_tool("get_architecture", {})
                assert not architecture.isError
                assert architecture.structuredContent["data"]["counts"]["modules"] == 1
                response = await session.call_tool("find_symbol", {"pattern": "run"})
                assert not response.isError
                assert response.structuredContent["data"]["matches"][0]["id"] == "source.py::run"
                context = await session.call_tool("get_code_context", {"path": "source.py", "start_line": 1, "end_line": 2})
                assert context.structuredContent["data"]["content"] == "def run():\n    return 1"
                refused = await session.call_tool("get_code_context", {"path": "../secret.py"})
                assert refused.isError
    asyncio.run(asyncio.wait_for(check(), timeout=30))
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
