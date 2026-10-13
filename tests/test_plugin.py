"""The Claude Code plugin must point at the same release as the package."""
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_plugin_version_and_server_pin_match_the_package():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert _json("plugin/.claude-plugin/plugin.json")["version"] == version
    server = _json("plugin/.mcp.json")["mcpServers"]["cartographer"]
    assert server["command"] == "uvx"
    assert server["args"] == ["--from", f"brownfield-cartographer[mcp]=={version}", "cartographer-mcp",
                              "--repo", "${CLAUDE_PROJECT_DIR}"]


def test_marketplace_lists_the_plugin_directory():
    marketplace = _json(".claude-plugin/marketplace.json")
    [entry] = marketplace["plugins"]
    assert entry["source"] == "./plugin"
    assert entry["name"] == _json("plugin/.claude-plugin/plugin.json")["name"] == "brownfield-cartographer"


def test_skill_has_name_and_trigger_description():
    text = (ROOT / "plugin" / "skills" / "data-impact" / "SKILL.md").read_text(encoding="utf-8")
    header = text.split("---")[1]
    assert "name: data-impact" in header
    assert "description:" in header and "impact" in header
