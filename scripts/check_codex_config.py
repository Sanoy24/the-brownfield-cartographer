"""Check a session-only Codex MCP configuration without editing host settings.

Add --launch to open an interactive Codex session with the checked overrides.
The default only validates index scope and asks `codex mcp get` to parse config;
it does not start a model session or prove host tool execution.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

from brownfield_cartographer.query_api import QueryEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Previously analyzed repository")
    parser.add_argument("--index-dir", type=Path, help="Artifact directory for custom analysis output")
    parser.add_argument("--launch", action="store_true", help="Open an interactive Codex session after the check")
    args = parser.parse_args()
    codex = shutil.which("codex")
    if codex is None:
        parser.error("Codex CLI is not on PATH")
    if importlib.util.find_spec("mcp") is None:
        parser.error("Install the MCP extra first: uv sync --extra mcp")
    try:
        engine = QueryEngine.open(args.repo, args.index_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    name = "brownfield_smoke"
    server_args = ["-I", "-m", "brownfield_cartographer.mcp_server", "--repo", engine.repo_root.as_posix()]
    if args.index_dir is not None:
        server_args += ["--index-dir", args.index_dir.resolve().as_posix()]
    settings = {
        "command": Path(sys.executable).as_posix(), "args": server_args,
        "cwd": engine.repo_root.as_posix(), "startup_timeout_sec": 60, "tool_timeout_sec": 60,
    }
    command = [codex, "-C", str(engine.repo_root)]
    for key, value in settings.items():
        command += ["-c", f"mcp_servers.{name}.{key}={json.dumps(value)}"]
    result = subprocess.run([*command, "mcp", "get", name, "--json"],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    if result.returncode:
        parser.error(f"Codex rejected the session configuration: {result.stderr.strip()}")
    configuration = json.loads(result.stdout)
    transport = configuration.get("transport", {})
    if any(transport.get(key) != settings[key] for key in ("command", "args", "cwd")):
        parser.error("Codex did not retain the expected scoped stdio configuration")
    print(f"Codex accepted session-only server {name} for {engine.repo_root} (index status: {engine.run_status}).")
    if args.launch:
        subprocess.run(command, cwd=engine.repo_root, check=True)
    else:
        print("No host settings were changed and no model session was started. Add --launch for the interactive check.")


if __name__ == "__main__":
    main()
