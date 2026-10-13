"""JavaScript/TypeScript structure: imports, resolution, functions, impact and syntax errors."""
import pytest

from brownfield_cartographer.orchestrator import Orchestrator
from brownfield_cartographer.query_api import QueryEngine

SOURCES = {
    "src/core.ts": (
        "export function helper(): number { return 1 }\n"
        "export const arrow = (x: number) => x * 2;\n"
        "export class Engine {\n"
        "  run(): number { return helper() }\n"
        "  static create() { return new Engine() }\n"
        "}\n"
        "export default function () { return 0 }\n"
        "function* gen() {}\n"
    ),
    # TypeScript ESM style: the ".js" specifier names the compiled output of core.ts.
    "src/util.js": "import { helper } from './core.js'\nexport const double = () => helper() * 2\n",
    "src/types.ts": "export interface Shape { area(): number }\n",
    "src/shapes.ts": (
        "import type { Shape } from './types'\n"
        "import helpers = require('./lib')\n"
        "export const unit: Shape | null = null\n"
    ),
    "src/index.ts": "export { double } from './util'\nexport * from './core'\n",
    "src/lib/index.js": "module.exports = {}\n",
    "src/app.tsx": (
        "import React from 'react'\n"
        "import lib from './lib'\n"
        "const lazy = import('./core')\n"
        "export const App = () => <div>{String(lib)}</div>\n"
    ),
    "legacy.cjs": (
        "const util = require('./src/util')\n"
        "const name = 'dynamic'\n"
        "require(name)\n"
        "// require('./src/core')\n"
        "const text = \"import x from './src/core'\"\n"
    ),
}


@pytest.fixture
def js_repo(tmp_path):
    for path, text in SOURCES.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    result = Orchestrator().run(str(tmp_path))
    assert result["status"] == "complete", result
    return tmp_path


def _edges(engine, edge_type):
    graph = engine.graph.graph
    return {(u, v) for u, v, d in graph.edges(data=True) if d.get("edge_type") == edge_type}


def test_relative_imports_resolve_and_external_or_dynamic_ones_do_not(js_repo):
    engine = QueryEngine.open(js_repo)
    assert _edges(engine, "IMPORTS") == {
        ("src/util.js", "src/core.ts"),
        ("src/shapes.ts", "src/types.ts"),
        ("src/shapes.ts", "src/lib/index.js"),
        ("src/index.ts", "src/util.js"),
        ("src/index.ts", "src/core.ts"),
        ("src/app.tsx", "src/lib/index.js"),
        ("src/app.tsx", "src/core.ts"),
        ("legacy.cjs", "src/util.js"),
    }


def test_functions_methods_and_arrow_functions_are_extracted(js_repo):
    graph = QueryEngine.open(js_repo).graph.graph
    functions = {n for n, d in graph.nodes(data=True) if d.get("node_type") == "function"}
    assert functions == {
        "src/core.ts::helper",
        "src/core.ts::arrow",
        "src/core.ts::Engine.run",
        "src/core.ts::Engine.create",
        "src/core.ts::default",
        "src/core.ts::gen",
        "src/util.js::double",
        "src/app.tsx::App",
    }
    assert graph.nodes["src/core.ts::Engine.run"]["line_start"] == 4


def test_impact_and_evidence_follow_reverse_imports(js_repo):
    engine = QueryEngine.open(js_repo)
    impact = engine.analyze_impact("src/core.ts")["data"]["results"]
    assert {row["node"] for row in impact} >= {"src/util.js", "src/index.ts", "src/app.tsx", "legacy.cjs"}
    direct = next(row for row in impact if row["node"] == "src/util.js")
    assert direct["evidence"]["line_range"] == [1, 1]


def test_js_and_ts_are_no_longer_listed_as_unsupported(js_repo):
    coverage = QueryEngine.open(js_repo).get_architecture()["coverage"]
    assert coverage["unsupported_structure"] == []


def test_syntax_errors_make_coverage_partial(tmp_path):
    (tmp_path / "ok.ts").write_text("export const a = 1\n", encoding="utf-8")
    (tmp_path / "broken.ts").write_text("export function (\n", encoding="utf-8")
    result = Orchestrator().run(str(tmp_path))
    assert result["status"] == "partial"
    diagnostics = QueryEngine.open(tmp_path).get_architecture()["coverage"]["diagnostics"]
    rows = [row for stage in diagnostics.values() for row in stage]
    assert any(row.get("path") == "broken.ts" and "line 1" in row["error"] for row in rows), rows


def test_dead_code_candidates_use_resolved_imports(js_repo):
    graph = QueryEngine.open(js_repo).graph.graph
    assert graph.nodes["src/app.tsx"]["is_dead_code_candidate"] is True
    assert graph.nodes["src/core.ts"]["is_dead_code_candidate"] is False
    assert graph.nodes["src/util.js"]["is_dead_code_candidate"] is False


def test_tsconfig_path_aliases_resolve_per_package(tmp_path):
    files = {
        "tsconfig.base.json": '{"compilerOptions": {"baseUrl": ".", "paths": {"@shared/*": ["libs/shared/*"]}}}',
        "apps/web/tsconfig.json": (
            '// JSONC: comments and trailing commas are allowed\n'
            '{"extends": "../../tsconfig.base.json",\n'
            ' "compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["src/*"], "~config": ["src/config.ts"],},},}\n'
        ),
        "apps/web/src/app.ts": (
            "import { log } from '@/utils/log'\n"
            "import config from '~config'\n"
            "import { format } from '@shared/format'\n"
            "import react from 'react'\n"
        ),
        "apps/web/src/utils/log.ts": "export const log = () => 1\n",
        "apps/web/src/config.ts": "export default {}\n",
        "libs/shared/format.ts": "export const format = () => 1\n",
        # No own `paths`: inherits the base aliases, resolved against the base config's baseUrl.
        "apps/api/tsconfig.json": '{"extends": "../../tsconfig.base"}',
        "apps/api/server.ts": "import { format } from '@shared/format'\n",
    }
    for path, text in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    Orchestrator().run(str(tmp_path))
    edges = _edges(QueryEngine.open(tmp_path), "IMPORTS")
    # apps/web declares its own `paths`, which replace the inherited ones (TypeScript semantics),
    # so '@shared/format' does not resolve there; 'react' is a package.
    assert edges == {
        ("apps/web/src/app.ts", "apps/web/src/utils/log.ts"),
        ("apps/web/src/app.ts", "apps/web/src/config.ts"),
        ("apps/api/server.ts", "libs/shared/format.ts"),
    }
