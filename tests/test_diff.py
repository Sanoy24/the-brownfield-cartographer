"""`cartographer diff`: map changed lines to graph nodes and report what they affect."""
import json
import subprocess

import pytest
from typer.testing import CliRunner

from brownfield_cartographer.cli import app
from brownfield_cartographer.diff import change_impact


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _write(repo, files):
    for path, text in files.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


DAG = (
    "from airflow import DAG\n"
    "from airflow.operators.bash import BashOperator\n"
    "with DAG('nightly') as dag:\n"
    "    stage = BashOperator(task_id='stage', bash_command='echo')\n"
    "    report = BashOperator(task_id='report', bash_command='echo')\n"
    "    stage >> report\n"
)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _write(tmp_path, {
        "sql/stage.sql": (
            "CREATE TABLE s.orders AS SELECT id, amount FROM raw.orders;\n"
            "CREATE TABLE s.customers AS SELECT id, name FROM raw.customers;\n"
        ),
        "sql/mart.sql": "CREATE TABLE m.revenue AS SELECT id, amount * 100 AS cents FROM s.orders;\n",
        "lib/util.py": "def helper():\n    return 1\n\n\ndef unused():\n    return 2\n",
        "app.py": "from lib.util import helper\n\n\ndef main():\n    return helper()\n",
        "lib/__init__.py": "",
        "old.py": "VALUE = 1\n",
        "dags/nightly.py": DAG,
    })
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "switch", "-q", "-c", "feature")
    return tmp_path


def test_changed_sql_statement_reports_only_its_downstream(repo):
    _write(repo, {"sql/stage.sql": (
        "CREATE TABLE s.orders AS SELECT id, amount, region FROM raw.orders;\n"
        "CREATE TABLE s.customers AS SELECT id, name FROM raw.customers;\n"
    )})
    report = change_impact(repo, "main")
    assert [c["node"] for c in report["changed"]] == ["s.orders"]
    assert {a["node"] for a in report["affected"]} == {"m.revenue"}
    assert report["affected"][0]["evidence"]["path"] == "sql/mart.sql"


def test_changed_function_reaches_importers_but_untouched_functions_do_not(repo):
    _write(repo, {"lib/util.py": "def helper():\n    return 10\n\n\ndef unused():\n    return 2\n"})
    report = change_impact(repo, "main")
    assert {c["node"] for c in report["changed"]} == {"lib/util.py", "lib/util.py::helper"}
    assert "app.py" in {a["node"] for a in report["affected"]}


def test_changed_task_reaches_downstream_tasks_and_committed_changes_count(repo):
    _write(repo, {"dags/nightly.py": DAG.replace("task_id='stage', bash_command='echo'",
                                                 "task_id='stage', bash_command='echo staged'")})
    _git(repo, "commit", "-q", "-am", "change stage")
    report = change_impact(repo, "main")
    assert "task:dags/nightly.py::nightly.stage" in {c["node"] for c in report["changed"]}
    assert "task:dags/nightly.py::nightly.report" in {a["node"] for a in report["affected"]}


def test_deleted_files_and_no_changes_are_reported(repo):
    (repo / "old.py").unlink()
    report = change_impact(repo, "main")
    assert report["deleted_files"] == ["old.py"]
    _git(repo, "checkout", "-q", "--", "old.py")
    assert change_impact(repo, "main")["changed"] == []


def test_cli_markdown_and_json(repo):
    _write(repo, {"sql/stage.sql": (
        "CREATE TABLE s.orders AS SELECT id, amount, region FROM raw.orders;\n"
        "CREATE TABLE s.customers AS SELECT id, name FROM raw.customers;\n"
    )})
    markdown = CliRunner().invoke(app, ["diff", "--repo", str(repo), "--base", "main"])
    assert markdown.exit_code == 0, markdown.output
    assert "m.revenue" in markdown.stdout and "sql/mart.sql" in markdown.stdout
    as_json = CliRunner().invoke(app, ["diff", "--repo", str(repo), "--base", "main", "--json"])
    assert json.loads(as_json.stdout)["affected"][0]["node"] == "m.revenue"


def test_schema_yaml_change_reports_configured_model_but_not_its_downstream(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _write(tmp_path, {
        "dbt_project.yml": "name: shop\nversion: '1.0'\nprofile: shop\n",
        "models/stg_orders.sql": "select id from {{ source('raw', 'orders') }}\n",
        "models/orders.sql": "select id from {{ ref('stg_orders') }}\n",
        "models/schema.yml": "version: 2\nmodels:\n  - name: stg_orders\n    description: staged\n",
        "package-lock.yml": "packages: []\n",
    })
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _write(tmp_path, {
        "models/schema.yml": "version: 2\nmodels:\n  - name: stg_orders\n    description: staged orders\n",
        "package-lock.yml": "packages: [x]\n",
    })
    report = change_impact(tmp_path, "main")
    assert [c["node"] for c in report["changed"]] == ["models/schema.yml"]
    assert [a["node"] for a in report["affected"]] == ["stg_orders"]
    assert report["unmapped_files"] == ["package-lock.yml"]


def test_deleted_module_reports_importers_that_still_use_it(repo):
    (repo / "lib/util.py").unlink()
    report = change_impact(repo, "main")
    removed = {c["node"] for c in report["changed"] if c["removed"]}
    assert {"lib/util.py", "lib/util.py::helper"} <= removed
    # The import no longer resolves in the current graph; it is found in the merge base.
    assert [(a["node"], a["edge_type"]) for a in report["affected"]] == [("app.py", "IMPORTS")]


def test_deleted_sql_file_reports_consumers_of_the_tables_it_wrote(repo):
    (repo / "sql/stage.sql").unlink()
    report = change_impact(repo, "main")
    assert report["deleted_files"] == ["sql/stage.sql"]
    assert {"s.orders", "s.customers"} <= {c["node"] for c in report["changed"]}
    assert "m.revenue" in {a["node"] for a in report["affected"]}
