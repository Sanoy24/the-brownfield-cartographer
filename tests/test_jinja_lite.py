"""Bounded rendering of literal `set` lists and `for` loops in dbt models."""
from brownfield_cartographer.analyzers.dag_config_parser import render_simple_model
from brownfield_cartographer.analyzers.jinja_lite import parse_macros, render


def _squash(text):
    return " ".join(text.split())


def test_set_list_for_loop_and_whitespace_control():
    text = (
        "{% set methods = ['card', 'coupon'] %}\n"
        "select\n"
        "    {% for m in methods -%}\n"
        "    sum(case when method = '{{ m }}' then amount end) as {{ m }}_amount,\n"
        "    {% endfor -%}\n"
        "    sum(amount) as total\n"
        "from payments"
    )
    assert _squash(render(text)) == (
        "select sum(case when method = 'card' then amount end) as card_amount, "
        "sum(case when method = 'coupon' then amount end) as coupon_amount, sum(amount) as total from payments")


def test_loop_last_separators_literal_lists_nesting_and_string_sets():
    text = (
        "{% set table = 'events' %}"
        "select {% for a in ['x', 'y'] %}{% for b in [1, 2] %}{{ a }}_{{ b }}"
        "{% if not loop.last %}, {% endif %}{% endfor %}{% if not loop.last %}, {% endif %}{% endfor %} "
        "from {{ table }}"
    )
    assert render(text) == "select x_1, x_2, y_1, y_2 from events"
    assert render("{% for c in ['a', 'b'] %}{% if loop.first %}{{ c }}{% else %}, {{ c }}{% endif %}{% endfor %}") == "a, b"


def test_anything_else_is_not_rendered():
    for text in (
        "select {{ dbt_utils.star(ref('x')) }} from t",          # macro call
        "{% set cols = get_columns() %}select 1",                 # computed set
        "{% for c in cols %}{{ c }}{% endfor %}",                 # undefined list
        "{% if target.name == 'prod' %}select 1{% endif %}",     # other conditions
        "{% for c in ['a'] %}{{ c | upper }}{% endfor %}",        # filters
        "{% for c in ['a'] %}select 1",                           # unclosed block
        "{% macro m() %}x{% endmacro %}",
    ):
        assert render(text) is None, text


def test_dbt_models_with_literal_loops_now_render():
    model = (
        "{% set methods = ['card', 'coupon'] %}\n"
        "select order_id, {% for m in methods %}sum({{ m }}) as {{ m }}_amount{% if not loop.last %},{% endif %}"
        "{% endfor %} from {{ ref('stg_payments') }} group by 1"
    )
    assert _squash(render_simple_model(model)) == (
        "select order_id, sum(card) as card_amount,sum(coupon) as coupon_amount from stg_payments group by 1")
    assert render_simple_model("select {{ dbt_utils.star(ref('x')) }} from t") is None


MACROS = """
{% macro clean(column_name) %}
CASE WHEN TRIM({{ column_name }}) = "" THEN NULL ELSE TRIM({{ column_name }}) END
{% endmacro %}
{% macro pass_status() %}"PASS"{% endmacro %}
{%- macro ranked(table, order_by, partition_by=None) -%}
select *, rank() over ({% if partition_by %}partition by {{ partition_by }} {% endif %}order by {{ order_by }}) as r from {{ table }}
{%- endmacro %}
{% macro nested(c) %}{{ clean(c) }}{% endmacro %}
{% macro uses_method(c) %}{{ c.replace('a', 'b') }}{% endmacro %}
"""


def test_project_macros_expand_with_literal_and_variable_arguments():
    macros = parse_macros(MACROS)
    assert set(macros) == {"clean", "pass_status", "ranked", "nested", "uses_method"}
    text = ("select {{ clean('name') }} as name, {{ pass_status() }} as status, "
            "{% for c in ['a', 'b'] %}{{ nested(c) }} as {{ c }}{% if not loop.last %}, {% endif %}{% endfor %} from t")
    assert _squash(render(text, macros)) == _squash(
        'select CASE WHEN TRIM(name) = "" THEN NULL ELSE TRIM(name) END as name, "PASS" as status, '
        'CASE WHEN TRIM(a) = "" THEN NULL ELSE TRIM(a) END as a, CASE WHEN TRIM(b) = "" THEN NULL ELSE TRIM(b) END as b from t')
    assert render("{{ ranked('events', order_by='ts') }}", macros) == "select *, rank() over (order by ts) as r from events"
    assert render("{{ ranked('events', 'ts', partition_by='id') }}", macros) == (
        "select *, rank() over (partition by id order by ts) as r from events")


def test_macros_outside_the_subset_are_not_expanded():
    macros = parse_macros(MACROS + "{% macro clean(x) %}{{ x }}{% endmacro %}")
    assert "clean" not in macros                                   # defined twice: ambiguous
    for text in ("{{ uses_method('x') }}",                           # Python method call in the body
                 "{{ dbt_utils.star(ref('x')) }}",                   # package macro
                 "{{ ranked() }}",                                   # missing required argument
                 "{{ pass_status(1) }}",                             # too many arguments
                 "{{ clean(some_column) }}"):                        # undefined variable argument
        assert render(text, macros) is None, text
    recursive = parse_macros("{% macro loop_forever(x) %}{{ loop_forever(x) }}{% endmacro %}")
    assert render("{{ loop_forever(1) }}", recursive) is None


def test_dbt_models_expand_project_macros(tmp_path):
    from brownfield_cartographer.graph.knowledge_graph import KnowledgeGraph
    from brownfield_cartographer.orchestrator import Orchestrator
    files = {
        "dbt_project.yml": "name: shop\nversion: '1.0'\nprofile: shop\n",
        "macros/clean.sql": "{% macro clean(column_name) %}TRIM({{ column_name }}){% endmacro %}\n",
        "macros/latest.sql": "{% macro latest(c) %}select {{ c }} from {{ ref('stg') }}{% endmacro %}\n",
        "models/stg.sql": "select {{ clean('name') }} as name, id from {{ source('raw', 'users') }}\n",
        "models/latest_names.sql": "{{ latest('name') }}\n",
    }
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8")
    result = Orchestrator().run(str(tmp_path))
    graph = KnowledgeGraph()
    graph.load_index(tmp_path / ".cartography" / tmp_path.name / "index.json")
    edges = {(u, v) for u, v, d in graph.graph.edges(data=True) if d.get("edge_type") == "DERIVES"}
    assert {("raw.users::name", "stg::name"), ("raw.users::id", "stg::id"),
            ("stg::name", "latest_names::name")} <= edges
    assert result["hydrologist"]["column_lineage"]["skipped_templates"] == 0


def test_a_dbt_project_is_traced_in_the_dialect_that_parses_most_models(tmp_path):
    from brownfield_cartographer.analyzers.dag_config_parser import parse_dbt_project
    files = {
        "dbt_project.yml": "name: shop\nversion: '1.0'\nprofile: shop\n",
        # Backtick identifiers and `* EXCEPT` only parse as BigQuery among the candidates here.
        "models/a.sql": "select `id`, * except (secret) from {{ source('raw', 'a') }}\n",
        "models/b.sql": "select * except (secret) from {{ source('raw', 'b') }}\n",
        # Valid in several dialects, but "%Y%m%d" is a string only in BigQuery; elsewhere a column.
        "models/c.sql": 'select parse_date("%Y%m%d", day) as day from {{ ref(\'a\') }}\n',
    }
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8")
    result = parse_dbt_project(tmp_path)
    [model_c] = [t for t in result["transformations"] if t.target_datasets == ["c"]]
    assert model_c.column_lineage == [{"target": "day", "sources": ["a::day"], "kind": "derived"}]
