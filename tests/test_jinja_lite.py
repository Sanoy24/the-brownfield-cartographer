"""Bounded rendering of literal `set` lists and `for` loops in dbt models."""
from brownfield_cartographer.analyzers.dag_config_parser import render_simple_model
from brownfield_cartographer.analyzers.jinja_lite import render


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
