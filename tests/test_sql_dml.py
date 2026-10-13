"""UPDATE, DELETE and MERGE write their target table from the tables they read."""
import pytest

from brownfield_cartographer.analyzers.sql_lineage import sql_statements


def _lineage(sql):
    return [(statement["sources"], statement["target"]) for statement in sql_statements(sql, "job.sql")]


@pytest.mark.parametrize("sql,expected", [
    ("UPDATE marts.t SET n = s.n FROM raw.s AS s JOIN raw.u AS u ON u.id = s.id WHERE marts.t.id = s.id",
     [(["raw.s", "raw.u"], "marts.t")]),
    ("UPDATE marts.t SET x = (SELECT max(v) FROM raw.s) WHERE id IN (SELECT id FROM marts.t)",
     [(["raw.s"], "marts.t")]),
    ("DELETE FROM marts.t WHERE id IN (SELECT id FROM raw.s)", [(["raw.s"], "marts.t")]),
    ("DELETE FROM marts.t USING raw.s WHERE marts.t.id = raw.s.id", [(["raw.s"], "marts.t")]),
    ("MERGE INTO marts.t AS t USING raw.s AS s ON t.id = s.id "
     "WHEN MATCHED THEN UPDATE SET n = s.n WHEN NOT MATCHED THEN INSERT (id, n) VALUES (s.id, s.n)",
     [(["raw.s"], "marts.t")]),
    ("MERGE INTO marts.t USING (SELECT * FROM raw.s JOIN raw.u USING (id)) AS src "
     "ON marts.t.id = src.id WHEN MATCHED THEN DELETE",
     [(["raw.s", "raw.u"], "marts.t")]),
    ("WITH c AS (SELECT * FROM raw.s) UPDATE marts.t SET n = c.n FROM c WHERE marts.t.id = c.id",
     [(["raw.s"], "marts.t")]),
])
def test_dml_reads_flow_into_the_written_table(sql, expected):
    assert _lineage(sql) == expected


def test_dml_without_other_tables_writes_without_sources():
    assert _lineage("UPDATE marts.t SET n = 1; DELETE FROM marts.stale") == [([], "marts.t"), ([], "marts.stale")]


def test_statements_without_table_lineage_are_skipped_quietly():
    script = ("BEGIN; DROP TABLE a.old; TRUNCATE TABLE a.t; ALTER TABLE a.t ADD COLUMN c INT; "
              "CREATE INDEX i ON a.t (c); CREATE SCHEMA s; SET search_path = s; GRANT SELECT ON a.t TO r; "
              "ANALYZE a.t; COMMIT; ROLLBACK;")
    errors = []
    assert sql_statements(script, "job.sql", errors=errors) == []
    assert errors == []


def test_statements_that_may_move_data_remain_unsupported():
    errors = []
    assert sql_statements("COPY a.t FROM 'f.csv'", "job.sql", errors=errors) == []
    assert errors and "copy" in errors[0]["error"]
