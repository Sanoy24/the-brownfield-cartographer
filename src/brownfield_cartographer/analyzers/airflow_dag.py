"""Static Airflow DAG structure: tasks, scheduling order, and what each task runs.

Tasks are scheduling units, not datasets. A task is identified as
`task:<file>::<dag_id>.<task_id>`. Supported forms:

- DAGs from `with DAG(...)`, `@dag`, and `dag=<variable>` arguments.
- Tasks from operator or sensor calls with a literal `task_id`, and from
  `@task` functions (including `@task(task_id=...)` and `@task.<kind>`).
- Order from `>>`/`<<` (evaluated as Airflow does: each expression returns
  its right-hand side), lists, `set_downstream`/`set_upstream`, `chain()`,
  `cross_downstream()`, and TaskFlow calls that pass one task's result to another.
- What a task runs: `python_callable`, the `@task` function itself, a `.sql`
  file or inline SQL in `sql=`, and `python <script>.py` in `bash_command`.

Tasks created in loops or with computed ids are reported, not guessed.
"""

from __future__ import annotations

import ast
import posixpath
import re

from brownfield_cartographer.analyzers import embedded_sql
from brownfield_cartographer.analyzers.sql_lineage import sql_statements
from brownfield_cartographer.models.schemas import (
    TransformationNode,
    TransformationType,
)

_PYTHON_SCRIPT = re.compile(r"\bpython3?\s+(\S+\.py)\b")


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _name(node.func)
    return None


def _literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _is_airflow(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "airflow" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "airflow":
            return True
    return False


def _span(node: ast.AST) -> tuple[int, int]:
    line = getattr(node, "lineno", 1)
    return line, getattr(node, "end_lineno", None) or line


class _DagVisitor:
    def __init__(self, tree: ast.Module, relative_path: str, known_paths: set[str], dialect: str | None):
        self.path = relative_path
        self.known = known_paths
        self.dialect = dialect
        self.tasks: dict[str, dict] = {}
        self.variables: dict[str, list[str]] = {}
        self.taskflow: dict[str, str] = {}
        self.dag_variables: dict[str, str] = {}
        self.created: dict[int, str | None] = {}
        self.schedules: dict[tuple[str, str], tuple[int, int]] = {}
        self.runs: dict[tuple[str, str], tuple[int, int]] = {}
        self.transformations: list[tuple[str, TransformationNode]] = []
        self.diagnostics: list[dict] = []
        self.search_paths: list[str] = []
        # Qualified-name prefix of the enclosing @dag function, matching indexed nested functions.
        self.prefix = ""
        self.functions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.imports = self._imports(tree)
        dags = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _name(n) == "DAG"]
        dag_functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and any(_name(d) == "dag" for d in n.decorator_list)]
        ids = {self._dag_id(n, report=False) for n in dags} | {self._decorated_dag_id(n, report=False) for n in dag_functions}
        self.default_dag = next(iter(ids)) if len(ids) == 1 else "_"

    # -- identities -------------------------------------------------------
    def _resolve_dag_id(self, node: ast.expr, report: bool) -> str:
        """A literal id, the `Path(__file__).stem` idiom, or `_` for an id computed at runtime."""
        literal = _literal(node)
        if literal:
            return literal
        if (isinstance(node, ast.Attribute) and node.attr == "stem" and isinstance(node.value, ast.Call)
                and _name(node.value.func) == "Path" and node.value.args
                and isinstance(node.value.args[0], ast.Name) and node.value.args[0].id == "__file__"):
            return posixpath.splitext(posixpath.basename(self.path))[0]
        if report:
            self._report(node, "Computed dag_id is not resolved; its tasks are grouped under '_'")
        return "_"

    def _dag_id(self, call: ast.Call, report: bool = True) -> str:
        paths = _keyword(call, "template_searchpath")
        values: list[ast.expr | None] = list(paths.elts) if isinstance(paths, (ast.List, ast.Tuple)) else [paths]
        if report:
            self.search_paths.extend(v for v in (_literal(item) for item in values) if v)
        given = call.args[0] if call.args else _keyword(call, "dag_id")
        return self._resolve_dag_id(given, report) if given is not None else "_"

    def _decorated_dag_id(self, function: ast.FunctionDef | ast.AsyncFunctionDef, report: bool = True) -> str:
        decorator = next(d for d in function.decorator_list if _name(d) == "dag")
        given = _keyword(decorator, "dag_id") if isinstance(decorator, ast.Call) else None
        # Without dag_id, Airflow names the DAG after the decorated function.
        return self._resolve_dag_id(given, report) if given is not None else function.name

    def _task_node(self, dag_id: str, task_id: str, operator: str, node: ast.AST) -> str:
        node_id = f"task:{self.path}::{dag_id}.{task_id}"
        self.tasks.setdefault(node_id, {"id": node_id, "dag_id": dag_id, "task_id": task_id,
                                        "operator": operator, "lines": _span(node)})
        return node_id

    # -- imports and file resolution --------------------------------------
    def _imports(self, tree: ast.Module) -> dict[str, tuple[str, str | None]]:
        """Local name -> (module dotted path, imported attribute or None)."""
        names: dict[str, tuple[str, str | None]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for alias in node.names:
                    names[alias.asname or alias.name] = (node.module, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names[alias.asname or alias.name.split(".")[0]] = (alias.name if alias.asname else alias.name.split(".")[0], None)
        return names

    def _module_path(self, dotted: str) -> str | None:
        base = dotted.replace(".", "/")
        for candidate in (f"{base}.py", f"{base}/__init__.py"):
            if candidate in self.known:
                return candidate
            matches = [p for p in self.known if p.endswith("/" + candidate)]
            if len(matches) == 1:
                return matches[0]
        return None

    def _file(self, value: str) -> str | None:
        value = value.strip().lstrip("./")
        directory = posixpath.dirname(self.path)
        candidates = [posixpath.normpath(posixpath.join(directory, value)), value]
        candidates += [posixpath.normpath(posixpath.join(p.strip("/"), value)) for p in self.search_paths]
        found = next((c for c in candidates if c in self.known), None)
        if found:
            return found
        matches = [p for p in self.known if p.endswith("/" + value)]
        return matches[0] if len(matches) == 1 else None

    def _callable_target(self, node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Name):
            if node.id in self.functions:
                return f"{self.path}::{node.id}"
            if node.id in self.imports and self.imports[node.id][1]:
                module, attribute = self.imports[node.id]
                path = self._module_path(module)
                return f"{path}::{attribute}" if path else None
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in self.imports:
            module, attribute = self.imports[node.value.id]
            path = self._module_path(f"{module}.{attribute}" if attribute else module)
            return f"{path}::{node.attr}" if path else None
        return None

    # -- diagnostics ------------------------------------------------------
    def _report(self, node: ast.AST, message: str, incomplete: bool = False) -> None:
        self.diagnostics.append({"path": self.path, "line": getattr(node, "lineno", 1),
                                 "error": message, "incomplete": incomplete})

    # -- tasks ------------------------------------------------------------
    def _operator_task(self, call: ast.Call, dag_id: str, in_loop: bool) -> str | None:
        if id(call) in self.created:
            return self.created[id(call)]
        operator = _name(call.func) or "operator"
        task_id = _literal(_keyword(call, "task_id"))
        dag_variable = _keyword(call, "dag")
        if isinstance(dag_variable, ast.Name) and dag_variable.id in self.dag_variables:
            dag_id = self.dag_variables[dag_variable.id]
        node_id = None
        if in_loop or task_id is None:
            self._report(call, f"Dynamic task ({operator}) created in a loop or with a computed task_id is not recorded")
        else:
            node_id = self._task_node(dag_id, task_id, operator, call)
            self._task_runs(node_id, call)
        self.created[id(call)] = node_id
        return node_id

    def _task_runs(self, task: str, call: ast.Call) -> None:
        span = _span(call)
        target = self._callable_target(_keyword(call, "python_callable"))
        if target:
            self.runs[(task, target)] = span
        command = _literal(_keyword(call, "bash_command"))
        if command:
            for script in _PYTHON_SCRIPT.findall(command):
                found = self._file(script)
                if found:
                    self.runs[(task, found)] = span
        sql = _keyword(call, "sql")
        items: list[ast.expr | None] = list(sql.elts) if isinstance(sql, (ast.List, ast.Tuple)) else [sql]
        for item in items:
            text = _literal(item)
            if not text:
                continue
            if text.strip().endswith(".sql"):
                found = self._file(text)
                if found:
                    self.runs[(task, found)] = span
                else:
                    self._report(call, f"SQL file {text.strip()!r} for task {task} was not found in the repository")
            elif embedded_sql.looks_like_sql(text):
                errors: list[dict] = []
                for statement in sql_statements(embedded_sql.normalize_placeholders(text), self.path, self.dialect, errors):
                    if statement["target"] and statement["sources"]:
                        self.transformations.append((task, TransformationNode(
                            source_datasets=statement["sources"], target_datasets=[statement["target"]],
                            transformation_type=TransformationType.SQL_QUERY, source_file=self.path,
                            line_range=span, sql_query_if_applicable=statement["text"][:500],
                            analysis_method=f"airflow_sql:{statement['dialect']}", confidence="syntax",
                            column_lineage=statement["columns"], unresolved_columns=statement["unresolved_columns"])))
                for error in errors:
                    self._report(call, f"Task SQL at line {call.lineno}: {error['error']}", incomplete=True)

    def _is_operator(self, call: ast.Call) -> bool:
        name = _name(call.func) or ""
        return name != "DAG" and (name.endswith(("Operator", "Sensor")) or _keyword(call, "task_id") is not None)

    # -- expressions ------------------------------------------------------
    def _refs(self, node: ast.AST, dag_id: str, in_loop: bool) -> list[str] | None:
        """Tasks an expression evaluates to, recording any dependencies inside it."""
        if isinstance(node, ast.Name):
            if node.id in self.variables:
                return self.variables[node.id]
            self._report(node, f"Unresolved task reference {node.id!r}")
            return None
        if isinstance(node, (ast.List, ast.Tuple)):
            tasks: list[str] = []
            for element in node.elts:
                tasks += self._refs(element, dag_id, in_loop) or []
            return tasks
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.RShift, ast.LShift)):
            left = self._refs(node.left, dag_id, in_loop)
            right = self._refs(node.right, dag_id, in_loop)
            upstream, downstream = (left, right) if isinstance(node.op, ast.RShift) else (right, left)
            self._depend(upstream, downstream, node)
            return right
        if isinstance(node, ast.Call):
            return self._call(node, dag_id, in_loop)
        return None

    def _call(self, call: ast.Call, dag_id: str, in_loop: bool) -> list[str] | None:
        name = _name(call.func)
        if isinstance(call.func, ast.Name) and name in self.taskflow:
            task = self.taskflow[name]
            for argument in [*call.args, *(k.value for k in call.keywords)]:
                upstream = self._refs(argument, dag_id, in_loop) if self._may_reference(argument) else None
                self._depend(upstream, [task], call)
            return [task]
        if isinstance(call.func, ast.Attribute) and name in {"set_downstream", "set_upstream"} and call.args:
            this = self._refs(call.func.value, dag_id, in_loop)
            other = self._refs(call.args[0], dag_id, in_loop)
            self._depend(this, other, call) if name == "set_downstream" else self._depend(other, this, call)
            return this
        if name in {"chain", "cross_downstream"}:
            groups = [self._refs(argument, dag_id, in_loop) for argument in call.args]
            for upstream, downstream in zip(groups, groups[1:], strict=False):
                if name == "chain" and upstream and downstream and len(upstream) == len(downstream) > 1:
                    for up, down in zip(upstream, downstream, strict=True):
                        self._depend([up], [down], call)
                else:
                    self._depend(upstream, downstream, call)
            return None
        if self._is_operator(call):
            created = self._operator_task(call, dag_id, in_loop)
            return [created] if created else None
        return None

    def _may_reference(self, node: ast.AST) -> bool:
        return (isinstance(node, ast.Name) and node.id in self.variables) or isinstance(node, (ast.Call, ast.List, ast.Tuple))

    def _depend(self, upstream: list[str] | None, downstream: list[str] | None, node: ast.AST) -> None:
        for up in upstream or []:
            for down in downstream or []:
                if up != down:
                    self.schedules.setdefault((up, down), _span(node))

    # -- statements -------------------------------------------------------
    def visit(self, statements: list[ast.stmt], dag_id: str, in_loop: bool = False) -> None:
        for statement in statements:
            if isinstance(statement, (ast.With, ast.AsyncWith)):
                inner = dag_id
                for item in statement.items:
                    if isinstance(item.context_expr, ast.Call) and _name(item.context_expr) == "DAG":
                        inner = self._dag_id(item.context_expr)
                        if isinstance(item.optional_vars, ast.Name):
                            self.dag_variables[item.optional_vars.id] = inner
                self.visit(statement.body, inner, in_loop)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function(statement, dag_id, in_loop)
            elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                self.visit(statement.body, dag_id, True)
                self.visit(statement.orelse, dag_id, in_loop)
            elif isinstance(statement, ast.If):
                self.visit(statement.body, dag_id, in_loop)
                self.visit(statement.orelse, dag_id, in_loop)
            elif isinstance(statement, ast.Try):
                for block in (statement.body, *(h.body for h in statement.handlers), statement.orelse, statement.finalbody):
                    self.visit(block, dag_id, in_loop)
            elif isinstance(statement, ast.Assign):
                self._assign(statement, dag_id, in_loop)
            elif isinstance(statement, ast.Expr):
                self._refs(statement.value, dag_id, in_loop) if self._is_task_expression(statement.value) else None

    def _is_task_expression(self, node: ast.AST) -> bool:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.RShift, ast.LShift)):
            return True
        return isinstance(node, ast.Call) and (
            _name(node.func) in {"chain", "cross_downstream", "set_downstream", "set_upstream"}
            or (isinstance(node.func, ast.Name) and node.func.id in self.taskflow) or self._is_operator(node))

    def _assign(self, statement: ast.Assign, dag_id: str, in_loop: bool) -> None:
        value = statement.value
        if isinstance(value, ast.Call) and _name(value) == "DAG":
            dag = self._dag_id(value)
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    self.dag_variables[target.id] = dag
            return
        tasks = self._refs(value, dag_id, in_loop) if self._is_task_expression(value) else None
        for target in statement.targets:
            if isinstance(target, ast.Name) and tasks:
                self.variables[target.id] = tasks

    def _function(self, function: ast.FunctionDef | ast.AsyncFunctionDef, dag_id: str, in_loop: bool) -> None:
        decorators = [_name(d.func) if isinstance(d, ast.Call) else
                      (d.value.id if isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name) else _name(d))
                      for d in function.decorator_list]
        if "dag" in decorators:
            outer, self.prefix = self.prefix, f"{self.prefix}{function.name}."
            self.visit(function.body, self._decorated_dag_id(function), in_loop)
            self.prefix = outer
            return
        if "task" not in decorators:
            return
        decorator = function.decorator_list[decorators.index("task")]
        given = _literal(_keyword(decorator, "task_id")) if isinstance(decorator, ast.Call) else None
        if in_loop:
            self._report(function, f"Dynamic task function {function.name!r} defined in a loop is not recorded")
            return
        task = self._task_node(dag_id, given or function.name, "@task", function)
        self.taskflow[function.name] = task
        self.runs[(task, f"{self.path}::{self.prefix}{function.name}")] = _span(function)


def parse_airflow(source: str, relative_path: str, known_paths: set[str], dialect: str | None = None) -> dict | None:
    """Return tasks, schedules, runs, task SQL transformations and diagnostics, or None if not an Airflow file."""
    tree = ast.parse(source)
    if not _is_airflow(tree):
        return None
    visitor = _DagVisitor(tree, relative_path, known_paths, dialect)
    visitor.visit(tree.body, visitor.default_dag)
    return {
        "tasks": list(visitor.tasks.values()),
        "schedules": [(up, down, span) for (up, down), span in visitor.schedules.items()],
        "runs": [(task, target, span) for (task, target), span in visitor.runs.items()],
        "transformations": visitor.transformations,
        "diagnostics": visitor.diagnostics,
    }
