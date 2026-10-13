# dbt lineage

Cartographer can read an existing dbt manifest without installing or running
dbt, connecting to a warehouse, or executing compiled SQL. This adapter reads
the lineage subset of [manifest v12](https://schemas.getdbt.com/dbt/manifest/v12.json).
See the [official artifact reference](https://docs.getdbt.com/reference/artifacts/manifest-json)
for how dbt produces the file and which dbt versions use this schema.

```sh
uv run cartographer analyze /path/to/dbt-project --dbt-manifest target/manifest.json --json
uv run cartographer lineage model.shop.orders --repo /path/to/dbt-project --json
```

The manifest path must be relative to the analyzed repository, use forward
slashes, and end in `.json`. Selection explicitly permits this single artifact
through ignore rules, including an ignored `target/` directory. Symlinks,
junctions, paths outside the repository, and protected internal directories
are still rejected. Arbitrary JSON files are not scanned automatically.
The default artifact size limit is 2 MiB, within the scan's 100 MiB total budget;
the adapter also caps resources at 10,000 and dependencies at 50,000. Python
callers can configure scanner byte limits using `ScanSettings`.

Dataset names are dbt unique IDs such as `model.shop.orders`,
`model.shop.orders.v1`, and `source.shop.raw.orders`. Package and model-version
identities remain distinct; available warehouse relation names are retained
as dataset metadata. Dependencies come from `depends_on.nodes`. Supported
dataset resources are models, seeds, snapshots and sources. Tests, macros,
exposures, metrics and semantic models are outside this adapter's scope.
Disabled resources are omitted. This is not full manifest schema validation.

Evidence is labeled `dbt_manifest` with `artifact` confidence, the supplying
artifact path and resource ID. Definition paths are preserved, but exact line
spans remain unknown. The manifest describes a generated snapshot: Cartographer
does not establish whether it reflects the current SQL files or warehouse state.
Regenerate it with your dbt workflow when definitions change. This limitation
is also included in query coverage diagnostics.

Manifest fingerprints and scanner settings participate in checkpoint matching.
Editing or removing the artifact invalidates cached analysis. Invalid,
unsupported, oversized or missing artifacts and unresolved dataset dependencies
produce partial coverage without advancing the successful checkpoint. A supplied
manifest takes precedence over the source-template fallback; definition files
listed by that manifest are excluded from standalone SQL parsing.

Without `--dbt-manifest`, the existing source fallback reads allowed project
configuration, model SQL and schema YAML. It supports basic `ref`/`source`
patterns and is less complete than dbt's own parsed artifact. Selecting a
manifest does not enable model calls or uploads.

With a manifest, column lineage comes from each model's `compiled_code`, where
refs and macros are already expanded. Run `dbt compile` (not only `dbt parse`)
before analysis so the manifest contains compiled SQL; models without it are
counted in a diagnostic. Tables in compiled SQL are mapped back to dbt unique IDs
through `relation_name`, so column nodes look like
`model.shop.orders::order_id`. The manifest's `adapter_type` selects the SQL
dialect. Cartographer never runs dbt itself.

Every `dbt_project.yml` in the repository is treated as a project root, so
projects in subfolders (for example `analytics/dbt/`) and several projects in
one repository are supported. The analysis summary lists them as
`dbt_projects`. All `.sql` files inside a project (models, macros, tests,
analyses, snapshots) are Jinja templates and are never parsed as plain SQL.
Snapshots are not yet modeled. Model names are not namespaced by project, so
two projects that define the same model name share one dataset node.

Python callers use the same pipeline:

```python
from brownfield_cartographer.orchestrator import Orchestrator
from brownfield_cartographer.scanner import ScanSettings

runner = Orchestrator(scan_settings=ScanSettings(dbt_manifest="target/manifest.json"))
result = runner.run("/path/to/dbt-project")
```
