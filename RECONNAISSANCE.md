# RECONNAISSANCE.md — Manual Day-One Analysis

## Target Codebase

**Repository:** [dbt-labs/jaffle-shop](https://github.com/dbt-labs/jaffle-shop)  
**Stack:** dbt Core (SQL + YAML + Jinja), DuckDB, Python (CI/CD), GitHub Actions  
**Size:** ~37 files, ~1,400 lines of SQL/YAML/Python  
**Purpose:** Full-featured dbt example project simulating a multi-location e-commerce business (customers, orders, items, products, supplies, locations)

---

## Manual Exploration Process (30 minutes)

### Repository Structure

```
jaffle-shop/
├── dbt_project.yml                          # Project config: name, version, materialization, seeds
├── packages.yml                             # dbt package dependencies (dbt_utils, codegen)
├── package-lock.yml                         # Locked package versions
├── Taskfile.yml                             # Task runner config (build, test, deploy)
├── .pre-commit-config.yaml                  # Pre-commit hooks (sqlfmt, yamllint)
├── .github/workflows/
│   ├── ci.yml                               # CI: lint + test on PR
│   ├── cd_staging.yml                       # CD: deploy to staging on merge
│   ├── cd_prod.yml                          # CD: deploy to production on release
│   └── scripts/dbt_cloud_run_job.py         # Python helper: trigger dbt Cloud jobs
├── macros/
│   ├── cents_to_dollars.sql                 # Custom macro: currency conversion
│   └── generate_schema_name.sql             # Schema routing macro
├── models/
│   ├── staging/                             # Staging layer — clean raw data
│   │   ├── __sources.yml                    # Source definitions (ecom schema)
│   │   ├── stg_customers.sql / .yml         # Clean raw_customers
│   │   ├── stg_orders.sql / .yml            # Clean raw_orders (+ order status enum)
│   │   ├── stg_order_items.sql / .yml       # Clean raw_items (order line items)
│   │   ├── stg_products.sql / .yml          # Clean raw_products (+ type/description)
│   │   ├── stg_locations.sql / .yml         # Clean raw_stores → locations
│   │   └── stg_supplies.sql / .yml          # Clean raw_supplies (+ perishable flag)
│   └── marts/                               # Marts layer — business logic
│       ├── customers.sql / .yml             # Customer 360: lifetime value, order count
│       ├── orders.sql / .yml                # Enriched orders: items, subtotals, tax
│       ├── order_items.sql / .yml           # Line-item detail with product + supply info
│       ├── products.sql / .yml              # Product catalog passthrough
│       ├── supplies.sql / .yml              # Supply catalog passthrough
│       ├── locations.sql / .yml             # Location/store passthrough
│       └── metricflow_time_spine.sql        # MetricFlow time spine for metrics layer
└── seeds/
    ├── raw_customers.csv                    # 100 customer records
    ├── raw_orders.csv                       # 99 order records
    ├── raw_items.csv                        # Order line items
    ├── raw_products.csv                     # Product catalog
    ├── raw_stores.csv                       # Store/location data
    └── raw_supplies.csv                     # Supply chain data
```

---

## The Five FDE Day-One Questions

### 1. What is the primary data ingestion path?

Data enters through **dbt seeds** (CSV → warehouse tables) from the `seeds/` directory:

- `seeds/raw_customers.csv` → `ecom.raw_customers`
- `seeds/raw_orders.csv` → `ecom.raw_orders`
- `seeds/raw_items.csv` → `ecom.raw_items`
- `seeds/raw_products.csv` → `ecom.raw_products`
- `seeds/raw_stores.csv` → `ecom.raw_stores`
- `seeds/raw_supplies.csv` → `ecom.raw_supplies`

These 6 seed files are defined as dbt sources in `models/staging/__sources.yml` under the `ecom` schema. In production, they would be replaced by live warehouse tables.

### 2. What are the 3–5 most critical output datasets?

1. **`customers`** (`models/marts/customers.sql`, 59 lines) — Customer 360 model. Joins customers with orders to compute: first order date, most recent order date, number of orders, and **customer lifetime value**. This is the primary analytical output and the most complex model.

2. **`orders`** (`models/marts/orders.sql`, 78 lines) — The most complex SQL model by line count. Enriched orders joining with order items to compute: item subtotals, discount amounts, tax, and order totals. Contains multi-CTE business logic.

3. **`order_items`** (`models/marts/order_items.sql`, 67 lines) — Line-item detail joining orders, items, products, and supplies. Computes per-item pricing with the `cents_to_dollars` macro.

4. **`products`** / **`supplies`** / **`locations`** — Simple passthrough models from staging.

5. **`metricflow_time_spine`** — MetricFlow infrastructure for dbt Semantic Layer metrics.

### 3. What is the blast radius if the most critical module fails?

The most critical single point of failure is **`stg_orders`** because it feeds:

- `orders` (mart) — directly
- `order_items` (mart) — directly
- `customers` (mart) — indirectly through `orders`

**Blast radius of `stg_orders`:** 3 out of 6 mart models fail — `orders`, `order_items`, and `customers`. This effectively takes down the entire core analytics layer.

**Blast radius of `stg_order_items`:** `order_items` fails, which cascades to `orders` (since orders depends on `order_items`), which cascades to `customers`. Again: 3 out of 6 marts.

The staging layer is the most vulnerable — any single staging model failure cascades to multiple marts.

### 4. Where is the business logic concentrated vs. distributed?

Business logic is **concentrated in 3 mart files**:

- **`models/marts/orders.sql`** (78 lines) — Heaviest logic: multi-CTE pipeline computing subtotals, discounts, tax amounts, and order-level aggregations
- **`models/marts/order_items.sql`** (67 lines) — Joins 4 staging models + applies `cents_to_dollars` macro for pricing
- **`models/marts/customers.sql`** (59 lines) — Customer lifetime value aggregation across orders

The staging models are **pure cleaning** — renaming columns, casting types, mapping enums (e.g., order status codes → labels). Zero business logic in staging.

Custom macros provide reusable business rules:

- `macros/cents_to_dollars.sql` — Currency unit conversion used across marts
- `macros/generate_schema_name.sql` — Dynamic schema routing per environment

### 5. What has changed most frequently in the last 90 days (git velocity)?

The repository is relatively stable as a reference implementation. Recent changes focus on:

- Adding MetricFlow semantic layer definitions (`metricflow_time_spine.sql`)
- CI/CD pipeline updates (GitHub Actions workflows)
- YAML schema enrichment (adding `data_tests`, column descriptions)
- Migration from `tests:` to `data_tests:` key in YAML files

The highest conceptual churn is in the **schema YAML files** — they've grown significantly with detailed column-level documentation and test definitions.

---

## Difficulty Analysis — What Was Hardest to Figure Out Manually?

| Difficulty | Area             | Notes                                                                                                                         |
| ---------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Easy**   | File structure   | Clean, standard dbt project layout with `staging/` and `marts/`                                                               |
| **Easy**   | Data sources     | Clearly defined in `__sources.yml` under `ecom` schema                                                                        |
| **Medium** | Full lineage DAG | 13 models with cross-dependencies between staging and marts — required reading all SQL files to trace `ref()` chains          |
| **Medium** | Business logic   | `orders.sql` has 4 CTEs chaining computations — required careful reading to understand the aggregation pipeline               |
| **Hard**   | Blast radius     | Required mentally tracing `ref()` dependencies across 13 models to understand which staging failures cascade to which marts   |
| **Hard**   | Macro impact     | `cents_to_dollars` is called in `order_items.sql` — understanding its effect requires reading the macro definition separately |
| **N/A**    | Change velocity  | Requires git log analysis — not feasible from code reading alone                                                              |

### Key Insight

This new jaffle-shop is substantially more complex than the deprecated version (37 files vs. 8, 6 source tables vs. 3, proper staging→marts architecture with cross-model dependencies). Even for an experienced engineer, manually reconstructing the full lineage DAG took ~15 minutes of careful `ref()` tracing. In a real 800k-line codebase, this would require automated tooling — exactly what the Cartographer provides.
