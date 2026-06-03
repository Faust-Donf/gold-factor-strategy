# Implementation Plan: GLD Gold Factor Strategy Prototype

Date: 2026-06-03
Spec: `docs/superpowers/specs/2026-06-03-gold-factor-strategy-design.md`

## 0. Implementation Goal

Build a reproducible Python research prototype for a GLD gold ETF long-only timing strategy. The system should download or load data, construct technical and macro factors, evaluate factor efficacy, build explainable long/cash signals, run cost-aware backtests, validate out of sample and with walk-forward testing, and export notebooks, an HTML report, and a latest-signal table.

The first implementation must remain focused on the approved v1 scope:

- Target: GLD only.
- Direction: long-only, no shorting, no leverage.
- Frequency: daily signal generation with minimum holding-period constraints.
- Data: Yahoo Finance + FRED, with local CSV adapters prepared as an extension point.
- Strategy: factor research first, explainable voting/scoring rule second.
- Validation: fixed train/test split plus walk-forward.
- Outputs: notebook-facing modules, HTML report, and latest signal table.

## 1. Work Breakdown

### Phase 1 — Project Foundation

Create the minimal project foundation needed for modular research without implementing strategy logic inside notebooks.

Planned files:

```text
gold-factor-strategy/
  pyproject.toml
  README.md
  .gitignore
  src/gold_strategy/__init__.py
  tests/
  notebooks/
  reports/html/
  reports/figures/
  reports/signals/
  data/raw/
  data/processed/
```

Tasks:

1. Add `pyproject.toml` with package metadata and dependencies.
2. Add `.gitignore` for caches, virtualenvs, raw data, generated reports, and notebook checkpoints.
3. Add package skeleton under `src/gold_strategy`.
4. Add README with project purpose, scope, quickstart, and expected outputs.
5. Verify import path with a minimal package import test.

Acceptance criteria:

- `pip/uv` can install the project in editable mode.
- `python -c "import gold_strategy"` succeeds.
- No generated data or report artifacts are committed by default.

---

### Phase 2 — Configuration and Constants

Centralize project settings so data symbols, dates, cost assumptions, and validation windows are not hardcoded across modules.

Planned files:

```text
src/gold_strategy/config.py
src/gold_strategy/constants.py
```

Tasks:

1. Define default target symbol: `GLD`.
2. Define market symbols such as `SPY`, `VIX`, `TLT`, `UUP`, and oil proxy.
3. Define FRED series identifiers for rates, real rates, breakeven inflation, and yield curve components.
4. Define default backtest start date as GLD inception-era start date, with the data loader allowed to trim to available history.
5. Define default cost scenarios: `0bp`, `5bp`, `10bp`.
6. Define default train/test split: training through `2016-12-31`, testing from `2017-01-01`.
7. Define default horizons: 5, 10, and 20 trading days.
8. Define default minimum holding periods: 5, 10, and 20 trading days.

Acceptance criteria:

- All downstream modules can import settings from one place.
- No core module hardcodes the approved symbols, horizons, or cost assumptions unless they are explicit function defaults referencing config.

---

### Phase 3 — Data Layer

Implement a data layer that separates data acquisition from factor and backtest logic.

Planned files:

```text
src/gold_strategy/data/yahoo_loader.py
src/gold_strategy/data/fred_loader.py
src/gold_strategy/data/local_loader.py
src/gold_strategy/data/panel_builder.py
src/gold_strategy/data/quality.py
```

Tasks:

1. Implement Yahoo loader for GLD and market symbols.
2. Implement FRED loader for macro series.
3. Implement local CSV loader with a consistent interface but minimal assumptions.
4. Implement daily panel builder using GLD trading dates as the master index.
5. Forward-fill macro data onto GLD trading days.
6. Preserve adjusted-close fields for return calculation.
7. Add data quality checks for missing target prices, missing macro spans, duplicated dates, and non-monotonic indexes.
8. Add optional caching under `data/raw` and `data/processed`.

Acceptance criteria:

- A single function can return a clean daily research panel indexed by date.
- The panel contains GLD adjusted close, GLD volume, market proxies, and macro series.
- Missing target price rows are excluded or flagged.
- Macro series are forward-filled only from historical observations.
- The data layer can run without strategy/backtest modules.

Key tests:

- Panel index is sorted and unique.
- GLD returns can be computed from adjusted close.
- Forward fill does not backfill earlier dates.
- Local CSV input can produce the same schema as downloaded data when columns are provided.

---

### Phase 4 — Feature Engineering

Implement technical and macro factors with strict lag discipline.

Planned files:

```text
src/gold_strategy/features/technical.py
src/gold_strategy/features/macro.py
src/gold_strategy/features/transforms.py
src/gold_strategy/features/targets.py
```

Tasks:

1. Implement technical factors:
   - Momentum: 5, 10, 20, 60, 120, 252 days.
   - Moving-average gaps: 20, 60, 120 days.
   - Moving-average crosses: 20/60 and 60/120.
   - Realized volatility: 20 and 60 days.
   - Volatility ratio: 20/60.
   - Drawdown from 20-day and 60-day highs.
   - Breakout flags for 20-day and 60-day highs.
   - Distance to 252-day high.
   - Volume z-score and volume ratio.
2. Implement macro factors:
   - Rate levels and changes.
   - Real-rate levels and changes.
   - Breakeven inflation levels and changes.
   - Yield curve spread.
   - Dollar proxy momentum and percentile/z-score.
   - VIX level and change.
   - SPY momentum and drawdown.
   - Oil proxy momentum.
3. Implement transforms:
   - Rolling z-score.
   - Rolling percentile rank.
   - Winsorization.
   - Direction adjustment.
4. Implement forward-return targets for 5, 10, and 20 trading days.
5. Explicitly separate features available at time `t` from future target returns.

Acceptance criteria:

- Feature functions accept a DataFrame and return a DataFrame without mutating input unexpectedly.
- Future returns are created only as labels for research, not as strategy inputs.
- Feature names are deterministic and documented.
- Feature creation can run independently of notebooks.

Key tests:

- Momentum uses only past prices.
- Moving averages use rolling historical windows.
- Forward returns shift in the correct direction.
- Rolling z-scores do not use future data.
- Output index aligns with input index.

---

### Phase 5 — Factor Research

Implement factor evaluation tools for IC, quantile returns, stability, correlation, and selection.

Planned files:

```text
src/gold_strategy/research/ic_analysis.py
src/gold_strategy/research/quantile_analysis.py
src/gold_strategy/research/stability.py
src/gold_strategy/research/factor_selection.py
src/gold_strategy/research/correlation.py
```

Tasks:

1. Compute Pearson IC and Spearman rank IC by factor and horizon.
2. Compute IC mean, IC standard deviation, ICIR, and positive-IC ratio.
3. Compute annual IC and rolling-window IC.
4. Compute quantile forward returns for 5, 10, and 20 days.
5. Compute monotonicity score across quantiles.
6. Compute high-quantile excess return versus full sample.
7. Compute factor correlation matrix and correlation clusters.
8. Implement a transparent factor selection table with reason codes.
9. Support separate analysis for technical, macro, and combined factor groups.

Acceptance criteria:

- Research outputs are tabular and notebook/report friendly.
- The selection result contains selected factors, rejected factors, and explanations.
- Highly correlated factors can be identified and de-duplicated.
- The analysis can be restricted to a training period.

Key tests:

- IC functions handle missing values gracefully.
- Quantile analysis does not fail when duplicate factor values exist.
- Selection does not use test-period statistics when a training mask is provided.
- Correlation grouping is deterministic for a fixed input.

---

### Phase 6 — Signal Construction

Convert selected factors into explainable long/cash signals.

Planned files:

```text
src/gold_strategy/strategy/scoring.py
src/gold_strategy/strategy/signal.py
src/gold_strategy/strategy/rules.py
```

Tasks:

1. Standardize selected factors using rolling z-score or percentile rank.
2. Apply factor direction so higher adjusted values consistently mean more bullish for GLD.
3. Implement equal-weight voting signal:
   - bullish vote;
   - neutral vote;
   - bearish vote.
4. Implement equal-weight score signal.
5. Implement optional ICIR-weighted score as a comparison, not as the default main strategy.
6. Convert score into raw signal using configurable thresholds.
7. Apply minimum holding-period constraints to produce final signal.
8. Generate reason strings for signal changes.

Acceptance criteria:

- Final position is always 0 or 1.
- Raw signal and final signal are both available for analysis.
- Signal output includes score, position, trade flag, days since last trade, and reason.
- Minimum holding-period logic prevents excessive flips.

Key tests:

- Position never becomes negative or above 1.
- Minimum holding period blocks early exits/entries when configured.
- Missing factor values do not crash signal generation.
- Equal-weight strategy is reproducible for a fixed selected-factor list.

---

### Phase 7 — Backtest Engine and Metrics

Implement a simple, transparent, vectorized long-only backtest engine.

Planned files:

```text
src/gold_strategy/backtest/engine.py
src/gold_strategy/backtest/costs.py
src/gold_strategy/backtest/metrics.py
src/gold_strategy/backtest/validation.py
```

Tasks:

1. Compute GLD daily returns from adjusted close.
2. Apply signal lag so position at `t-1` earns return at `t`.
3. Deduct costs only when position changes.
4. Support cost scenarios: 0bp, 5bp, 10bp.
5. Compute buy-and-hold benchmark.
6. Compute annual return, annual volatility, Sharpe, Sortino, Calmar, max drawdown, win rate, exposure, turnover, number of trades, average holding days, best year, worst year, excess return, and drawdown reduction.
7. Implement fixed train/test split evaluation.
8. Implement walk-forward validation with 5-year training and 1-year testing windows.

Acceptance criteria:

- Strategy returns are computed without look-ahead bias.
- Costs reduce returns only on trading dates.
- Benchmark uses the same GLD adjusted-close series.
- Metrics are generated for strategy and benchmark in a comparable table.
- Validation outputs can be consumed by reports.

Key tests:

- A constant long signal matches buy-and-hold before costs.
- A constant cash signal has zero return before costs.
- A single buy transaction deducts one-side cost once.
- A buy then sell deducts two one-side costs.
- Max drawdown is non-positive and consistent with equity curve.
- Walk-forward windows do not overlap incorrectly between train and test periods.

---

### Phase 8 — Reporting and Exports

Produce user-facing research outputs.

Planned files:

```text
src/gold_strategy/reporting/plots.py
src/gold_strategy/reporting/html_report.py
src/gold_strategy/reporting/signal_export.py
notebooks/01_factor_research.ipynb
notebooks/02_strategy_backtest.ipynb
```

Tasks:

1. Implement plotting helpers for:
   - equity curve;
   - drawdown curve;
   - annual returns;
   - factor IC bar charts;
   - quantile return charts;
   - cost sensitivity comparison;
   - walk-forward results.
2. Implement HTML report generation with a Jinja template or notebook export workflow.
3. Implement latest-signal export to CSV and HTML.
4. Create notebook templates that call the package modules rather than containing core logic.
5. Ensure reports include limitations about macro data release lag and free data quality.

Acceptance criteria:

- Running the reporting workflow produces an HTML report under `reports/html`.
- Running the latest-signal workflow produces a table under `reports/signals`.
- Report includes executive summary, factor research, selected factors, backtest, validation, cost sensitivity, and latest signal.
- Notebook outputs are reproducible from the package modules.

Key tests/checks:

- HTML file is created and non-empty.
- Latest signal table has required fields.
- Report generation handles empty or partially missing optional plots gracefully.

---

### Phase 9 — Command Entrypoints

Add lightweight CLI/module entrypoints after the internal modules are stable.

Planned files:

```text
src/gold_strategy/run_research.py
src/gold_strategy/run_backtest.py
src/gold_strategy/generate_report.py
src/gold_strategy/latest_signal.py
```

Tasks:

1. Add `run_research` to build the panel, create factors, and export factor research tables.
2. Add `run_backtest` to generate signals and performance outputs.
3. Add `generate_report` to render the HTML report.
4. Add `latest_signal` to export the current GLD signal.
5. Keep arguments minimal for v1:
   - start date;
   - end date;
   - force refresh;
   - cost scenario;
   - minimum holding period;
   - output directory.

Acceptance criteria:

- Each entrypoint can be run with `python -m gold_strategy.<module>`.
- Entrypoints fail with clear errors when data cannot be downloaded.
- Entrypoints do not require notebook execution.

---

### Phase 10 — Final Verification

Run the end-to-end checks expected for the v1 prototype.

Tasks:

1. Install package in editable mode.
2. Run unit tests.
3. Run data build.
4. Run factor research.
5. Run strategy backtest.
6. Run cost sensitivity.
7. Run fixed sample-out validation.
8. Run walk-forward validation.
9. Generate latest signal table.
10. Generate HTML report.
11. Check git status for unwanted artifacts.
12. Update README with actual commands and known limitations.

Acceptance criteria:

- Tests pass.
- End-to-end run completes with downloaded or cached data.
- Reports and signal table are generated locally but not committed unless explicitly intended.
- README contains runnable commands.
- Known limitations are documented.

## 2. Implementation Order

Use this order to reduce integration risk:

1. Project foundation.
2. Config/constants.
3. Data layer.
4. Feature layer.
5. Research layer.
6. Signal layer.
7. Backtest layer.
8. Reporting layer.
9. Entrypoints.
10. Notebooks.
11. Final verification and README polish.

Do not begin with notebooks. Notebooks should be thin consumers of tested package functions.

## 3. Dependency Boundaries

Maintain these boundaries:

```text
data -> features -> research -> strategy -> backtest -> reporting
```

Rules:

- `data` must not depend on strategy or backtest code.
- `features` must not depend on research or backtest code.
- `research` may depend on features outputs but must not create trade positions.
- `strategy` may use selected-factor outputs but must not compute performance metrics.
- `backtest` may use positions and prices but must not select factors.
- `reporting` may read outputs from all layers but should not contain business logic.

## 4. Anti-Lookahead Requirements

Every implementation phase must preserve these constraints:

1. Future returns are labels only.
2. Features at date `t` use information available at or before `t`.
3. Strategy return at date `t` uses position from `t-1`.
4. Macro data is forward-filled, never backfilled.
5. Factor selection for test evaluation uses training-period statistics only.
6. Walk-forward uses only prior-window data to select factors and parameters.

## 5. Main v1 Defaults

Use the following defaults unless implementation discovers a documented blocker:

```text
target_symbol: GLD
signal_frequency: daily
position_set: {0, 1}
base_cost_bps: 5
cost_sensitivity_bps: [0, 5, 10]
horizons: [5, 10, 20]
minimum_holding_days: [5, 10, 20]
main_weighting: equal_weight
comparison_weighting: ICIR_weighted
train_end: 2016-12-31
test_start: 2017-01-01
walk_forward_train_years: 5
walk_forward_test_years: 1
cash_return: 0
execution_lag: 1 trading day
```

## 6. Files Not to Commit by Default

Do not commit these unless the user explicitly asks for examples:

```text
data/raw/*
data/processed/*
reports/html/*
reports/figures/*
reports/signals/*
*.ipynb_checkpoints
__pycache__
.pytest_cache
.venv
```

## 7. Definition of Done

The implementation is complete when all of the following are true:

1. The project installs locally.
2. Data can be downloaded or loaded into a clean GLD daily panel.
3. Technical and macro factors are generated.
4. Forward return labels for 5, 10, and 20 days are generated.
5. IC, quantile, stability, and correlation analyses run.
6. Selected factors are exported with reasons.
7. Technical, macro, and combined strategies can be compared.
8. Final long/cash signal respects minimum holding constraints.
9. Backtest runs with 0bp, 5bp, and 10bp costs.
10. Buy-and-hold benchmark comparison is produced.
11. Fixed train/test validation is produced.
12. Walk-forward validation is produced.
13. Latest signal table is exported.
14. HTML report is generated.
15. Core tests pass.
16. README documents usage and limitations.

## 8. Open Decisions Deferred to Implementation

These are implementation-level decisions, not spec blockers:

1. Exact chart library split between matplotlib and plotly.
2. Whether HTML report is generated directly with Jinja or via notebook export.
3. Exact FRED series IDs after verifying availability.
4. Whether first CLI entrypoints use argparse or a simpler module-level config.
5. Whether notebooks are committed as clean templates or generated after a run.

Each deferred decision should be resolved in favor of simplicity and reproducibility.

