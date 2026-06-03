# Gold Factor Strategy

GLD Gold ETF Factor Timing Strategy Prototype.

## Quickstart

This project uses a standard `pyproject.toml` and requires Python >= 3.10.
To install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Running the Code

You can run the entrypoints as follows:

```bash
python -m gold_strategy.run_research
python -m gold_strategy.run_backtest
python -m gold_strategy.generate_report
python -m gold_strategy.latest_signal
```

Or explore via notebooks:
- `notebooks/01_factor_research.ipynb`
- `notebooks/02_strategy_backtest.ipynb`

## Known Limitations

- **Rate Limits**: The default setup relies on `yfinance` and `pandas-datareader` (FRED). You may encounter rate limiting errors ("Too Many Requests") if you run data fetches frequently. The data loaders degrade gracefully to empty DataFrames if data cannot be fetched.
- **Macro Data Lag**: FRED macro data is forward-filled. True lag in data release is not fully simulated unless local cache overrides specify exact release dates.
- **Free Data Quality**: Free API endpoints may have missing or erroneous daily points.
