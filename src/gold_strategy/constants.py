from datetime import date
from typing import List, Dict

# Target and Market Symbols
TARGET_SYMBOL = "GLD"
MARKET_SYMBOLS = ["SPY", "^VIX", "TLT", "UUP", "USO"]

# FRED Series IDs
FRED_SERIES = {
    "DGS10": "DGS10",  # 10-Year Treasury Constant Maturity Rate
    "DFII10": "DFII10", # 10-Year Treasury Inflation-Indexed Security (Real Rate)
    "T10YIE": "T10YIE", # 10-Year Breakeven Inflation Rate
    "T10Y2Y": "T10Y2Y", # 10-Year minus 2-Year Treasury Yield Spread
}

# Dates
DEFAULT_START_DATE = date(2004, 11, 18)  # GLD inception
DEFAULT_TRAIN_END_DATE = date(2016, 12, 31)
DEFAULT_TEST_START_DATE = date(2017, 1, 1)

# Horizons and Parameters
DEFAULT_HORIZONS = [5, 10, 20]
DEFAULT_MIN_HOLDING_PERIODS = [5, 10, 20]

# Costs (in bps)
DEFAULT_COST_SCENARIOS = [0, 5, 10]
