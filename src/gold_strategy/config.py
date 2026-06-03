from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict

from gold_strategy import constants

@dataclass
class StrategyConfig:
    target_symbol: str = constants.TARGET_SYMBOL
    market_symbols: List[str] = field(default_factory=lambda: constants.MARKET_SYMBOLS.copy())
    fred_series: Dict[str, str] = field(default_factory=lambda: constants.FRED_SERIES.copy())
    
    start_date: date = constants.DEFAULT_START_DATE
    train_end_date: date = constants.DEFAULT_TRAIN_END_DATE
    test_start_date: date = constants.DEFAULT_TEST_START_DATE
    
    horizons: List[int] = field(default_factory=lambda: constants.DEFAULT_HORIZONS.copy())
    min_holding_periods: List[int] = field(default_factory=lambda: constants.DEFAULT_MIN_HOLDING_PERIODS.copy())
    
    cost_scenarios: List[int] = field(default_factory=lambda: constants.DEFAULT_COST_SCENARIOS.copy())
    base_cost_bps: int = 5
    
    walk_forward_train_years: int = 5
    walk_forward_test_years: int = 1
    
    main_weighting: str = "equal_weight"
    comparison_weighting: str = "ICIR_weighted"
    
    signal_frequency: str = "daily"
    execution_lag: int = 1

    @classmethod
    def get_default(cls) -> "StrategyConfig":
        return cls()
