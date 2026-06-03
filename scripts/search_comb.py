import pandas as pd
import itertools
from gold_strategy.config import StrategyConfig
from gold_strategy.data import build_panel
from gold_strategy.features import create_technical_features, create_macro_features
from gold_strategy.strategy import generate_signals
from gold_strategy.backtest import run_backtest, compute_metrics

config = StrategyConfig.get_default()
panel = build_panel(config)

tech = create_technical_features(panel, config.target_symbol)
macro = create_macro_features(panel)
all_features = tech.join(macro).dropna(how='all')

best_factors = {
    "Vol_60d": -1,
    "Mom_60d": -1,
    "Oil_Mom_20d": -1,
    "DD_60d": 1,
    "DD_252d": 1,
    "MA_Gap_20d": -1,
    "Volume_Ratio_20_60": 1,
    "MA_Gap_120d": -1,
    "Breakout_20d": -1,
    "MA_Cross_20_60": -1,
    "VIX_Chg_20d": -1,
    "Mom_5d": 1
}

# Try combinations of 1 to 4 factors
keys = list(best_factors.keys())
best_ret = 0
best_comb = None

results = []

for r in range(1, 5):
    for comb in itertools.combinations(keys, r):
        selected = {k: best_factors[k] for k in comb}
        sig = generate_signals(all_features, selected, min_holding_days=5)
        res = run_backtest(panel, sig, config.target_symbol, config.base_cost_bps)
        m = compute_metrics(res)
        
        if m["Ann_Return"] > best_ret and m["Max_DD"] > -0.40:
            best_ret = m["Ann_Return"]
            best_comb = selected
            print(f"New best Return! Ret: {best_ret:.3f}, Sharpe: {m['Sharpe']:.3f}, DD: {m['Max_DD']:.3f} | {list(selected.keys())}")

print(f"Best Comb: {best_comb}")
