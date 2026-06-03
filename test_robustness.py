import pandas as pd
import numpy as np
from gold_strategy.config import StrategyConfig
from gold_strategy.data import build_panel
from gold_strategy.features import create_technical_features, create_macro_features
from gold_strategy.strategy.signal import generate_asymmetric_signals
from gold_strategy.backtest import run_backtest, compute_metrics

config = StrategyConfig.get_default()
panel = build_panel(config)
tech = create_technical_features(panel, config.target_symbol)
macro = create_macro_features(panel)
all_features = tech.join(macro).dropna(how='all')

best_factors = {
    'Breakout_60d': 1, 'IEF_DD_252d': 1, 'Mom_60d': -1, 
    'GLD_SPY_Corr_20d': 1, 'IEF_Mom_10d': 1, 'ATR_Ratio_14_60': 1, 
    'GLD_vs_TLT_RS_60d': -1, 'Mom_Accel_20_60': 1, 
    'GLD_vs_SPY_RS_60d': -1, 'VIX_Mom_20d': 1
}

print("=== Parameter Sensitivity Analysis (参数敏感性测试) ===")
print("Testing slight variations of Entry & Exit Thresholds around (-0.15, -0.45)")
print(f"{'Entry':<6} | {'Exit':<6} | {'Ann_Ret':<8} | {'Sharpe':<6} | {'Max_DD':<8} | Trades")
print("-" * 55)

for entry in [-0.10, -0.12, -0.15, -0.18, -0.20]:
    for exit in [-0.40, -0.42, -0.45, -0.48, -0.50]:
        sig = generate_asymmetric_signals(all_features, best_factors, entry, exit, hold_days=5)
        res = run_backtest(panel, sig, config.target_symbol, cost_bps=5)
        m = compute_metrics(res)
        print(f"{entry:<6.2f} | {exit:<6.2f} | {m['Ann_Return']:.2%}   | {m['Sharpe']:.3f}  | {m['Max_DD']:.2%}  | {int(m['Num_Trades'])}")
