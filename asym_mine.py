"""
Asymmetric entry/exit strategy with the best 7 factors from greedy_mine.
- Entry threshold: permissive (enter long easily)
- Exit threshold: strict (only exit when signals are extremely bearish)
- Also tries per-factor optimal zscore windows
"""
import pandas as pd
import numpy as np
from gold_strategy.config import StrategyConfig
from gold_strategy.data import build_panel
from gold_strategy.features import create_technical_features, create_macro_features
from gold_strategy.strategy.scoring import standardize_factors
from gold_strategy.strategy.rules import apply_holding_period
from gold_strategy.backtest import run_backtest, compute_metrics

config = StrategyConfig.get_default()
panel = build_panel(config)
tech = create_technical_features(panel, config.target_symbol)
macro = create_macro_features(panel)
all_features = tech.join(macro).dropna(how='all')

# Best 7 factors from greedy mine
BASE_FACTORS = {
    'Breakout_60d': 1, 'IEF_DD_252d': 1, 'Mom_60d': -1,
    'GLD_SPY_Corr_20d': 1, 'IEF_Mom_10d': 1, 'ATR_Ratio_14_60': 1,
    'GLD_vs_TLT_RS_60d': -1
}

# Additional strong candidates from ultra mine
EXTRA_FACTORS = {
    'Vol_Z_20d': 1, 'Mom_Accel_20_60': 1, 'Kurt_60d': -1,
    'PV_Div_20d': -1, 'GLD_vs_IEF_RS_60d': -1, 'DistLow_60d': -1,
    'Skew_60d': 1, 'VIX_Mom_20d': 1, 'GLD_vs_SPY_RS_60d': -1,
    'TLT_DD_252d': 1, 'IEF_Z_60d': 1, 'Breakout_20d': 1,
}

ALL_CANDIDATES = {**BASE_FACTORS, **EXTRA_FACTORS}

def asymmetric_signal(features, factors, entry_thr, exit_thr, hold_days, zscore_window):
    """
    Asymmetric signal: 
    - If currently cash and score > entry_thr -> go long
    - If currently long and score < exit_thr -> go cash
    - This creates hysteresis that keeps us invested longer
    """
    std = standardize_factors(features, factors, window=zscore_window)
    score = std.mean(axis=1).fillna(0)
    
    n = len(score)
    position = np.zeros(n)
    current_pos = 0
    days_in = 0
    
    for i in range(n):
        s = score.iloc[i]
        if current_pos == 0:
            # Currently cash - need score > entry_thr to go long
            if s > entry_thr:
                current_pos = 1
                days_in = 1
            else:
                current_pos = 0
                days_in = 0
        else:
            # Currently long
            days_in += 1
            if s < exit_thr and days_in >= hold_days:
                current_pos = 0
                days_in = 0
            else:
                current_pos = 1
        position[i] = current_pos
    
    sig_df = pd.DataFrame({
        'raw_position': position,
        'position': position,
        'days_held': 0,
        'trade_flag': 0
    }, index=features.index[:n])
    
    return sig_df

def eval_asym(features, panel, factors, entry_thr, exit_thr, hold_days, zw, cost=5):
    try:
        sig = asymmetric_signal(features, factors, entry_thr, exit_thr, hold_days, zw)
        res = run_backtest(panel, sig, config.target_symbol, cost)
        m = compute_metrics(res)
        return m
    except:
        return None

print(f"Features: {len(all_features.columns)}")

# ============================================================
# PHASE 1: Test base 7 factors with asymmetric thresholds
# ============================================================
print("\n=== PHASE 1: Asymmetric threshold on base factors ===")
best_ret = 0
best_cfg = None

for entry in np.arange(-0.6, 0.3, 0.1):
    for exit_t in np.arange(-1.5, -0.2, 0.1):
        if exit_t >= entry:
            continue
        for hp in [3, 5, 10]:
            for zw in [60, 120, 252]:
                m = eval_asym(all_features, panel, BASE_FACTORS, entry, exit_t, hp, zw)
                if m and m["Ann_Return"] > best_ret and m["Max_DD"] > -0.45:
                    best_ret = m["Ann_Return"]
                    best_cfg = {"entry": entry, "exit": exit_t, "hp": hp, "zw": zw}
                    if best_ret > 0.14:
                        print(f"  Ret={best_ret:.4f} Sharpe={m['Sharpe']:.3f} DD={m['Max_DD']:.3f} Exp={m['Exposure']:.2f} | entry={entry:.1f} exit={exit_t:.1f} hp={hp} zw={zw}")

print(f"\nPhase 1 best: Ret={best_ret:.4f}, Config={best_cfg}")

# ============================================================
# PHASE 2: Greedy add factors with asymmetric signal
# ============================================================
print("\n=== PHASE 2: Greedy factor addition with asymmetric signal ===")
current_factors = dict(BASE_FACTORS)

for iteration in range(8):
    best_add = None
    best_add_ret = best_ret
    local_cfg = dict(best_cfg)
    
    for name, direction in EXTRA_FACTORS.items():
        if name in current_factors:
            continue
        trial = {**current_factors, name: direction}
        
        for entry in np.arange(-0.6, 0.3, 0.15):
            for exit_t in np.arange(-1.5, -0.2, 0.15):
                if exit_t >= entry:
                    continue
                for hp in [3, 5, 10]:
                    for zw in [60, 120, 252]:
                        m = eval_asym(all_features, panel, trial, entry, exit_t, hp, zw)
                        if m and m["Ann_Return"] > best_add_ret and m["Max_DD"] > -0.45:
                            best_add_ret = m["Ann_Return"]
                            best_add = name
                            local_cfg = {"entry": entry, "exit": exit_t, "hp": hp, "zw": zw}
    
    if best_add is None:
        print(f"  Iter {iteration+1}: No improvement. Stopping.")
        break
    
    current_factors[best_add] = EXTRA_FACTORS[best_add]
    best_ret = best_add_ret
    best_cfg = local_cfg
    
    m = eval_asym(all_features, panel, current_factors, best_cfg["entry"], best_cfg["exit"], best_cfg["hp"], best_cfg["zw"])
    print(f"  Iter {iteration+1}: +{best_add} | Ret={m['Ann_Return']:.4f} Sharpe={m['Sharpe']:.3f} DD={m['Max_DD']:.3f} Exp={m['Exposure']:.2f} | {best_cfg}")

# ============================================================
# PHASE 3: Ultra fine-tune
# ============================================================
print(f"\n=== PHASE 3: Fine-tune ===")
print(f"Factors: {current_factors}")

for entry in np.arange(-0.8, 0.4, 0.05):
    for exit_t in np.arange(-2.0, entry, 0.1):
        for hp in [1, 2, 3, 5, 7, 10]:
            for zw in [40, 60, 90, 120, 180, 252]:
                m = eval_asym(all_features, panel, current_factors, entry, exit_t, hp, zw)
                if m and m["Ann_Return"] > best_ret and m["Max_DD"] > -0.45:
                    best_ret = m["Ann_Return"]
                    best_cfg = {"entry": entry, "exit": exit_t, "hp": hp, "zw": zw}
                    print(f"  Ret={best_ret:.4f} Sharpe={m['Sharpe']:.3f} DD={m['Max_DD']:.3f} Exp={m['Exposure']:.2f} | {best_cfg}")

# Final
m = eval_asym(all_features, panel, current_factors, best_cfg["entry"], best_cfg["exit"], best_cfg["hp"], best_cfg["zw"])
print(f"\n{'='*60}")
print(f"=== FINAL BEST ===")
print(f"{'='*60}")
print(f"Factors:    {current_factors}")
print(f"Entry Thr:  {best_cfg['entry']:.2f}")
print(f"Exit Thr:   {best_cfg['exit']:.2f}")
print(f"Hold Days:  {best_cfg['hp']}")
print(f"ZScore Win: {best_cfg['zw']}")
print(f"Ann Return: {m['Ann_Return']:.4f}")
print(f"BM Return:  {m['BM_Return']:.4f}")
print(f"Sharpe:     {m['Sharpe']:.4f}")
print(f"Max DD:     {m['Max_DD']:.4f}")
print(f"Exposure:   {m['Exposure']:.4f}")
print(f"Trades:     {m['Num_Trades']:.0f}")
print(f"Win Rate:   {m['Win_Rate']:.4f}")
