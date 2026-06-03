"""
Long-Short Strategy Mining.
Allows short selling to capture returns during gold bear markets (e.g., 2013-2015).
Target: 20%+ Annualized Return.
"""
import pandas as pd
import numpy as np
import itertools
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

def long_short_signal(features, factors, thr_long, thr_short, hold_days, zw):
    """
    Long-Short signal:
    Score > thr_long -> +1
    Score < thr_short -> -1
    Else -> 0
    """
    std = standardize_factors(features, factors, window=zw)
    score = std.mean(axis=1).fillna(0)
    
    n = len(score)
    position = np.zeros(n)
    
    # Simple thresholding
    pos_raw = np.zeros(n)
    pos_raw[score > thr_long] = 1
    pos_raw[score < thr_short] = -1
    
    # We apply a holding period logic manually or via rule
    # A simple way is to use rolling median or just simple hold
    if hold_days > 1:
        # crude holding: forward fill if non-zero, but that might override
        # Better: use apply_holding_period for long and short separately
        sig_long = apply_holding_period(pd.Series((pos_raw == 1).astype(int), index=score.index), hold_days)
        sig_short = apply_holding_period(pd.Series((pos_raw == -1).astype(int), index=score.index), hold_days)
        
        pos_final = sig_long["position"] - sig_short["position"]
        # Resolve conflicts (if both 1, then 0)
        pos_final = np.clip(pos_final, -1, 1)
    else:
        pos_final = pd.Series(pos_raw, index=score.index)
        
    sig_df = pd.DataFrame({
        'raw_position': pos_final,
        'position': pos_final,
        'days_held': 0,
        'trade_flag': 0
    }, index=features.index)
    
    return sig_df

def eval_ls(features, panel, factors, thr_long, thr_short, hp, zw, cost=5):
    try:
        sig = long_short_signal(features, factors, thr_long, thr_short, hp, zw)
        res = run_backtest(panel, sig, config.target_symbol, cost)
        m = compute_metrics(res)
        return m
    except Exception as e:
        # print(f"Error eval_ls: {e}")
        return None

# ============================================================
# PHASE 1: Screen factors for Long/Short
# ============================================================
print("\n=== PHASE 1: Long/Short Factor Screen ===")
results = []
for feat in all_features.columns:
    for d in [1, -1]:
        for tl in [-0.2, 0.0, 0.2]:
            for ts in [-0.2, -0.5, -0.8]:
                m = eval_ls(all_features, panel, {feat: d}, tl, ts, 3, 120)
                if m and m["Ann_Return"] > 0.05:  # Lowered threshold
                    results.append({
                        "Feature": feat, "Dir": d, "ThrL": tl, "ThrS": ts,
                        "Ret": m["Ann_Return"], "Sharpe": m["Sharpe"], "DD": m["Max_DD"]
                    })

if not results:
    print("NO FACTORS FOUND > 5% RETURN!")
    import sys; sys.exit(1)

df = pd.DataFrame(results).sort_values("Ret", ascending=False)
print("Top 30 L/S factors:")
print(df.head(30).to_string(index=False))

# ============================================================
# PHASE 2: Greedy search
# ============================================================
print("\n=== PHASE 2: Greedy forward selection (L/S) ===")
candidates = {}
seen = set()
for _, row in df.iterrows():
    name = row["Feature"]
    if name not in seen and len(candidates) < 20:
        candidates[name] = int(row["Dir"])
        seen.add(name)

best_combo = {}
best_ret = 0
best_tl = 0
best_ts = 0
best_hp = 3
best_zw = 120

for iteration in range(8):
    best_add = None
    best_add_ret = best_ret
    local_tl, local_ts, local_hp, local_zw = 0, 0, 0, 0
    
    for name, direction in candidates.items():
        if name in best_combo:
            continue
        trial = {**best_combo, name: direction}
        
        for tl in [-0.2, 0.0, 0.2]:
            for ts in [-0.8, -0.5, -0.2]:
                for hp in [3, 5, 10]:
                    for zw in [60, 120, 252]:
                        m = eval_ls(all_features, panel, trial, tl, ts, hp, zw)
                        if m and m["Ann_Return"] > best_add_ret and m["Max_DD"] > -0.45:
                            best_add_ret = m["Ann_Return"]
                            best_add = name
                            local_tl, local_ts, local_hp, local_zw = tl, ts, hp, zw
    
    if best_add is None:
        print(f"  Iter {iteration+1}: No improvement. Stopping.")
        break
    
    best_combo[best_add] = candidates[best_add]
    best_ret = best_add_ret
    best_tl, best_ts, best_hp, best_zw = local_tl, local_ts, local_hp, local_zw
    
    m = eval_ls(all_features, panel, best_combo, best_tl, best_ts, best_hp, best_zw)
    print(f"  Iter {iteration+1}: +{best_add} | Ret={m['Ann_Return']:.4f} Sharpe={m['Sharpe']:.3f} DD={m['Max_DD']:.3f} Exp={m['Exposure']:.2f}")

print(f"\nFinal L/S Result: Ret={best_ret:.4f}, Factors={best_combo}, ThrL={best_tl}, ThrS={best_ts}, HP={best_hp}, ZW={best_zw}")
