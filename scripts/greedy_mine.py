"""
Greedy forward-selection factor mining + aggressive threshold/holding period tuning.
Strategy: Start from the best single factor, greedily add factors that improve return the most.
Then fine-tune threshold and holding period on the best combo.
"""
import pandas as pd
import numpy as np
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

print(f"Total features: {len(all_features.columns)}, Data rows: {len(panel)}")

# ============================================================
# PHASE 1: Screen ALL factors at multiple thresholds
# ============================================================
print("\n=== PHASE 1: Broad Screen ===")
results = []
for feat in all_features.columns:
    for d in [1, -1]:
        for thr in [-0.5, -0.3, -0.1]:
            try:
                sig = generate_signals(all_features, {feat: d}, min_holding_days=3, threshold=thr)
                res = run_backtest(panel, sig, config.target_symbol, 5)
                m = compute_metrics(res)
                results.append({
                    "Feature": feat, "Dir": d, "Thr": thr,
                    "Ret": m["Ann_Return"], "Sharpe": m["Sharpe"],
                    "DD": m["Max_DD"], "Exp": m["Exposure"]
                })
            except:
                pass

df = pd.DataFrame(results).sort_values("Ret", ascending=False)
print("Top 30 single factors by return:")
print(df.head(30).to_string(index=False))

# ============================================================
# PHASE 2: Greedy forward selection
# ============================================================
print("\n=== PHASE 2: Greedy Forward Selection ===")

# Build candidate pool: top 25 unique factors by return
candidates = {}
seen = set()
for _, row in df.iterrows():
    name = row["Feature"]
    if name not in seen and len(candidates) < 25:
        candidates[name] = int(row["Dir"])
        seen.add(name)

print(f"Candidate pool ({len(candidates)}): {candidates}")

# Greedy search
best_combo = {}
best_ret = 0
best_thr = -0.3
best_hp = 3

for iteration in range(8):  # Add up to 8 factors
    best_add = None
    best_add_ret = best_ret
    best_add_thr = best_thr
    best_add_hp = best_hp
    
    for name, direction in candidates.items():
        if name in best_combo:
            continue
        
        trial = {**best_combo, name: direction}
        
        for thr in [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]:
            for hp in [3, 5, 10]:
                try:
                    sig = generate_signals(all_features, trial, min_holding_days=hp, threshold=thr)
                    res = run_backtest(panel, sig, config.target_symbol, 5)
                    m = compute_metrics(res)
                    
                    if m["Ann_Return"] > best_add_ret and m["Max_DD"] > -0.45:
                        best_add_ret = m["Ann_Return"]
                        best_add = name
                        best_add_thr = thr
                        best_add_hp = hp
                except:
                    pass
    
    if best_add is None:
        print(f"  Iteration {iteration+1}: No improvement found. Stopping.")
        break
    
    best_combo[best_add] = candidates[best_add]
    best_ret = best_add_ret
    best_thr = best_add_thr
    best_hp = best_add_hp
    
    # Evaluate current best
    sig = generate_signals(all_features, best_combo, min_holding_days=best_hp, threshold=best_thr)
    res = run_backtest(panel, sig, config.target_symbol, 5)
    m = compute_metrics(res)
    
    print(f"  Iteration {iteration+1}: +{best_add} | Ret={m['Ann_Return']:.4f} Sharpe={m['Sharpe']:.3f} DD={m['Max_DD']:.3f} Exp={m['Exposure']:.2f} thr={best_thr} hp={best_hp}")

# ============================================================
# PHASE 3: Fine-tune threshold and holding period
# ============================================================
print(f"\n=== PHASE 3: Fine-tune on best combo ===")
print(f"Best combo so far: {best_combo}")

best_final_ret = best_ret
best_final_thr = best_thr
best_final_hp = best_hp

for thr in np.arange(-0.8, 0.1, 0.05):
    for hp in [1, 2, 3, 5, 7, 10, 15]:
        try:
            sig = generate_signals(all_features, best_combo, min_holding_days=hp, threshold=thr)
            res = run_backtest(panel, sig, config.target_symbol, 5)
            m = compute_metrics(res)
            
            if m["Ann_Return"] > best_final_ret and m["Max_DD"] > -0.45:
                best_final_ret = m["Ann_Return"]
                best_final_thr = thr
                best_final_hp = hp
                print(f"  Fine-tune: Ret={best_final_ret:.4f} Sharpe={m['Sharpe']:.3f} DD={m['Max_DD']:.3f} Exp={m['Exposure']:.2f} thr={thr:.2f} hp={hp}")
        except:
            pass

# Final evaluation
sig = generate_signals(all_features, best_combo, min_holding_days=best_final_hp, threshold=best_final_thr)
res = run_backtest(panel, sig, config.target_symbol, 5)
m = compute_metrics(res)

print(f"\n{'='*60}")
print(f"=== FINAL BEST CONFIGURATION ===")
print(f"{'='*60}")
print(f"Factors: {best_combo}")
print(f"Threshold: {best_final_thr:.2f}")
print(f"Hold Period: {best_final_hp}")
print(f"Ann Return: {m['Ann_Return']:.4f}")
print(f"BM Return:  {m['BM_Return']:.4f}")
print(f"Sharpe:     {m['Sharpe']:.4f}")
print(f"BM Sharpe:  {m['BM_Sharpe']:.4f}")
print(f"Max DD:     {m['Max_DD']:.4f}")
print(f"BM Max DD:  {m['BM_Max_DD']:.4f}")
print(f"Exposure:   {m['Exposure']:.4f}")
print(f"Trades:     {m['Num_Trades']:.0f}")
print(f"Win Rate:   {m['Win_Rate']:.4f}")
