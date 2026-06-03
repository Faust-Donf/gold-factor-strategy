"""
Massive factor mining script.
Phase 1: Screen all individual factors in both directions.
Phase 2: Exhaustive combinatorial search over top factors × thresholds × holding periods.
"""
import pandas as pd
import numpy as np
import itertools
import sys
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

print(f"Total features available: {len(all_features.columns)}")
print(f"Data rows: {len(panel)}")

# ============================================================
# PHASE 1: Screen all individual factors
# ============================================================
print("\n=== PHASE 1: Individual Factor Screening ===")
individual_results = []

for feat in all_features.columns:
    for d in [1, -1]:
        try:
            sig = generate_signals(all_features, {feat: d}, min_holding_days=5, threshold=-0.3)
            res = run_backtest(panel, sig, config.target_symbol, config.base_cost_bps)
            m = compute_metrics(res)
            individual_results.append({
                "Feature": feat, "Dir": d,
                "Ann_Ret": m["Ann_Return"], "Sharpe": m["Sharpe"],
                "Max_DD": m["Max_DD"], "Exposure": m["Exposure"],
                "Trades": m["Num_Trades"]
            })
        except Exception:
            pass

ind_df = pd.DataFrame(individual_results)
# Sort by a composite score: heavily weight return, also consider sharpe and DD
ind_df["Score"] = ind_df["Ann_Ret"] * 2 + ind_df["Sharpe"] * 0.5 + (1 + ind_df["Max_DD"])
ind_df = ind_df.sort_values("Score", ascending=False)

print("\nTop 25 individual factors:")
print(ind_df.head(25).to_string(index=False))

# Pick top 15 factors for combinatorial search
top_n = 15
top_factors = {}
seen_names = set()
for _, row in ind_df.iterrows():
    name = row["Feature"]
    if name not in seen_names and len(top_factors) < top_n:
        top_factors[name] = int(row["Dir"])
        seen_names.add(name)

print(f"\nTop {top_n} factors for combo search: {top_factors}")

# ============================================================
# PHASE 2: Combinatorial search
# ============================================================
print("\n=== PHASE 2: Combinatorial Search ===")
keys = list(top_factors.keys())

best_ret = 0.0
best_config = None
best_metrics = None
tested = 0

thresholds = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
hold_periods = [3, 5, 10]

for r in range(2, 6):
    for comb in itertools.combinations(keys, r):
        selected = {k: top_factors[k] for k in comb}
        for thr in thresholds:
            for hp in hold_periods:
                try:
                    sig = generate_signals(all_features, selected, min_holding_days=hp, threshold=thr)
                    res = run_backtest(panel, sig, config.target_symbol, config.base_cost_bps)
                    m = compute_metrics(res)
                    tested += 1
                    
                    if m["Ann_Return"] > best_ret and m["Max_DD"] > -0.45:
                        best_ret = m["Ann_Return"]
                        best_config = {"factors": selected, "threshold": thr, "hold_period": hp}
                        best_metrics = m
                        print(f"[{tested}] NEW BEST Ret={best_ret:.4f} Sharpe={m['Sharpe']:.3f} DD={m['Max_DD']:.3f} Exp={m['Exposure']:.2f} | thr={thr} hp={hp} | {list(selected.keys())}")
                        
                except Exception:
                    pass

print(f"\nTotal combinations tested: {tested}")
print(f"\n=== BEST CONFIGURATION ===")
print(f"Factors: {best_config['factors']}")
print(f"Threshold: {best_config['threshold']}")
print(f"Hold Period: {best_config['hold_period']}")
print(f"Ann Return: {best_metrics['Ann_Return']:.4f}")
print(f"Sharpe: {best_metrics['Sharpe']:.4f}")
print(f"Max DD: {best_metrics['Max_DD']:.4f}")
print(f"Exposure: {best_metrics['Exposure']:.4f}")
print(f"Trades: {best_metrics['Num_Trades']:.0f}")
