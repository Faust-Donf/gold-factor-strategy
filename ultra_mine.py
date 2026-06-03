"""
Ultra-aggressive mining: High-exposure crash-avoidance strategy.
Instead of "when to buy", focus on "when to NOT hold".
Stay long by default, go cash only when crash signals fire.

Also explores:
- Shorter z-score windows (60, 120)
- Voting-based signals instead of z-score averaging
- Asymmetric entry/exit thresholds
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

print(f"Features: {len(all_features.columns)}, Rows: {len(panel)}")

# ============================================================
# Custom signal generator with multiple modes
# ============================================================
def generate_custom_signal(features, factors, mode="zscore", threshold=-0.3, 
                           hold_days=5, zscore_window=252):
    """
    Modes:
    - zscore: standard z-score average > threshold
    - vote: majority voting (>= vote_threshold fraction bullish)  
    - crash_avoid: default long, go cash only when score < crash_threshold
    """
    std = standardize_factors(features, factors, window=zscore_window)
    
    if mode == "zscore":
        score = std.mean(axis=1)
        raw = (score > threshold).astype(int)
    elif mode == "vote":
        # Each factor votes: z > 0 = bullish, z < 0 = bearish
        votes = (std > 0).astype(int)
        vote_pct = votes.mean(axis=1)
        raw = (vote_pct >= threshold).astype(int)
    elif mode == "crash_avoid":
        # Default is LONG. Go cash only when score is very negative.
        score = std.mean(axis=1)
        raw = (score > threshold).astype(int)
        # Override: if score is NaN (warmup), stay long
        raw = raw.fillna(1).astype(int)
    else:
        score = std.mean(axis=1)
        raw = (score > threshold).astype(int)
    
    sig_df = apply_holding_period(raw, hold_days)
    return sig_df

def eval_config(features, panel, factors, mode, threshold, hold_days, zscore_window, cost=5):
    try:
        sig = generate_custom_signal(features, factors, mode, threshold, hold_days, zscore_window)
        res = run_backtest(panel, sig, config.target_symbol, cost)
        m = compute_metrics(res)
        return m
    except:
        return None

# ============================================================
# PHASE 1: Screen crash-avoidance factors
# Find factors that, when inverted and used with very negative threshold,
# give HIGH exposure + HIGH return
# ============================================================
print("\n=== PHASE 1: Crash-Avoidance Factor Screen ===")
results = []

for feat in all_features.columns:
    for d in [1, -1]:
        for mode in ["crash_avoid", "zscore"]:
            for thr in [-0.8, -0.6, -0.4]:
                for zw in [60, 120, 252]:
                    m = eval_config(all_features, panel, {feat: d}, mode, thr, 3, zw)
                    if m and m["Exposure"] > 0.7 and m["Max_DD"] > -0.45:
                        results.append({
                            "Feature": feat, "Dir": d, "Mode": mode, "Thr": thr, "ZW": zw,
                            "Ret": m["Ann_Return"], "Sharpe": m["Sharpe"],
                            "DD": m["Max_DD"], "Exp": m["Exposure"], "Trades": m["Num_Trades"]
                        })

df = pd.DataFrame(results).sort_values("Ret", ascending=False)
print(f"Total configs tested: {len(results)}")
print("\nTop 30:")
print(df.head(30).to_string(index=False))

# ============================================================
# PHASE 2: Greedy forward selection with all modes
# ============================================================
print("\n=== PHASE 2: Greedy Forward Selection (Multi-Mode) ===")

# Get top 20 unique factors
candidates = {}
seen = set()
for _, row in df.iterrows():
    name = row["Feature"]
    if name not in seen and len(candidates) < 20:
        candidates[name] = {"dir": int(row["Dir"]), "mode": row["Mode"], "zw": int(row["ZW"])}
        seen.add(name)

print(f"Candidates: {list(candidates.keys())}")

best_combo = {}
best_ret = 0
best_mode = "crash_avoid"
best_thr = -0.5
best_hp = 3
best_zw = 120

for iteration in range(8):
    best_add = None
    best_add_ret = best_ret
    local_best_thr = best_thr
    local_best_hp = best_hp
    local_best_mode = best_mode
    local_best_zw = best_zw
    
    for name, info in candidates.items():
        if name in best_combo:
            continue
        
        trial = {**best_combo, name: info["dir"]}
        
        for mode in ["crash_avoid", "zscore"]:
            for thr in [-0.8, -0.6, -0.5, -0.4, -0.3, -0.2]:
                for hp in [3, 5, 10]:
                    for zw in [60, 120, 252]:
                        m = eval_config(all_features, panel, trial, mode, thr, hp, zw)
                        if m and m["Ann_Return"] > best_add_ret and m["Max_DD"] > -0.45:
                            best_add_ret = m["Ann_Return"]
                            best_add = name
                            local_best_thr = thr
                            local_best_hp = hp
                            local_best_mode = mode
                            local_best_zw = zw
    
    if best_add is None:
        print(f"  Iter {iteration+1}: No improvement. Stopping.")
        break
    
    best_combo[best_add] = candidates[best_add]["dir"]
    best_ret = best_add_ret
    best_thr = local_best_thr
    best_hp = local_best_hp
    best_mode = local_best_mode
    best_zw = local_best_zw
    
    m = eval_config(all_features, panel, best_combo, best_mode, best_thr, best_hp, best_zw)
    print(f"  Iter {iteration+1}: +{best_add} | Ret={m['Ann_Return']:.4f} Sharpe={m['Sharpe']:.3f} DD={m['Max_DD']:.3f} Exp={m['Exposure']:.2f} mode={best_mode} thr={best_thr} hp={best_hp} zw={best_zw}")

# ============================================================
# PHASE 3: Ultra-fine tuning
# ============================================================
print(f"\n=== PHASE 3: Ultra Fine-Tune ===")
print(f"Combo: {best_combo}")

best_final_ret = best_ret
best_final_config = {"thr": best_thr, "hp": best_hp, "mode": best_mode, "zw": best_zw}

for thr in np.arange(-1.0, 0.1, 0.05):
    for hp in [1, 2, 3, 5, 7, 10]:
        for zw in [40, 60, 90, 120, 180, 252]:
            for mode in ["crash_avoid", "zscore"]:
                m = eval_config(all_features, panel, best_combo, mode, thr, hp, zw)
                if m and m["Ann_Return"] > best_final_ret and m["Max_DD"] > -0.45:
                    best_final_ret = m["Ann_Return"]
                    best_final_config = {"thr": thr, "hp": hp, "mode": mode, "zw": zw}
                    print(f"  Ret={best_final_ret:.4f} Sharpe={m['Sharpe']:.3f} DD={m['Max_DD']:.3f} Exp={m['Exposure']:.2f} | {best_final_config}")

# Final
cfg = best_final_config
m = eval_config(all_features, panel, best_combo, cfg["mode"], cfg["thr"], cfg["hp"], cfg["zw"])

print(f"\n{'='*60}")
print(f"=== FINAL BEST ===")
print(f"{'='*60}")
print(f"Factors:    {best_combo}")
print(f"Mode:       {cfg['mode']}")
print(f"Threshold:  {cfg['thr']:.2f}")
print(f"Hold Days:  {cfg['hp']}")
print(f"ZScore Win: {cfg['zw']}")
print(f"Ann Return: {m['Ann_Return']:.4f}")
print(f"BM Return:  {m['BM_Return']:.4f}")
print(f"Sharpe:     {m['Sharpe']:.4f}")
print(f"Max DD:     {m['Max_DD']:.4f}")
print(f"Exposure:   {m['Exposure']:.4f}")
print(f"Trades:     {m['Num_Trades']:.0f}")
print(f"Win Rate:   {m['Win_Rate']:.4f}")
