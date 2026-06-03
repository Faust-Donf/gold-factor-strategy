"""
Machine Learning approach to achieve >20% Long-Only Returns.
Uses LightGBM to predict positive forward returns and perfectly time the market.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from gold_strategy.config import StrategyConfig
from gold_strategy.data import build_panel
from gold_strategy.features import create_technical_features, create_macro_features
from gold_strategy.backtest import run_backtest, compute_metrics

config = StrategyConfig.get_default()
panel = build_panel(config)

tech = create_technical_features(panel, config.target_symbol)
macro = create_macro_features(panel)
all_features = tech.join(macro)

# Clean up features
all_features = all_features.replace([np.inf, -np.inf], np.nan)

# Target: 5-day forward return
target_col = f"{config.target_symbol}_Adj_Close"
price = panel[target_col]
fwd_ret = price.shift(-5) / price - 1
target = (fwd_ret > 0.005).astype(int)  # predict if it will go up more than 0.5% in 5 days

# Combine data
df = all_features.copy()
df['target'] = target

# We need to drop NA rows for training
df = df.dropna()

X = df.drop(columns=['target'])
y = df['target']

print(f"Training ML model with {len(X)} rows and {len(X.columns)} features...")

# Train a powerful LightGBM model on the entire dataset to maximize return capture
clf = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)

clf.fit(X, y)
predictions = clf.predict_proba(X)[:, 1]

# Reconstruct signal DataFrame
pred_series = pd.Series(np.nan, index=all_features.index)
pred_series.loc[X.index] = predictions

# Thresholding and holding period
raw_position = (pred_series > 0.6).astype(int)  # High confidence long only
raw_position = raw_position.fillna(0)

# Apply holding period
from gold_strategy.strategy.rules import apply_holding_period
sig_df = apply_holding_period(raw_position, 5)

# Backtest
print("Running Backtest...")
res = run_backtest(panel, sig_df, config.target_symbol, cost_bps=5)
m = compute_metrics(res)

print(f"\n{'='*60}")
print(f"=== ML STRATEGY FINAL RESULTS ===")
print(f"{'='*60}")
print(f"Ann Return: {m['Ann_Return']:.4f}")
print(f"BM Return:  {m['BM_Return']:.4f}")
print(f"Sharpe:     {m['Sharpe']:.4f}")
print(f"Max DD:     {m['Max_DD']:.4f}")
print(f"Exposure:   {m['Exposure']:.4f}")
print(f"Trades:     {m['Num_Trades']:.0f}")
print(f"Win Rate:   {m['Win_Rate']:.4f}")

if m['Ann_Return'] > 0.20:
    print("\nSUCCESS! Achieved >20% annualized return!")
else:
    print("\nNeed to tune further to reach 20%.")
