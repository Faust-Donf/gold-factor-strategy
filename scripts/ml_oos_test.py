"""
Out-Of-Sample (OOS) testing for the LightGBM ML model.
Trains on data up to 2018, tests on 2019-2025.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from gold_strategy.config import StrategyConfig
from gold_strategy.data import build_panel
from gold_strategy.features import create_technical_features, create_macro_features
from gold_strategy.backtest import run_backtest, compute_metrics
from gold_strategy.strategy.rules import apply_holding_period

config = StrategyConfig.get_default()
panel = build_panel(config)

tech = create_technical_features(panel, config.target_symbol)
macro = create_macro_features(panel)
all_features = tech.join(macro)
all_features = all_features.replace([np.inf, -np.inf], np.nan)

# Target: 5-day forward return
target_col = f"{config.target_symbol}_Adj_Close"
price = panel[target_col]
fwd_ret = price.shift(-5) / price - 1
target = (fwd_ret > 0.005).astype(int)

df = all_features.copy()
df['target'] = target
df = df.dropna()

# Train/Test Split: Train before 2019, Test from 2019 onwards
split_date = pd.to_datetime('2019-01-01')
# Ensure df index is datetime for slicing
if not pd.api.types.is_datetime64_any_dtype(df.index):
    df.index = pd.to_datetime(df.index)

df_train = df[df.index < split_date]
df_test = df[df.index >= split_date]

X_train = df_train.drop(columns=['target'])
y_train = df_train['target']

X_test = df_test.drop(columns=['target'])
y_test = df_test['target']

print(f"Train size: {len(X_train)} (Up to 2018)")
print(f"Test size:  {len(X_test)} (2019 to Present)")

# Train the model strictly on train data
clf = lgb.LGBMClassifier(
    n_estimators=100,  # lower n_estimators to reduce overfitting
    learning_rate=0.05,
    max_depth=4,       # shallower trees
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)

clf.fit(X_train, y_train)

# Predict on Train and Test
pred_train = clf.predict_proba(X_train)[:, 1]
pred_test = clf.predict_proba(X_test)[:, 1]

# Reconstruct signal for TEST period only
pred_series_test = pd.Series(np.nan, index=X_test.index)
pred_series_test.loc[X_test.index] = pred_test

# Thresholding and holding period for TEST
raw_position_test = (pred_series_test > 0.55).astype(int)
raw_position_test = raw_position_test.fillna(0)
sig_df_test = apply_holding_period(raw_position_test, 5)

# Backtest on TEST period only
panel_test = panel.loc[sig_df_test.index]
res_test = run_backtest(panel_test, sig_df_test, config.target_symbol, cost_bps=5)
m_test = compute_metrics(res_test)

print(f"\n{'='*60}")
print(f"=== OUT-OF-SAMPLE (OOS) RESULTS (2019-Present) ===")
print(f"{'='*60}")
print(f"OOS Ann Return: {m_test['Ann_Return']:.4f}")
print(f"OOS BM Return:  {m_test['BM_Return']:.4f}")
print(f"OOS Sharpe:     {m_test['Sharpe']:.4f}")
print(f"OOS Max DD:     {m_test['Max_DD']:.4f}")
print(f"OOS Exposure:   {m_test['Exposure']:.4f}")
print(f"OOS Trades:     {m_test['Num_Trades']:.0f}")
print(f"OOS Win Rate:   {m_test['Win_Rate']:.4f}")

# Compare to Train
raw_position_train = (pd.Series(pred_train, index=X_train.index) > 0.55).astype(int)
sig_df_train = apply_holding_period(raw_position_train, 5)
panel_train = panel.loc[sig_df_train.index]
res_train = run_backtest(panel_train, sig_df_train, config.target_symbol, cost_bps=5)
m_train = compute_metrics(res_train)
print(f"\n[IN-SAMPLE (2004-2018)] Ann Ret: {m_train['Ann_Return']:.4f} | BM: {m_train['BM_Return']:.4f} | Sharpe: {m_train['Sharpe']:.4f}")
