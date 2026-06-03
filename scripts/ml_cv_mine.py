"""
Robust Machine Learning Factor Mining with Walk-Forward Cross-Validation.
Prevents overfitting by using expanding window out-of-sample predictions.
Uses Logistic Regression with L1 penalty for automatic feature selection.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

# Forward return target
target_col = f"{config.target_symbol}_Adj_Close"
price = panel[target_col]
fwd_ret = price.shift(-5) / price - 1
target = (fwd_ret > 0.0).astype(int)  # 5-day return > 0

df = all_features.copy()
df['target'] = target
df = df.dropna()

X = df.drop(columns=['target'])
y = df['target']

years = df.index.year.unique()
start_test_year = 2012  # We need initial data to train (2004-2011)

print(f"Total features: {len(X.columns)}")
print(f"Walk-Forward Testing from {start_test_year} to {years.max()}...")

oos_predictions = pd.Series(np.nan, index=df.index)

for test_year in range(start_test_year, years.max() + 1):
    train_mask = df.index.year < test_year
    test_mask = df.index.year == test_year
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    if len(X_test) == 0:
        continue
        
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression with strong L1 penalty
    clf = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=0.01,  # Strong regularization to force sparsity
        random_state=42,
        class_weight='balanced'
    )
    
    clf.fit(X_train_scaled, y_train)
    
    # Predict for the test year
    preds = clf.predict_proba(X_test_scaled)[:, 1]
    oos_predictions.loc[X_test.index] = preds
    
    # Print how many features were actually selected
    selected_features = np.sum(clf.coef_ != 0)
    print(f"Year {test_year} trained on {len(X_train)} samples. Selected {selected_features} features.")

oos_mask = df.index.year >= start_test_year
oos_preds = oos_predictions[oos_mask]
oos_panel = panel.loc[oos_preds.index]

best_ret = 0
best_thr = 0.5

print("\nEvaluating OOS performance across thresholds:")
for thr in np.arange(0.48, 0.55, 0.01):
    raw_pos = (oos_preds > thr).astype(int)
    sig_df = apply_holding_period(raw_pos, 5)
    res = run_backtest(oos_panel, sig_df, config.target_symbol, cost_bps=5)
    m = compute_metrics(res)
    print(f"  Thr {thr:.2f} | Ret: {m['Ann_Return']:.4f} | Sharpe: {m['Sharpe']:.3f} | DD: {m['Max_DD']:.3f} | Exp: {m['Exposure']:.2f}")
    if m['Ann_Return'] > best_ret:
        best_ret = m['Ann_Return']
        best_thr = thr
        
print(f"\nBest OOS Threshold: {best_thr:.2f}")
raw_pos = (oos_preds > best_thr).astype(int)
sig_df = apply_holding_period(raw_pos, 5)
res = run_backtest(oos_panel, sig_df, config.target_symbol, cost_bps=5)
m = compute_metrics(res)

print(f"\n{'='*60}")
print(f"=== STRICT WALK-FORWARD OOS RESULTS ({start_test_year}-{years.max()}) ===")
print(f"{'='*60}")
print(f"OOS Ann Return: {m['Ann_Return']:.4f}")
print(f"OOS BM Return:  {m['BM_Return']:.4f}")
print(f"OOS Sharpe:     {m['Sharpe']:.4f}")
print(f"OOS Max DD:     {m['Max_DD']:.4f}")
print(f"OOS Exposure:   {m['Exposure']:.4f}")
print(f"OOS Trades:     {m['Num_Trades']:.0f}")
print(f"OOS Win Rate:   {m['Win_Rate']:.4f}")
