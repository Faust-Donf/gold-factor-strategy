import pandas as pd
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

results = []

for feat in all_features.columns:
    # Test positive direction
    sig_pos = generate_signals(all_features, {feat: 1}, min_holding_days=5)
    res_pos = run_backtest(panel, sig_pos, config.target_symbol, config.base_cost_bps)
    m_pos = compute_metrics(res_pos)
    
    # Test negative direction
    sig_neg = generate_signals(all_features, {feat: -1}, min_holding_days=5)
    res_neg = run_backtest(panel, sig_neg, config.target_symbol, config.base_cost_bps)
    m_neg = compute_metrics(res_neg)
    
    results.append({
        "Feature": feat,
        "Dir": 1,
        "Ann_Ret": m_pos["Ann_Return"],
        "Sharpe": m_pos["Sharpe"],
        "Max_DD": m_pos["Max_DD"]
    })
    
    results.append({
        "Feature": feat,
        "Dir": -1,
        "Ann_Ret": m_neg["Ann_Return"],
        "Sharpe": m_neg["Sharpe"],
        "Max_DD": m_neg["Max_DD"]
    })

res_df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
print(res_df.head(20).to_string())
