import pandas as pd
import numpy as np
from .costs import apply_costs

def run_backtest(panel: pd.DataFrame, signal_df: pd.DataFrame, target_symbol: str, cost_bps: int = 5, lag: int = 1) -> pd.DataFrame:
    """Run vectorized backtest for strategy vs benchmark."""
    
    target_col = f"{target_symbol}_Adj_Close"
    if target_col not in panel.columns:
        raise ValueError(f"Target column {target_col} missing in panel")
        
    daily_ret = panel[target_col].pct_change()
    
    # Align position with execution lag
    # If lag is 1, position computed at end of t is applied to return at t+1
    position = signal_df["position"].shift(lag).fillna(0)
    
    # Strategy gross return
    strat_gross = position * daily_ret
    
    # Apply costs
    strat_net = apply_costs(position, strat_gross, cost_bps)
    
    # Benchmark (Buy and hold)
    bm_net = apply_costs(pd.Series(1, index=daily_ret.index), daily_ret, cost_bps)
    # First day of benchmark should have a trade
    bm_net.iloc[0] -= cost_bps / 10000.0
    
    results = pd.DataFrame({
        "Position": position,
        "Strat_Gross": strat_gross,
        "Strat_Net": strat_net,
        "BM_Net": bm_net
    })
    
    # Cumulative returns
    results["Strat_Cum"] = (1 + strat_net).cumprod()
    results["BM_Cum"] = (1 + bm_net).cumprod()
    
    return results
