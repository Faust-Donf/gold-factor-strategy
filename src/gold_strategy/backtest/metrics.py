import pandas as pd
import numpy as np

def compute_drawdown(cum_ret: pd.Series) -> pd.Series:
    """Compute drawdown series."""
    peak = cum_ret.cummax()
    return cum_ret / peak - 1

def compute_metrics(backtest_results: pd.DataFrame) -> dict:
    """Compute standard backtest performance metrics."""
    metrics = {}
    
    strat = backtest_results["Strat_Net"]
    bm = backtest_results["BM_Net"]
    pos = backtest_results["Position"]
    
    days = len(strat.dropna())
    years = days / 252.0
    
    # Returns
    strat_ann = (1 + strat).prod() ** (1 / years) - 1 if years > 0 else 0
    bm_ann = (1 + bm).prod() ** (1 / years) - 1 if years > 0 else 0
    
    # Volatility
    strat_vol = strat.std() * np.sqrt(252)
    bm_vol = bm.std() * np.sqrt(252)
    
    # Sharpe (rf=0)
    strat_sharpe = strat_ann / strat_vol if strat_vol > 0 else 0
    bm_sharpe = bm_ann / bm_vol if bm_vol > 0 else 0
    
    # Drawdown
    strat_dd = compute_drawdown((1 + strat).cumprod())
    bm_dd = compute_drawdown((1 + bm).cumprod())
    
    strat_max_dd = strat_dd.min()
    bm_max_dd = bm_dd.min()
    
    # Trade stats
    trades = pos.diff().abs().sum() / 2 # one round trip = 1 trade
    exposure = pos.mean()
    win_rate = (strat > 0).sum() / (strat != 0).sum() if (strat != 0).sum() > 0 else 0
    
    metrics = {
        "Ann_Return": strat_ann,
        "BM_Return": bm_ann,
        "Ann_Vol": strat_vol,
        "BM_Vol": bm_vol,
        "Sharpe": strat_sharpe,
        "BM_Sharpe": bm_sharpe,
        "Max_DD": strat_max_dd,
        "BM_Max_DD": bm_max_dd,
        "Exposure": exposure,
        "Num_Trades": trades,
        "Win_Rate": win_rate
    }
    
    return metrics
