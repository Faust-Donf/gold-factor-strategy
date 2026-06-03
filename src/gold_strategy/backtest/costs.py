import pandas as pd

def apply_costs(position: pd.Series, base_return: pd.Series, cost_bps: int = 5) -> pd.Series:
    """Apply trading costs when position changes."""
    cost_dec = cost_bps / 10000.0
    
    # trades occur when position changes
    # assuming long-only, position diff is trade size
    trades = position.diff().fillna(0).abs()
    
    # subtract cost
    net_return = base_return - (trades * cost_dec)
    return net_return
