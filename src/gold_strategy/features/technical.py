import pandas as pd
import numpy as np
from .transforms import rolling_zscore

def create_technical_features(panel: pd.DataFrame, target_symbol: str) -> pd.DataFrame:
    features = pd.DataFrame(index=panel.index)
    col = f"{target_symbol}_Adj_Close"
    vol = f"{target_symbol}_Volume"
    
    if col not in panel.columns:
        return features
        
    # Momentum (past returns)
    for h in [5, 10, 20, 60, 120, 252]:
        features[f"Mom_{h}d"] = panel[col] / panel[col].shift(h) - 1
        
    # Moving Average Gaps
    for ma in [20, 60, 120]:
        features[f"MA_Gap_{ma}d"] = panel[col] / panel[col].rolling(ma).mean() - 1
        
    # Moving Average Crosses
    ma20 = panel[col].rolling(20).mean()
    ma60 = panel[col].rolling(60).mean()
    ma120 = panel[col].rolling(120).mean()
    features["MA_Cross_20_60"] = ma20 / ma60 - 1
    features["MA_Cross_60_120"] = ma60 / ma120 - 1
    
    # Realized Volatility
    daily_ret = panel[col].pct_change()
    for v in [20, 60]:
        features[f"Vol_{v}d"] = daily_ret.rolling(v).std() * np.sqrt(252)
        
    features["Vol_Ratio_20_60"] = features["Vol_20d"] / features["Vol_60d"]
    
    # Drawdown & Breakout
    for d in [20, 60, 252]:
        high = panel[col].rolling(d).max()
        features[f"DD_{d}d"] = panel[col] / high - 1
        if d in [20, 60]:
            features[f"Breakout_{d}d"] = (panel[col] >= high).astype(int)
            
    # Volume
    if vol in panel.columns:
        features["Vol_Z_20d"] = rolling_zscore(panel[vol], 20)
        vol20 = panel[vol].rolling(20).mean()
        vol60 = panel[vol].rolling(60).mean()
        features["Volume_Ratio_20_60"] = vol20 / vol60
        
    return features
