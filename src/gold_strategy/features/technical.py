import pandas as pd
import numpy as np
from .transforms import rolling_zscore

def create_technical_features(panel: pd.DataFrame, target_symbol: str) -> pd.DataFrame:
    features = pd.DataFrame(index=panel.index)
    col = f"{target_symbol}_Adj_Close"
    vol = f"{target_symbol}_Volume"
    
    if col not in panel.columns:
        return features
    
    price = panel[col]
    daily_ret = price.pct_change()
        
    # === Momentum (past returns) ===
    for h in [5, 10, 20, 60, 120, 252]:
        features[f"Mom_{h}d"] = price / price.shift(h) - 1
    
    # === Momentum acceleration (2nd derivative) ===
    features["Mom_Accel_5_20"] = features["Mom_5d"] - features["Mom_20d"]
    features["Mom_Accel_20_60"] = features["Mom_20d"] - features["Mom_60d"]
    features["Mom_Accel_60_252"] = features["Mom_60d"] - features["Mom_252d"]
        
    # === Moving Average Gaps ===
    for ma in [10, 20, 60, 120, 200]:
        features[f"MA_Gap_{ma}d"] = price / price.rolling(ma).mean() - 1
        
    # === Moving Average Crosses ===
    ma10 = price.rolling(10).mean()
    ma20 = price.rolling(20).mean()
    ma50 = price.rolling(50).mean()
    ma60 = price.rolling(60).mean()
    ma120 = price.rolling(120).mean()
    ma200 = price.rolling(200).mean()
    features["MA_Cross_10_50"] = ma10 / ma50 - 1
    features["MA_Cross_20_60"] = ma20 / ma60 - 1
    features["MA_Cross_50_200"] = ma50 / ma200 - 1
    features["MA_Cross_60_120"] = ma60 / ma120 - 1
    features["MA_Cross_20_200"] = ma20 / ma200 - 1
    
    # === Realized Volatility ===
    for v in [10, 20, 60, 120]:
        features[f"Vol_{v}d"] = daily_ret.rolling(v).std() * np.sqrt(252)
        
    features["Vol_Ratio_10_60"] = features["Vol_10d"] / features["Vol_60d"]
    features["Vol_Ratio_20_60"] = features["Vol_20d"] / features["Vol_60d"]
    features["Vol_Ratio_20_120"] = features["Vol_20d"] / features["Vol_120d"]
    features["Vol_Z_60d"] = rolling_zscore(features["Vol_60d"], 252)
    
    # === Drawdown & Breakout ===
    for d in [10, 20, 60, 120, 252]:
        high = price.rolling(d).max()
        features[f"DD_{d}d"] = price / high - 1
        if d in [10, 20, 60]:
            features[f"Breakout_{d}d"] = (price >= high).astype(int)
    
    # === Distance from low ===
    for d in [20, 60, 252]:
        low = price.rolling(d).min()
        features[f"DistLow_{d}d"] = price / low - 1
    
    # === RSI-like ===
    for w in [14, 28]:
        up = daily_ret.clip(lower=0).rolling(w).mean()
        down = (-daily_ret.clip(upper=0)).rolling(w).mean()
        rs = up / down.replace(0, np.nan)
        features[f"RSI_{w}d"] = 100 - 100 / (1 + rs)
    
    # === Bollinger Band position ===
    for w in [20, 60]:
        ma = price.rolling(w).mean()
        std = price.rolling(w).std()
        features[f"BB_Pos_{w}d"] = (price - ma) / (2 * std)
    
    # === Rate of Change of Volume ===
    if vol in panel.columns:
        features["Vol_Z_20d"] = rolling_zscore(panel[vol], 20)
        vol10 = panel[vol].rolling(10).mean()
        vol20 = panel[vol].rolling(20).mean()
        vol60 = panel[vol].rolling(60).mean()
        features["Volume_Ratio_10_60"] = vol10 / vol60
        features["Volume_Ratio_20_60"] = vol20 / vol60
        # Price-Volume divergence: price up but volume down
        features["PV_Div_20d"] = features["Mom_20d"] - (vol20 / vol20.shift(20) - 1)
        
    # === Mean Reversion signals ===
    features["MeanRev_Z_20"] = rolling_zscore(price, 20)
    features["MeanRev_Z_60"] = rolling_zscore(price, 60)
    features["MeanRev_Z_120"] = rolling_zscore(price, 120)
    
    # === Trend Strength (ADX proxy using directional movement) ===
    high_price = panel.get(f"{target_symbol}_High", price * 1.005)  # approx if missing
    low_price = panel.get(f"{target_symbol}_Low", price * 0.995)
    tr = pd.concat([
        high_price - low_price,
        (high_price - price.shift(1)).abs(),
        (low_price - price.shift(1)).abs()
    ], axis=1).max(axis=1)
    features["ATR_Ratio_14_60"] = tr.rolling(14).mean() / tr.rolling(60).mean()
    
    # === Skewness of returns ===
    for w in [20, 60]:
        features[f"Skew_{w}d"] = daily_ret.rolling(w).skew()
    
    # === Kurtosis of returns ===
    features["Kurt_60d"] = daily_ret.rolling(60).kurt()
    
    # === Consecutive up/down days ===
    up_days = (daily_ret > 0).astype(int)
    down_days = (daily_ret < 0).astype(int)
    features["ConsecUp_5d"] = up_days.rolling(5).sum()
    features["ConsecDown_5d"] = down_days.rolling(5).sum()
    
    return features
