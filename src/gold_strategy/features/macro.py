import pandas as pd
import numpy as np

def create_macro_features(panel: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=panel.index)
    
    # === Cross-asset momentum at multiple horizons ===
    cross_assets = {
        "SPY": "SPY_Adj_Close",
        "VIX": "^VIX_Adj_Close",
        "UUP": "UUP_Adj_Close",
        "USO": "USO_Adj_Close",
        "TLT": "TLT_Adj_Close",
        "IEF": "IEF_Adj_Close",
    }
    
    for name, col in cross_assets.items():
        if col not in panel.columns:
            continue
        p = panel[col]
        for h in [5, 10, 20, 60]:
            features[f"{name}_Mom_{h}d"] = p / p.shift(h) - 1
        # Volatility
        r = p.pct_change()
        features[f"{name}_Vol_20d"] = r.rolling(20).std() * np.sqrt(252)
        # Distance from 252d high
        high252 = p.rolling(252).max()
        features[f"{name}_DD_252d"] = p / high252 - 1
        # Z-score
        features[f"{name}_Z_60d"] = (p - p.rolling(60).mean()) / p.rolling(60).std()
    
    # === GLD relative to other assets (relative strength) ===
    gld = panel.get("GLD_Adj_Close")
    if gld is not None:
        for name, col in cross_assets.items():
            if col not in panel.columns or name == "VIX":
                continue
            other = panel[col]
            ratio = gld / other
            for h in [20, 60]:
                features[f"GLD_vs_{name}_RS_{h}d"] = ratio / ratio.shift(h) - 1
    
    # === VIX term structure proxy (VIX level vs its MA) ===
    if "^VIX_Adj_Close" in panel.columns:
        vix = panel["^VIX_Adj_Close"]
        features["VIX_Level"] = vix
        features["VIX_Chg_5d"] = vix.diff(5)
        features["VIX_Chg_20d"] = vix.diff(20)
        features["VIX_Z_60d"] = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
        features["VIX_MA_Ratio"] = vix / vix.rolling(20).mean()
        # VIX spike detection
        features["VIX_Spike"] = (vix / vix.shift(1) - 1).rolling(5).max()
    
    # === Bond market signals ===
    if "TLT_Adj_Close" in panel.columns and "IEF_Adj_Close" in panel.columns:
        tlt = panel["TLT_Adj_Close"]
        ief = panel["IEF_Adj_Close"]
        # Duration spread (long vs intermediate)
        ratio = tlt / ief
        features["Bond_Duration_Spread_20d"] = ratio / ratio.shift(20) - 1
        features["Bond_Duration_Spread_60d"] = ratio / ratio.shift(60) - 1
    
    # === Gold-Oil ratio (historically mean-reverting) ===
    if "USO_Adj_Close" in panel.columns and gld is not None:
        oil = panel["USO_Adj_Close"]
        gold_oil = gld / oil
        features["GoldOil_Ratio_Z"] = (gold_oil - gold_oil.rolling(120).mean()) / gold_oil.rolling(120).std()
        features["GoldOil_Mom_20d"] = gold_oil / gold_oil.shift(20) - 1
    
    # === Risk-on / Risk-off regime ===
    if "SPY_Adj_Close" in panel.columns and "TLT_Adj_Close" in panel.columns:
        spy = panel["SPY_Adj_Close"]
        tlt = panel["TLT_Adj_Close"]
        # SPY/TLT ratio as risk appetite proxy
        risk_ratio = spy / tlt
        features["RiskOn_Mom_20d"] = risk_ratio / risk_ratio.shift(20) - 1
        features["RiskOn_Mom_60d"] = risk_ratio / risk_ratio.shift(60) - 1
        features["RiskOn_Z_60d"] = (risk_ratio - risk_ratio.rolling(60).mean()) / risk_ratio.rolling(60).std()
    
    # === Dollar strength impact ===
    if "UUP_Adj_Close" in panel.columns:
        uup = panel["UUP_Adj_Close"]
        features["USD_Mom_5d"] = uup / uup.shift(5) - 1
        features["USD_Mom_20d"] = uup / uup.shift(20) - 1
        features["USD_Mom_60d"] = uup / uup.shift(60) - 1
        features["USD_Z_120d"] = (uup - uup.rolling(120).mean()) / uup.rolling(120).std()
    
    # === Correlation regime (rolling GLD-SPY correlation) ===
    if "SPY_Adj_Close" in panel.columns and gld is not None:
        gld_ret = gld.pct_change()
        spy_ret = panel["SPY_Adj_Close"].pct_change()
        features["GLD_SPY_Corr_60d"] = gld_ret.rolling(60).corr(spy_ret)
        features["GLD_SPY_Corr_20d"] = gld_ret.rolling(20).corr(spy_ret)
    
    # === FRED yields (if available) ===
    if "FRED_DGS10" in panel.columns:
        features["DGS10_Level"] = panel["FRED_DGS10"]
        features["DGS10_Chg_20d"] = panel["FRED_DGS10"].diff(20)
        
    if "FRED_DFII10" in panel.columns:
        features["RealYield10_Level"] = panel["FRED_DFII10"]
        features["RealYield10_Chg_20d"] = panel["FRED_DFII10"].diff(20)
        
    if "FRED_T10YIE" in panel.columns:
        features["Breakeven10_Level"] = panel["FRED_T10YIE"]
        features["Breakeven10_Chg_20d"] = panel["FRED_T10YIE"].diff(20)
        
    if "FRED_T10Y2Y" in panel.columns:
        features["YieldCurve_Spread"] = panel["FRED_T10Y2Y"]
        features["YieldCurve_Chg_20d"] = panel["FRED_T10Y2Y"].diff(20)
        
    return features
