import pandas as pd

def create_macro_features(panel: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=panel.index)
    
    # Yields
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
        
    # Markets
    if "UUP_Adj_Close" in panel.columns:
        features["USD_Mom_20d"] = panel["UUP_Adj_Close"] / panel["UUP_Adj_Close"].shift(20) - 1
        
    if "^VIX_Adj_Close" in panel.columns:
        features["VIX_Level"] = panel["^VIX_Adj_Close"]
        features["VIX_Chg_20d"] = panel["^VIX_Adj_Close"].diff(20)
        
    if "SPY_Adj_Close" in panel.columns:
        features["SPY_Mom_20d"] = panel["SPY_Adj_Close"] / panel["SPY_Adj_Close"].shift(20) - 1
        spy_high = panel["SPY_Adj_Close"].rolling(252).max()
        features["SPY_DD_252d"] = panel["SPY_Adj_Close"] / spy_high - 1
        
    if "USO_Adj_Close" in panel.columns:
        features["Oil_Mom_20d"] = panel["USO_Adj_Close"] / panel["USO_Adj_Close"].shift(20) - 1
        
    return features
