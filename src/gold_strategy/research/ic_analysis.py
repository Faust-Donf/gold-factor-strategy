import pandas as pd
import numpy as np

def compute_ic(features: pd.DataFrame, targets: pd.DataFrame, method: str = 'spearman') -> pd.DataFrame:
    """Compute Information Coefficient (IC) between features and target returns."""
    ic_dict = {}
    for target_col in targets.columns:
        ic_dict[target_col] = features.corrwith(targets[target_col], method=method)
    return pd.DataFrame(ic_dict)

def compute_ic_summary(ic_series: pd.Series) -> pd.Series:
    """Compute summary stats for a single factor's IC over time.
    For this we'd need rolling IC or daily IC, but standard is to just take 
    overall IC if we just used `corrwith`. Wait, `corrwith` gives mean IC across all time 
    if done on panel, but usually IC is computed per cross-section. 
    Since this is a single asset strategy, the 'IC' is just the time-series correlation
    between the factor and the forward return.
    So compute_ic above gives the overall time-series correlation.
    
    If we want ICIR, we need rolling correlation or chunked correlation.
    For single asset, we can compute rolling correlation over 252 days, then summarize.
    """
    pass

def compute_timeseries_ic_stats(features: pd.DataFrame, targets: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Compute time-series rolling IC stats for single asset."""
    stats = []
    for target_col in targets.columns:
        for factor in features.columns:
            # rolling correlation
            rolling_corr = features[factor].rolling(window).corr(targets[target_col])
            mean_ic = rolling_corr.mean()
            std_ic = rolling_corr.std()
            icir = mean_ic / std_ic if std_ic != 0 else 0
            pos_ratio = (rolling_corr > 0).mean()
            
            stats.append({
                "Target": target_col,
                "Factor": factor,
                "Mean_IC": mean_ic,
                "Std_IC": std_ic,
                "ICIR": icir,
                "Positive_IC_Ratio": pos_ratio
            })
            
    return pd.DataFrame(stats)
