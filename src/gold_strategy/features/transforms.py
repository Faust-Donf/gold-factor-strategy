import pandas as pd

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling z-score of a series."""
    roll = series.rolling(window)
    return (series - roll.mean()) / roll.std()

def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling percentile rank."""
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize series at given quantiles."""
    return series.clip(lower=series.quantile(lower), upper=series.quantile(upper))
