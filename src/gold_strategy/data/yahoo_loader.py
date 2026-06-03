import yfinance as yf
import pandas as pd
from datetime import date
from typing import List, Optional
import os

def load_yahoo_data(symbols: List[str], start_date: date, end_date: Optional[date] = None, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Download daily adjusted close prices for given symbols."""
    # We only need Adj Close and Volume for target symbol, Adj Close for others.
    # To keep it simple, we download all and then extract what we need.
    
    dfs = []
    for sym in symbols:
        cache_file = os.path.join(cache_dir, f"{sym}_yahoo.csv") if cache_dir else None
        
        if cache_file and os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Ensure index is timezone naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        else:
            ticker = yf.Ticker(sym)
            try:
                df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            except Exception:
                continue
            if df.empty:
                continue
            
            # Ensure index is timezone naive date
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            
            if cache_file:
                df.to_csv(cache_file)
                
        # Rename columns to standard format
        if sym == "GLD":
            # For GLD, keep Adj Close and Volume
            df_sym = pd.DataFrame({
                "GLD_Adj_Close": df["Adj Close"],
                "GLD_Volume": df["Volume"]
            })
        else:
            df_sym = pd.DataFrame({
                f"{sym}_Adj_Close": df["Adj Close"]
            })
            
        dfs.append(df_sym)
        
    if not dfs:
        return pd.DataFrame()
        
    panel = pd.concat(dfs, axis=1)
    return panel
