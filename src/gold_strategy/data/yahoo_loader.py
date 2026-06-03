import yfinance as yf
import pandas as pd
from datetime import date
from typing import List, Optional
import os
import requests
import time

def load_yahoo_data(symbols: List[str], start_date: date, end_date: Optional[date] = None, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Download daily adjusted close prices for given symbols using yf.download."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    
    time.sleep(2) # avoid instant blocking
    
    try:
        raw = yf.download(symbols, start=start_date, end=end_date, session=session, auto_adjust=False, progress=False)
    except Exception as e:
        print(f"yfinance download failed: {e}")
        return pd.DataFrame()
        
    if raw.empty:
        return pd.DataFrame()
        
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
        
    panel = pd.DataFrame(index=raw.index)
    
    for sym in symbols:
        if len(symbols) > 1:
            if 'Adj Close' in raw and sym in raw['Adj Close']:
                panel[f"{sym}_Adj_Close"] = raw['Adj Close'][sym]
            if 'Volume' in raw and sym in raw['Volume'] and sym == "GLD":
                panel[f"{sym}_Volume"] = raw['Volume'][sym]
        else:
            if 'Adj Close' in raw:
                panel[f"{sym}_Adj_Close"] = raw['Adj Close']
            if 'Volume' in raw and sym == "GLD":
                panel[f"{sym}_Volume"] = raw['Volume']
                
    panel = panel.dropna(how='all')
    
    if "GLD_Adj_Close" in panel.columns:
        panel = panel.dropna(subset=["GLD_Adj_Close"])
        
    return panel
