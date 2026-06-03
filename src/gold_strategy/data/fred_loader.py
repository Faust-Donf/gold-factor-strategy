try:
    import pandas_datareader.data as web
except Exception:
    web = None

import pandas as pd

from datetime import date
from typing import Dict, Optional
import os

def load_fred_data(series_map: Dict[str, str], start_date: date, end_date: Optional[date] = None, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Download macro series from FRED."""
    dfs = []
    
    for name, series_id in series_map.items():
        cache_file = os.path.join(cache_dir, f"{series_id}_fred.csv") if cache_dir else None
        
        if cache_file and os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        else:
            if web is None:
                df = pd.DataFrame()
            else:
                try:
                    df = web.DataReader(series_id, "fred", start_date, end_date)
                except Exception as e:
                    # If fail, return empty
                    df = pd.DataFrame()
                
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                if cache_file:
                    df.to_csv(cache_file)
                    
        if not df.empty:
            df_sym = pd.DataFrame({
                f"FRED_{name}": df[series_id]
            })
            dfs.append(df_sym)
            
    if not dfs:
        return pd.DataFrame()
        
    panel = pd.concat(dfs, axis=1)
    return panel
