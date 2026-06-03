import pandas as pd
import os

def load_local_data(filepath: str) -> pd.DataFrame:
    """Load pre-processed CSV data if available, allowing skipping downloads."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Local file not found: {filepath}")
        
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    df.index = pd.to_datetime(df.index).normalize()
    return df
