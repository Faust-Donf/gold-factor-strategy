import pandas as pd

def export_latest_signal(signal_df: pd.DataFrame, out_csv: str):
    """Export the most recent signal."""
    if signal_df.empty:
        return
        
    latest = signal_df.iloc[[-1]].copy()
    latest.index.name = "Date"
    latest.to_csv(out_csv)
    
def export_signal_html(signal_df: pd.DataFrame, out_html: str):
    if signal_df.empty:
        return
        
    latest = signal_df.iloc[[-1]].copy()
    latest.index.name = "Date"
    latest.to_html(out_html)
