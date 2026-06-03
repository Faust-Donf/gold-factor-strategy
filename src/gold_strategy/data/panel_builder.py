import pandas as pd
import os
from typing import Optional

from gold_strategy.config import StrategyConfig
from gold_strategy.data.yahoo_loader import load_yahoo_data
from gold_strategy.data.fred_loader import load_fred_data
from gold_strategy.data.local_loader import load_local_data
from gold_strategy.data.quality import check_data_quality, clean_panel

def build_panel(config: StrategyConfig, cache_dir: Optional[str] = None, local_csv: Optional[str] = None) -> pd.DataFrame:
    """Build the main daily research panel."""
    
    if local_csv is None and hasattr(config, 'local_csv'):
        local_csv = config.local_csv
        
    if local_csv and os.path.exists(local_csv):
        print(f"Loading data from local CSV: {local_csv}")
        panel = pd.read_csv(local_csv, index_col=0, parse_dates=True)
        # FRED data needs to be merged in still if it's missing from the panel
        if "FRED_DGS10" not in panel.columns:
            fred_panel = load_fred_data(config.fred_series, config.start_date, None, cache_dir)
            if not fred_panel.empty:
                panel = panel.join(fred_panel, how="left")
                panel.update(panel.ffill())
        return panel
        panel = load_local_data(local_csv)
    else:
        # 1. Download target and market data
        symbols = [config.target_symbol] + config.market_symbols
        yahoo_df = load_yahoo_data(symbols, config.start_date, cache_dir=cache_dir)
        
        # 2. Download macro data
        fred_df = load_fred_data(config.fred_series, config.start_date, cache_dir=cache_dir)
        
        # 3. Combine. Join all on outer to get all dates first.
        # Actually, best practice: reindex to target symbol's dates, but macro might be on weekends/holidays.
        # We will merge outer, then forward fill macro, then filter to target symbol's dates.
        panel = pd.concat([yahoo_df, fred_df], axis=1)
        
        # Forward fill ONLY macro features (FRED_* columns)
        fred_cols = [c for c in panel.columns if c.startswith("FRED_")]
        panel[fred_cols] = panel[fred_cols].ffill()
        
    # Clean panel to keep only target trading days and ensure monotonic
    panel = clean_panel(panel, config.target_symbol)
    
    # Optional: ensure we have a return column
    target_col = f"{config.target_symbol}_Adj_Close"
    target_ret = f"{config.target_symbol}_Return"
    if target_col in panel.columns and target_ret not in panel.columns:
        panel[target_ret] = panel[target_col].pct_change()
        
    # Quality check
    q_res = check_data_quality(panel, config.target_symbol)
    if not q_res["is_valid"]:
        # We just log or print warnings, don't crash
        print(f"Data Quality Issues Found: {q_res['issues']}")
        
    return panel
