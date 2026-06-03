import pandas as pd
from datetime import date
from typing import Callable, Any

def run_train_test_split(panel: pd.DataFrame, train_end: date, test_start: date, pipeline_func: Callable) -> dict:
    """Run fixed train/test split. pipeline_func should return signal."""
    
    train_panel = panel.loc[:pd.Timestamp(train_end)]
    test_panel = panel.loc[pd.Timestamp(test_start):]
    
    train_sig = pipeline_func(train_panel)
    test_sig = pipeline_func(test_panel)
    
    return {
        "train": {"panel": train_panel, "signal": train_sig},
        "test": {"panel": test_panel, "signal": test_sig}
    }

def run_walk_forward(panel: pd.DataFrame, train_years: int, test_years: int, pipeline_func: Callable) -> list:
    """Run walk forward validation."""
    
    results = []
    
    start_year = panel.index.year.min()
    end_year = panel.index.year.max()
    
    for train_start in range(start_year, end_year - train_years - test_years + 2, test_years):
        train_end_yr = train_start + train_years - 1
        test_start_yr = train_end_yr + 1
        test_end_yr = test_start_yr + test_years - 1
        
        train_mask = (panel.index.year >= train_start) & (panel.index.year <= train_end_yr)
        test_mask = (panel.index.year >= test_start_yr) & (panel.index.year <= test_end_yr)
        
        if test_mask.sum() == 0:
            continue
            
        train_panel = panel[train_mask]
        test_panel = panel[test_mask]
        
        if train_panel.empty or test_panel.empty:
            continue
            
        test_sig = pipeline_func(train_panel, test_panel)
        
        results.append({
            "train_years": f"{train_start}-{train_end_yr}",
            "test_years": f"{test_start_yr}-{test_end_yr}",
            "signal": test_sig
        })
        
    return results
