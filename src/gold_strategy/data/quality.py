import pandas as pd

def check_data_quality(panel: pd.DataFrame, target_symbol: str = "GLD") -> dict:
    """Check panel quality constraints."""
    issues = []
    
    # Check monotonic index
    if not panel.index.is_monotonic_increasing:
        issues.append("Index is not monotonic increasing.")
        
    # Check duplicate dates
    if panel.index.duplicated().any():
        issues.append("Index contains duplicate dates.")
        
    # Check target price missing
    target_col = f"{target_symbol}_Adj_Close"
    if target_col not in panel.columns:
        issues.append(f"Target column {target_col} is missing.")
    elif panel[target_col].isna().sum() > 0:
        issues.append(f"Target column {target_col} has missing values.")
        
    # Check for excessive missing macro data at the end (not forward filled well)
    # This might happen if macro data is delayed. We just warn.
    
    is_valid = len(issues) == 0
    return {
        "is_valid": is_valid,
        "issues": issues,
        "missing_counts": panel.isna().sum().to_dict()
    }

def clean_panel(panel: pd.DataFrame, target_symbol: str = "GLD") -> pd.DataFrame:
    """Clean the panel according to rules."""
    # Ensure index uniqueness and monotonic
    panel = panel[~panel.index.duplicated(keep='first')]
    panel = panel.sort_index()
    
    # Drop rows where target is missing
    target_col = f"{target_symbol}_Adj_Close"
    if target_col in panel.columns:
        panel = panel.dropna(subset=[target_col])
        
    return panel
