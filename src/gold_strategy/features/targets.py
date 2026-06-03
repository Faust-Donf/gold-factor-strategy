import pandas as pd
from typing import List

def create_targets(panel: pd.DataFrame, target_symbol: str, horizons: List[int]) -> pd.DataFrame:
    """Create forward returns (labels). Must NOT be used as features."""
    targets = pd.DataFrame(index=panel.index)
    col = f"{target_symbol}_Adj_Close"
    
    if col not in panel.columns:
        return targets
        
    for h in horizons:
        # Shift back: price at t+h / price at t - 1
        targets[f"Target_Ret_{h}d"] = panel[col].shift(-h) / panel[col] - 1
        
    return targets
