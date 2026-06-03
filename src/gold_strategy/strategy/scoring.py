import pandas as pd
import numpy as np

def standardize_factors(features: pd.DataFrame, selected_factors: dict, window: int = 252) -> pd.DataFrame:
    """Standardize selected factors using rolling z-score and align direction."""
    standardized = pd.DataFrame(index=features.index)
    
    for factor, direction in selected_factors.items():
        if factor not in features.columns:
            continue
            
        roll = features[factor].rolling(window)
        zscore = (features[factor] - roll.mean()) / roll.std()
        
        # Apply direction: if direction is -1, invert zscore so higher is bullish
        standardized[factor] = zscore * direction
        
    return standardized

def compute_equal_weight_score(std_factors: pd.DataFrame) -> pd.Series:
    """Compute equal-weight score across factors."""
    return std_factors.mean(axis=1)

def compute_icir_weight_score(std_factors: pd.DataFrame, icir_weights: dict) -> pd.Series:
    """Compute ICIR weighted score."""
    weighted = pd.DataFrame(index=std_factors.index)
    total_weight = sum(abs(w) for w in icir_weights.values())
    
    if total_weight == 0:
        return std_factors.mean(axis=1)
        
    for factor, weight in icir_weights.items():
        if factor in std_factors.columns:
            weighted[factor] = std_factors[factor] * (abs(weight) / total_weight)
            
    return weighted.sum(axis=1)

def compute_voting_signal(std_factors: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """Equal weight vote: > threshold -> +1, < -threshold -> -1, else 0. Then average votes."""
    votes = pd.DataFrame(index=std_factors.index)
    for col in std_factors.columns:
        votes[col] = 0
        votes.loc[std_factors[col] > threshold, col] = 1
        votes.loc[std_factors[col] < -threshold, col] = -1
        
    return votes.mean(axis=1)
