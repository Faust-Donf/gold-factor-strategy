import pandas as pd
import numpy as np
from .scoring import standardize_factors, compute_equal_weight_score
from .rules import apply_holding_period

def generate_signals(features: pd.DataFrame, selected_factors: dict, min_holding_days: int = 5, threshold: float = -0.25, zscore_window: int = 252) -> pd.DataFrame:
    """Original linear signal generation."""
    std_factors = standardize_factors(features, selected_factors, window=zscore_window)
    score = compute_equal_weight_score(std_factors)
    raw_signal = (score > threshold).astype(int)
    signal_df = apply_holding_period(raw_signal, min_holding_days)
    signal_df["score"] = score
    
    reasons = []
    for i in range(len(signal_df)):
        if signal_df["trade_flag"].iloc[i] == 1:
            pos = signal_df["position"].iloc[i]
            reasons.append(f"Entered {'Long' if pos==1 else 'Cash'} due to score={score.iloc[i]:.2f}")
        else:
            reasons.append("")
    signal_df["reason"] = reasons
    return signal_df

def generate_asymmetric_signals(features: pd.DataFrame, factors: dict, entry_thr: float, exit_thr: float, hold_days: int, zscore_window: int = 252) -> pd.DataFrame:
    """
    Asymmetric signal generation.
    - If currently cash and score > entry_thr -> go long
    - If currently long and score < exit_thr -> go cash
    """
    std = standardize_factors(features, factors, window=zscore_window)
    score = std.mean(axis=1).fillna(0)
    
    n = len(score)
    position = np.zeros(n)
    trade_flag = np.zeros(n)
    current_pos = 0
    days_in = 0
    
    for i in range(n):
        s = score.iloc[i]
        if current_pos == 0:
            if s > entry_thr:
                current_pos = 1
                days_in = 1
                trade_flag[i] = 1
            else:
                current_pos = 0
                days_in = 0
                trade_flag[i] = 0
        else:
            days_in += 1
            if s < exit_thr and days_in >= hold_days:
                current_pos = 0
                days_in = 0
                trade_flag[i] = 1
            else:
                current_pos = 1
                trade_flag[i] = 0
        position[i] = current_pos
        
    signal_df = pd.DataFrame({
        'raw_position': position,
        'position': position,
        'days_held': 0, # Simplified
        'trade_flag': trade_flag,
        'score': score
    }, index=features.index[:n])
    
    reasons = []
    for i in range(len(signal_df)):
        if signal_df["trade_flag"].iloc[i] == 1:
            pos = signal_df["position"].iloc[i]
            reasons.append(f"Entered {'Long' if pos==1 else 'Cash'} due to score={score.iloc[i]:.2f}")
        else:
            reasons.append("")
    signal_df["reason"] = reasons
    
    return signal_df
