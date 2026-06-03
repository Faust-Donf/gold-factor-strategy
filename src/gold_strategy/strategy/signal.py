import pandas as pd
from .scoring import standardize_factors, compute_equal_weight_score
from .rules import apply_holding_period

def generate_signals(features: pd.DataFrame, selected_factors: dict, min_holding_days: int = 5) -> pd.DataFrame:
    """End-to-end signal generation from features."""
    
    # 1. Standardize and orient
    std_factors = standardize_factors(features, selected_factors)
    
    # 2. Score
    score = compute_equal_weight_score(std_factors)
    
    # 3. Raw Signal: score > threshold -> 1 (long), else 0 (cash)
    # We introduce a bullish bias threshold (-0.25) to stay invested longer and reduce whipsaws.
    raw_signal = (score > -0.25).astype(int)
    
    # 4. Apply rules
    signal_df = apply_holding_period(raw_signal, min_holding_days)
    signal_df["score"] = score
    
    # 5. Reason
    reasons = []
    for i in range(len(signal_df)):
        if signal_df["trade_flag"].iloc[i] == 1:
            pos = signal_df["position"].iloc[i]
            reasons.append(f"Entered {'Long' if pos==1 else 'Cash'} due to score={score.iloc[i]:.2f}")
        else:
            reasons.append("")
    signal_df["reason"] = reasons
    
    return signal_df
