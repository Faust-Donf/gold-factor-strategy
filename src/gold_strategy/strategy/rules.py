import pandas as pd
import numpy as np

def apply_holding_period(raw_signal: pd.Series, min_holding_days: int) -> pd.DataFrame:
    """Apply minimum holding period constraints."""
    
    result = pd.DataFrame(index=raw_signal.index)
    result["raw_position"] = raw_signal.clip(0, 1).round() # 0 or 1
    
    final_pos = np.zeros(len(result))
    days_held = np.zeros(len(result))
    trade_flag = np.zeros(len(result))
    
    current_pos = 0
    days_since_trade = 0
    
    raw_arr = result["raw_position"].fillna(0).values
    
    for i in range(len(result)):
        desired_pos = raw_arr[i]
        
        if desired_pos != current_pos:
            if days_since_trade >= min_holding_days or days_since_trade == 0:
                # trade allowed
                current_pos = desired_pos
                days_since_trade = 1
                trade_flag[i] = 1
            else:
                # trade blocked
                days_since_trade += 1
        else:
            if current_pos > 0:
                days_since_trade += 1
            else:
                days_since_trade = 0 # reset if flat? Or just keep counting since last flat trade.
                # Usually days_held only counts when in position
                
        final_pos[i] = current_pos
        days_held[i] = days_since_trade
        
    result["position"] = final_pos
    result["days_held"] = days_held
    result["trade_flag"] = trade_flag
    
    return result
