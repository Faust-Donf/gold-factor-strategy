import pandas as pd
import numpy as np

def compute_quantile_returns(features: pd.DataFrame, targets: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """Compute mean forward returns by factor quantile."""
    results = []
    
    for target_col in targets.columns:
        for factor in features.columns:
            df = pd.DataFrame({
                "factor": features[factor],
                "target": targets[target_col]
            }).dropna()
            
            if df.empty:
                continue
                
            try:
                # use rank and qcut to handle duplicates
                ranks = df["factor"].rank(method="first")
                df["quantile"] = pd.qcut(ranks, quantiles, labels=False) + 1
            except ValueError:
                continue
                
            q_rets = df.groupby("quantile")["target"].mean()
            
            # monotonicity
            q_diffs = q_rets.diff().dropna()
            monotonicity = (q_diffs > 0).mean() if not q_diffs.empty else 0
            
            # top vs full
            top_q_ret = q_rets.iloc[-1] if len(q_rets) > 0 else np.nan
            full_ret = df["target"].mean()
            top_excess = top_q_ret - full_ret
            
            res_dict = {
                "Target": target_col,
                "Factor": factor,
                "Monotonicity": monotonicity,
                "Top_vs_Full": top_excess
            }
            for q, val in q_rets.items():
                res_dict[f"Q{q}_Ret"] = val
                
            results.append(res_dict)
            
    return pd.DataFrame(results)
