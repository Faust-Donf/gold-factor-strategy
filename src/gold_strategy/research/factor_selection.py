import pandas as pd

def select_factors(ic_stats: pd.DataFrame, corr_clusters: dict, min_icir: float = 0.5) -> pd.DataFrame:
    """Select factors based on ICIR and correlation clusters."""
    # This is a simplified selector
    # Usually we'd look at ICIR > threshold and pick the best in each cluster
    
    selections = []
    
    # group by factor to get average ICIR across targets, or just pick a target
    # For simplicity, we just use the first target or average
    avg_icir = ic_stats.groupby("Factor")["ICIR"].mean()
    
    for cluster_id, factors in corr_clusters.items():
        # find factor with highest avg_icir in cluster
        best_factor = None
        best_val = -float('inf')
        
        for f in factors:
            if f in avg_icir and abs(avg_icir[f]) > best_val:
                best_val = abs(avg_icir[f])
                best_factor = f
                
        for f in factors:
            if f == best_factor and best_val >= min_icir:
                selections.append({"Factor": f, "Selected": True, "Reason": f"Best in cluster {cluster_id} with ICIR {best_val:.2f}"})
            elif f == best_factor:
                selections.append({"Factor": f, "Selected": False, "Reason": f"Best in cluster {cluster_id} but ICIR {best_val:.2f} < {min_icir}"})
            else:
                selections.append({"Factor": f, "Selected": False, "Reason": f"Correlated with {best_factor} (cluster {cluster_id})"})
                
    return pd.DataFrame(selections)
