import pandas as pd
import scipy.cluster.hierarchy as sch

def compute_correlation(features: pd.DataFrame, method: str = 'spearman') -> pd.DataFrame:
    """Compute correlation matrix."""
    return features.corr(method=method)

def get_correlation_clusters(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> dict:
    """Group highly correlated factors."""
    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        return {}
        
    # distance matrix
    dist = 1 - corr_matrix.abs()
    dist = dist.clip(0, 1) # handle floating point
    
    # linkage
    linkage = sch.linkage(sch.distance.squareform(dist), method='complete')
    
    # clusters
    labels = sch.fcluster(linkage, 1 - threshold, criterion='distance')
    
    clusters = {}
    for factor, label in zip(corr_matrix.columns, labels):
        clusters.setdefault(label, []).append(factor)
        
    return clusters
