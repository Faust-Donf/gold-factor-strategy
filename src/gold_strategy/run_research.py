import argparse
from gold_strategy.config import StrategyConfig
from gold_strategy.data import build_panel
from gold_strategy.features import create_technical_features, create_macro_features, create_targets
from gold_strategy.research import compute_ic_summary, compute_timeseries_ic_stats, compute_correlation, get_correlation_clusters, select_factors

def main():
    config = StrategyConfig.get_default()
    panel = build_panel(config)
    if panel.empty:
        print("Data download failed or panel is empty. Exiting.")
        return
    
    tech_features = create_technical_features(panel, config.target_symbol)
    macro_features = create_macro_features(panel)
    features = tech_features.join(macro_features)
    
    targets = create_targets(panel, config.target_symbol, config.horizons)
    
    # Restrict to train period
    train_mask = panel.index <= pd.Timestamp(config.train_end_date)
    
    ic_stats = compute_timeseries_ic_stats(features[train_mask], targets[train_mask])
    corr = compute_correlation(features[train_mask])
    clusters = get_correlation_clusters(corr)
    
    selections = select_factors(ic_stats, clusters)
    print("Factor Selection Complete.")
    print(selections[selections["Selected"] == True])
    
if __name__ == "__main__":
    import pandas as pd
    main()
