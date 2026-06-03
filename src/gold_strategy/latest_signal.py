import os
import pandas as pd
from gold_strategy.config import StrategyConfig
from gold_strategy.data import build_panel
from gold_strategy.features import create_technical_features, create_macro_features
from gold_strategy.strategy import generate_signals
from gold_strategy.reporting import export_latest_signal

def main():
    config = StrategyConfig.get_default()
    panel = build_panel(config)
    if panel.empty:
        print("Data download failed or panel is empty. Exiting.")
        return
    
    tech_features = create_technical_features(panel, config.target_symbol)
    macro_features = create_macro_features(panel)
    features = tech_features.join(macro_features)
    
    selected = {"Mom_10d": 1, "Mom_60d": 1, "DGS10_Chg_20d": -1}
    signal_df = generate_signals(features, selected, min_holding_days=5)
    
    os.makedirs("reports/signals", exist_ok=True)
    export_latest_signal(signal_df, "reports/signals/latest_signal.csv")
    print("Latest signal exported to reports/signals/latest_signal.csv")

if __name__ == "__main__":
    main()
