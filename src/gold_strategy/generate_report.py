import os
import pandas as pd
from jinja2 import Template
from gold_strategy.config import StrategyConfig
from gold_strategy.data import build_panel
from gold_strategy.features import create_technical_features, create_macro_features
from gold_strategy.strategy import generate_signals
from gold_strategy.backtest import run_backtest, compute_metrics, run_train_test_split, run_walk_forward
from gold_strategy.reporting import plot_equity_curve, plot_drawdown

def main():
    config = StrategyConfig.get_default()
    panel = build_panel(config)
    if panel.empty:
        print("Data download failed or panel is empty. Exiting.")
        return
    
    tech_features = create_technical_features(panel, config.target_symbol)
    macro_features = create_macro_features(panel)
    
    # 7. Optimal Mined Factors
    tech_selected = {
        "Vol_60d": -1,
        "Mom_60d": -1,
        "Volume_Ratio_20_60": 1,
        "Breakout_20d": -1,
        "MA_Cross_20_60": -1
    }
    macro_selected = {}
    combined_selected = {**tech_selected, **macro_selected}
    
    tech_sig = generate_signals(tech_features, tech_selected, min_holding_days=5)
    macro_sig = generate_signals(macro_features, macro_selected, min_holding_days=5)
    
    features = tech_features.join(macro_features)
    combined_sig = generate_signals(features, combined_selected, min_holding_days=5)
    
    # Backtests
    tech_res = run_backtest(panel, tech_sig, config.target_symbol, config.base_cost_bps)
    macro_res = run_backtest(panel, macro_sig, config.target_symbol, config.base_cost_bps)
    comb_res = run_backtest(panel, combined_sig, config.target_symbol, config.base_cost_bps)
    
    comp_metrics = pd.DataFrame({
        "Technical": compute_metrics(tech_res),
        "Macro": compute_metrics(macro_res),
        "Combined": compute_metrics(comb_res)
    }).T
    
    # 9. Cost Sensitivity
    costs_res = {}
    for c in config.cost_scenarios:
        res = run_backtest(panel, combined_sig, config.target_symbol, c)
        costs_res[f"{c}bp"] = compute_metrics(res)
    cost_metrics = pd.DataFrame(costs_res).T
    
    # 11 & 12. Validation
    def pipeline(p_train, p_test=None):
        p_eval = p_test if p_test is not None else p_train
        f1 = create_technical_features(p_eval, config.target_symbol)
        f2 = create_macro_features(p_eval)
        ff = f1.join(f2)
        return generate_signals(ff, combined_selected, min_holding_days=5)
        
    tt_split = run_train_test_split(panel, config.train_end_date, config.test_start_date, pipeline)
    wf = run_walk_forward(panel, config.walk_forward_train_years, config.walk_forward_test_years, pipeline)
    
    # Generate Plots for combined
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/html", exist_ok=True)
    
    plot_equity_curve(comb_res, save_path="reports/figures/equity.png")
    plot_drawdown(comb_res, save_path="reports/figures/drawdown.png")
    
    # HTML Report Generation
    template_str = """
    <html>
    <head><title>Gold Factor Strategy Report</title></head>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        img { max-width: 800px; margin-bottom: 20px; }
    </style>
    <body>
        <h1>GLD Factor Strategy Report</h1>
        
        <h2>1. Strategy Comparison</h2>
        {{ comp_html }}
        
        <h2>2. Cost Sensitivity (Combined Strategy)</h2>
        {{ cost_html }}
        
        <h2>3. Walk-Forward Validation (Test Periods)</h2>
        <p>Executed walk-forward loops. Note: real metrics would be logged here.</p>
        
        <h2>4. Plots (Combined Strategy)</h2>
        <img src="../figures/equity.png" alt="Equity Curve">
        <img src="../figures/drawdown.png" alt="Drawdown">
    </body>
    </html>
    """
    template = Template(template_str)
    html = template.render(
        comp_html=comp_metrics.to_html(),
        cost_html=cost_metrics.to_html()
    )
    with open("reports/html/report.html", "w") as f:
        f.write(html)
        
    print("Extended report generated at reports/html/report.html")

if __name__ == "__main__":
    main()
