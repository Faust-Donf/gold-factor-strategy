import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from gold_strategy.config import StrategyConfig
from gold_strategy.data import build_panel
from gold_strategy.features import create_technical_features, create_macro_features
from gold_strategy.strategy.signal import generate_asymmetric_signals
from gold_strategy.backtest import run_backtest, compute_metrics

def df_to_html(df: pd.DataFrame, title: str) -> str:
    html = f"<h2>{title}</h2>"
    html += df.to_html(classes="table", float_format=lambda x: f"{x:.4f}")
    return html

def create_charts(panel, results, sig_df):
    plt.style.use('bmh')
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Equity Curve
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(results.index, results["Strat_Cum"], label="Strategy (Asymmetric)", color="firebrick", linewidth=2)
    ax1.plot(results.index, results["BM_Cum"], label="Benchmark (Buy & Hold)", color="navy", alpha=0.7)
    ax1.set_title("Cumulative Returns", fontsize=14, fontweight='bold')
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    strat_dd = results["Strat_Cum"] / results["Strat_Cum"].cummax() - 1
    bm_dd = results["BM_Cum"] / results["BM_Cum"].cummax() - 1
    ax2.fill_between(strat_dd.index, strat_dd, 0, color="firebrick", alpha=0.3, label="Strategy DD")
    ax2.fill_between(bm_dd.index, bm_dd, 0, color="navy", alpha=0.2, label="Benchmark DD")
    ax2.set_title("Drawdown", fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    # 3. Price and S/B Points
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    price = panel["GLD_Adj_Close"]
    ax3.plot(price.index, price, color="gray", alpha=0.5, label="GLD Price")
    
    # Find Buy and Sell points
    # Buy: trade_flag == 1 and position == 1
    # Sell: trade_flag == 1 and position == 0
    buys = sig_df[(sig_df["trade_flag"] == 1) & (sig_df["position"] == 1)]
    sells = sig_df[(sig_df["trade_flag"] == 1) & (sig_df["position"] == 0)]
    
    ax3.scatter(buys.index, price.loc[buys.index], marker='^', color='green', s=80, label='Buy (Go Long)', zorder=5)
    ax3.scatter(sells.index, price.loc[sells.index], marker='v', color='red', s=80, label='Sell (Go Cash)', zorder=5)
    
    # Shade regions where we are long
    ax3.fill_between(price.index, price.min(), price.max(), where=(sig_df["position"] == 1), color='green', alpha=0.1)
    
    ax3.set_title("GLD Price with Long/Cash Regimes", fontsize=14, fontweight='bold')
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def generate_report():
    config = StrategyConfig.get_default()
    panel = build_panel(config)
    
    tech = create_technical_features(panel, config.target_symbol)
    macro = create_macro_features(panel)
    all_features = tech.join(macro).dropna(how='all')
    
    # The Best Asymmetric Configuration from mining
    best_factors = {
        'Breakout_60d': 1, 'IEF_DD_252d': 1, 'Mom_60d': -1, 
        'GLD_SPY_Corr_20d': 1, 'IEF_Mom_10d': 1, 'ATR_Ratio_14_60': 1, 
        'GLD_vs_TLT_RS_60d': -1, 'Mom_Accel_20_60': 1, 
        'GLD_vs_SPY_RS_60d': -1, 'VIX_Mom_20d': 1
    }
    
    entry_thr = -0.15
    exit_thr = -0.45
    hold_days = 5
    zscore_window = 252
    cost_bps = 5
    
    sig = generate_asymmetric_signals(all_features, best_factors, entry_thr, exit_thr, hold_days, zscore_window)
    res = run_backtest(panel, sig, config.target_symbol, cost_bps)
    m = compute_metrics(res)
    
    metrics_df = pd.DataFrame([m]).T
    metrics_df.columns = ["Value"]
    
    img_b64 = create_charts(panel, res, sig)
    
    html_content = f"""
    <html>
    <head>
        <title>Gold Factor Strategy Report (Asymmetric)</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
            .container {{ display: flex; flex-direction: column; max-width: 1200px; margin: auto; }}
            .table {{ border-collapse: collapse; width: 400px; margin-bottom: 30px; background: white; }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            .table th {{ background-color: #2c3e50; color: white; }}
            .table tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .chart {{ margin-top: 30px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .config {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 30px;}}
            .config h3 {{ margin-top: 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Gold Factor Quantitative Strategy - Final Report</h1>
            
            <div class="config">
                <h3>Final Factor Composition (10 Factors)</h3>
                <p><b>Technical:</b> Breakout_60d (+), Mom_60d (-), ATR_Ratio_14_60 (+), Mom_Accel_20_60 (+)</p>
                <p><b>Macro / Relative:</b> IEF_DD_252d (+), GLD_SPY_Corr_20d (+), IEF_Mom_10d (+), GLD_vs_TLT_RS_60d (-), GLD_vs_SPY_RS_60d (-), VIX_Mom_20d (+)</p>
                <p><b>Logic:</b> Asymmetric Thresholds. Enter Long when score > {entry_thr}, Exit to Cash when score < {exit_thr}. Hold period: {hold_days} days.</p>
                <p><i>This allows massive exposure during uptrends while retaining extreme crash avoidance capabilities.</i></p>
            </div>
            
            {df_to_html(metrics_df, "Performance Metrics (vs Buy & Hold GLD)")}
            
            <h2>Visualization & Trade Points</h2>
            <p>Green Triangles (▲) indicate Buy signals. Red Triangles (▼) indicate Sell (Go Cash) signals.</p>
            <img class="chart" src="data:image/png;base64,{img_b64}" alt="Strategy Charts" style="width:100%; max-width:1200px;"/>
        </div>
    </body>
    </html>
    """
    
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Report generated at report.html")
    print(f"Ann Return: {m['Ann_Return']:.4f}, Sharpe: {m['Sharpe']:.4f}, Max DD: {m['Max_DD']:.4f}")

if __name__ == "__main__":
    generate_report()
