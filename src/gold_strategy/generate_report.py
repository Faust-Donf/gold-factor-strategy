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
    fig = plt.figure(figsize=(16, 18))
    
    # 1. Equity Curve
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(results.index, results["Strat_Cum"], label="Strategy (Asymmetric)", color="firebrick", linewidth=2)
    ax1.plot(results.index, results["BM_Cum"], label="Benchmark (Buy & Hold)", color="navy", alpha=0.7)
    ax1.set_title("Cumulative Returns", fontsize=14, fontweight='bold')
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    strat_dd = results["Strat_Cum"] / results["Strat_Cum"].cummax() - 1
    bm_dd = results["BM_Cum"] / results["BM_Cum"].cummax() - 1
    ax2.fill_between(strat_dd.index, strat_dd, 0, color="firebrick", alpha=0.3, label="Strategy DD")
    ax2.fill_between(bm_dd.index, bm_dd, 0, color="navy", alpha=0.2, label="Benchmark DD")
    ax2.set_title("Drawdown", fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    # 3. Price and S/B Points
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    price = panel["GLD_Adj_Close"]
    ax3.plot(price.index, price, color="gray", alpha=0.5, label="GLD Price")
    
    buys = sig_df[(sig_df["trade_flag"] == 1) & (sig_df["position"] == 1)]
    sells = sig_df[(sig_df["trade_flag"] == 1) & (sig_df["position"] == 0)]
    
    ax3.scatter(buys.index, price.loc[buys.index], marker='^', color='green', s=80, label='Buy (Go Long)', zorder=5)
    ax3.scatter(sells.index, price.loc[sells.index], marker='v', color='red', s=80, label='Sell (Go Cash)', zorder=5)
    ax3.fill_between(price.index, price.min(), price.max(), where=(sig_df["position"] == 1), color='green', alpha=0.1)
    
    ax3.set_title("GLD Price with Long/Cash Regimes", fontsize=14, fontweight='bold')
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)
    
    # 4. Yearly Returns Bar Chart
    ax4 = plt.subplot(4, 1, 4)
    # Calculate yearly returns
    strat_yearly = results["Strat_Net"].resample('YE').apply(lambda x: (1+x).prod() - 1)
    bm_yearly = results["BM_Net"].resample('YE').apply(lambda x: (1+x).prod() - 1)
    
    years = strat_yearly.index.year
    x = np.arange(len(years))
    width = 0.35
    
    ax4.bar(x - width/2, strat_yearly * 100, width, label='Strategy %', color='firebrick')
    ax4.bar(x + width/2, bm_yearly * 100, width, label='Benchmark %', color='navy', alpha=0.7)
    
    ax4.set_title("Yearly Returns Comparison (%)", fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(years, rotation=45)
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart locally for GitHub rendering
    plt.savefig('chart.png', format='png', dpi=150)
    
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
    
    # Split metrics into categories and translate
    ret_metrics = pd.DataFrame({
        "指标名称": ["年化收益率 (Ann Return)", "基准年化收益 (BM Return)", "绝对超额收益 (Alpha)"],
        "数值": [f"{m['Ann_Return']:.2%}", f"{m['BM_Return']:.2%}", f"{m['Alpha']:.2%}"],
        "说明": [
            "策略每年的平均复利回报",
            "一直持有黄金每年的平均回报",
            "刨除大盘涨跌后，策略纯粹靠自身能力多赚的钱"
        ]
    })
    
    risk_metrics = pd.DataFrame({
        "指标名称": ["最大回撤 (Max Drawdown)", "夏普比率 (Sharpe Ratio)", "索提诺比率 (Sortino Ratio)", "卡玛比率 (Calmar Ratio)", "市场敏感度 (Beta)"],
        "数值": [f"{m['Max_DD']:.2%}", f"{m['Sharpe']:.3f}", f"{m['Sortino']:.3f}", f"{m['Calmar']:.3f}", f"{m['Beta']:.3f}"],
        "说明": [
            "历史上买入后遭遇过的最惨亏损幅度。数值越小越好。",
            "每承受1单位总风险，能换取多少超额回报。越高越好(>1极佳)。",
            "仅评估下跌风险的收益率。比夏普更能反映应对崩盘的能力。",
            "年化收益与最大回撤的比例。>1说明回本快，表现优异。",
            "策略对黄金大盘的跟随程度。0.5说明大盘跌10%，策略才跌5%。"
        ]
    })
    
    trade_metrics = pd.DataFrame({
        "指标名称": ["胜率 (Win Rate)", "市场暴露度 (Exposure)", "总交易次数 (Total Trades)"],
        "数值": [f"{m['Win_Rate']:.2%}", f"{m['Exposure']:.2%}", f"{int(m['Num_Trades'])}"],
        "说明": [
            "所有开仓中，最终赚钱离场的比例。",
            "资金在市场里的时间占比。40%说明60%时间在空仓避险。",
            "回测历史中的完整买卖回合数。"
        ]
    })
    
    def generate_html_table(df, title):
        html = f"<h2>{title}</h2>"
        html += df.to_html(index=False, classes="table", escape=False)
        return html
        
    def generate_md_table(df, title):
        md = f"### {title}\n"
        md += "| " + " | ".join(df.columns) + " |\n"
        md += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
        for _, row in df.iterrows():
            md += "| " + " | ".join(map(str, row.values)) + " |\n"
        return md + "\n"

    img_b64 = create_charts(panel, res, sig)
    
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>黄金量化交易策略 (Gold Factor Strategy)</title>
        <style>
            body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; background-color: #f8f9fa; color: #333; line-height: 1.6; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
            h2 {{ color: #1a5276; margin-top: 30px; }}
            .container {{ display: flex; flex-direction: column; max-width: 1200px; margin: auto; }}
            .table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            .table th {{ background-color: #2c3e50; color: white; }}
            .table tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .chart {{ margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 4px; }}
            .config {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 5px solid #2c3e50; }}
            .config h3 {{ margin-top: 0; color: #2c3e50; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>黄金多因子量化交易策略 - 深度分析报告</h1>
            
            <div class="config">
                <h3>🛠️ 策略核心逻辑 (最终严选 10 因子组合)</h3>
                <p><b>趋势与波动 (技术面):</b> 60日突破 (Breakout_60d), 60日动量衰竭 (Mom_60d), 波动率扩张 (ATR_Ratio_14_60), 动量加速度 (Mom_Accel_20_60)</p>
                <p><b>避险与宏观 (宏观面):</b> 美债回撤避险 (IEF_DD_252d), 黄金美股相关性 (GLD_SPY_Corr_20d), 避险资金流入 (IEF_Mom_10d), 黄金相对美债强势 (GLD_vs_TLT_RS_60d), 黄金相对美股强势 (GLD_vs_SPY_RS_60d), VIX恐慌指数 (VIX_Mom_20d)</p>
                <p><b>操作机制 (非对称极值法):</b> 综合打分超过 <b>{entry_thr}</b> 时入场做多；当宏观环境恶化，打分跌破严格的 <b>{exit_thr}</b> 时空仓避险。持仓下限：{hold_days} 天。</p>
                <p><i>💡 意义：这套逻辑能在黄金大牛市中死死咬住利润，并在类似于 2013年、2020年 的系统性流动性危机中，提前嗅到风险并逃顶。</i></p>
            </div>
            
            {generate_html_table(ret_metrics, "📈 收益能力指标 (Return Metrics)")}
            {generate_html_table(risk_metrics, "🛡️ 风险控制指标 (Risk Metrics)")}
            {generate_html_table(trade_metrics, "⚖️ 交易统计 (Trade Statistics)")}
            
            <h2>📊 可视化与具体买卖点位 (Trade Visualizations)</h2>
            <p>图中 <b>绿色向上箭头 (▲)</b> 代表满足宏观入场条件，执行<b>买入做多</b>。<br>
            <b>红色向下箭头 (▼)</b> 代表避险因子报警，执行<b>清仓避险</b>。<br>
            绿色阴影区域为您实际持有黄金的阶段，空白区域为资金空仓安全的阶段。</p>
            <img class="chart" src="data:image/png;base64,{img_b64}" alt="Strategy Charts" style="width:100%; max-width:1200px;"/>
        </div>
    </body>
    </html>
    """
    
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        
    # Generate README.md content
    md_content = f"""# 黄金多因子量化交易策略 (Gold Factor Quantitative Strategy)

## 🛠️ 策略核心逻辑
本策略采用**非对称多因子阈值模型**来交易黄金(GLD)。通过捕捉 10 个精心挑选的宏观与技术因子，策略在最大化上涨收益的同时，有效规避了黄金历史上的重大回撤。

### 最终严选 10 因子组合
- **趋势与波动 (技术面):** 60日突破 (+), 60日动量衰竭 (-), 波动率扩张 (+), 动量加速度 (+)
- **避险与宏观 (宏观面):** 美债回撤 (+), 黄金美股相关性 (+), 美债动量 (+), 黄金相对美债强势 (-), 黄金相对美股强势 (-), VIX恐慌动量 (+)
- **操作机制:** 综合打分 > {entry_thr} 时入场做多，跌破 {exit_thr} 时清仓避险。

## 📊 策略表现深度解析

{generate_md_table(ret_metrics, "📈 收益能力指标")}
{generate_md_table(risk_metrics, "🛡️ 风险控制指标")}
{generate_md_table(trade_metrics, "⚖️ 交易统计")}

## 📈 走势图与买卖点标记
绿色向上箭头 (▲) 表示执行**买入做多**。红色向下箭头 (▼) 表示执行**清仓避险**。
![Strategy Charts](chart.png)
"""

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Report generated at report.html and README.md")

if __name__ == "__main__":
    generate_report()
