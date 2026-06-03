import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_equity_curve(results: pd.DataFrame, title: str = "Equity Curve", save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["Strat_Cum"], label="Strategy", color="blue")
    ax.plot(results["BM_Cum"], label="Benchmark", color="gray", linestyle="--")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        return fig

def plot_drawdown(results: pd.DataFrame, title: str = "Drawdown", save_path: str = None):
    from gold_strategy.backtest.metrics import compute_drawdown
    strat_dd = compute_drawdown(results["Strat_Cum"])
    bm_dd = compute_drawdown(results["BM_Cum"])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(strat_dd.index, strat_dd, 0, alpha=0.5, color="red", label="Strategy DD")
    ax.plot(bm_dd, color="gray", linestyle="--", label="Benchmark DD")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        return fig
