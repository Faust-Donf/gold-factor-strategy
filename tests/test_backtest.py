import pytest
import pandas as pd
import numpy as np
from gold_strategy.backtest.engine import apply_costs, run_backtest

def test_apply_costs():
    pos = pd.Series([0, 1, 1, 0])
    ret = pd.Series([0.01, 0.02, -0.01, 0.05])
    # cost is applied on trade
    # t=1: pos diff 1 -> cost 5bp
    # t=2: pos diff 0 -> cost 0
    # t=3: pos diff 1 -> cost 5bp
    
    net = apply_costs(pos, ret, cost_bps=5)
    assert np.isclose(net.iloc[0], 0.01) # no trade
    assert np.isclose(net.iloc[1], 0.02 - 0.0005)
    assert np.isclose(net.iloc[2], -0.01)
    assert np.isclose(net.iloc[3], 0.05 - 0.0005)

def test_run_backtest():
    panel = pd.DataFrame({
        "GLD_Adj_Close": [100, 101, 102, 100, 105]
    }, index=pd.date_range("2020-01-01", periods=5))
    
    # daily_ret: NaN, 0.01, ~0.0099, -0.0196, 0.05
    signal = pd.DataFrame({
        "position": [0, 1, 1, 0, 0]
    }, index=panel.index)
    
    res = run_backtest(panel, signal, "GLD", cost_bps=0, lag=1)
    
    assert "Strat_Net" in res.columns
    # lag=1: signal at t=1 (pos=1) -> executed at end of t=1 -> returns at t=2
    # ret at t=2 is 102/101 - 1
    assert np.isclose(res["Strat_Gross"].iloc[2], 102/101 - 1)
