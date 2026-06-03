import pytest
import pandas as pd
import numpy as np
from gold_strategy.backtest.metrics import compute_metrics, compute_drawdown

def test_compute_drawdown():
    cum_ret = pd.Series([1.0, 1.1, 1.05, 1.2, 1.1])
    dd = compute_drawdown(cum_ret)
    
    assert np.isclose(dd.iloc[0], 0)
    assert np.isclose(dd.iloc[1], 0)
    assert np.isclose(dd.iloc[2], 1.05 / 1.1 - 1)
    assert np.isclose(dd.iloc[3], 0)
    assert np.isclose(dd.iloc[4], 1.1 / 1.2 - 1)

def test_compute_metrics():
    # Construct a dummy results df
    results = pd.DataFrame({
        "Strat_Net": [0.01, 0.01, -0.01, 0.02, 0.01],
        "BM_Net": [0.005, 0.005, 0.005, 0.005, 0.005],
        "Position": [1, 1, 1, 1, 1]
    }, index=pd.date_range("2020-01-01", periods=5))
    
    m = compute_metrics(results)
    assert "Ann_Return" in m
    assert "Sharpe" in m
    assert "Max_DD" in m
    assert "Win_Rate" in m
    
    assert m["Exposure"] == 1.0
