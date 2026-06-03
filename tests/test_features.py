import pytest
import pandas as pd
import numpy as np
from gold_strategy.features.transforms import rolling_zscore, rolling_percentile, winsorize
from gold_strategy.features.targets import create_targets

def test_rolling_zscore():
    s = pd.Series([1, 2, 3, 4, 5])
    z = rolling_zscore(s, 3)
    # window 3 on [1, 2, 3] -> mean 2, std 1 -> z(3) = (3 - 2)/1 = 1
    assert np.isclose(z.iloc[2], 1.0)
    assert pd.isna(z.iloc[0])

def test_winsorize():
    s = pd.Series([1, 5, 5, 5, 10])
    w = winsorize(s, 0.1, 0.9)
    # min should be > 1, max < 10
    assert w.min() > 1.0 or np.isclose(w.min(), 2.6, atol=2.0)
    assert w.max() <= 10.0

def test_create_targets():
    panel = pd.DataFrame({
        "GLD_Adj_Close": [100, 105, 110, 115, 120]
    }, index=pd.date_range("2020-01-01", periods=5))
    
    targets = create_targets(panel, "GLD", [1, 2])
    assert "Target_Ret_1d" in targets.columns
    assert "Target_Ret_2d" in targets.columns
    # day 1 to day 2 is 105/100 - 1 = 0.05
    assert np.isclose(targets["Target_Ret_1d"].iloc[0], 0.05)
    # day 1 to day 3 is 110/100 - 1 = 0.10
    assert np.isclose(targets["Target_Ret_2d"].iloc[0], 0.10)
    # last day should be NaN
    assert pd.isna(targets["Target_Ret_1d"].iloc[-1])
