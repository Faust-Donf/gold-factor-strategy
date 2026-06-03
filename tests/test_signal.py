import pytest
import pandas as pd
import numpy as np
from gold_strategy.strategy.rules import apply_holding_period

def test_apply_holding_period():
    raw_signal = pd.Series([0, 1, 1, 0, 1, 1, 0])
    
    # min holding = 2
    res = apply_holding_period(raw_signal, 2)
    pos = res["position"].values
    
    # t=0: pos=0
    # t=1: raw=1 -> pos=1 (trade_flag=1, days=1)
    # t=2: raw=1 -> pos=1 (days=2)
    # t=3: raw=0 -> pos=0 (trade_flag=1, days=1) (since days=2 >= min_holding)
    # t=4: raw=1 -> pos=1 (days=1)
    # t=5: raw=1 -> pos=1 (days=2)
    # t=6: raw=0 -> pos=0 (days=1)
    
    assert pos[1] == 1
    assert pos[2] == 1
    assert pos[3] == 0
    assert pos[4] == 0  # blocked by cash hold period
    assert pos[5] == 1
    assert pos[6] == 1  # blocked by long hold period

def test_apply_holding_period_blocked():
    raw_signal = pd.Series([0, 1, 0, 1, 1, 0])
    
    # min holding = 3
    res = apply_holding_period(raw_signal, 3)
    pos = res["position"].values
    
    # t=0: pos=0
    # t=1: raw=1 -> trade allowed, pos=1 (days=1)
    # t=2: raw=0 -> trade blocked! pos=1 (days=2)
    # t=3: raw=1 -> pos=1 (days=3)
    # t=4: raw=1 -> pos=1 (days=4)
    # t=5: raw=0 -> trade allowed, pos=0 (days=1)
    
    assert pos[1] == 1
    assert pos[2] == 1  # Blocked from going to 0
    assert pos[3] == 1
    assert pos[4] == 1
    assert pos[5] == 0  # Reached 3 days, can trade now
