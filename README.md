# Gold Factor Quantitative Strategy

## Strategy Overview
This strategy employs a multi-factor asymmetric threshold model to trade Gold (GLD). 
By capturing non-linear relationships between 10 carefully selected macro and technical factors, the model maximizes upside exposure while effectively dodging major macro drawdowns.

### Final Factor Composition (10 Factors)
- **Technical:** Breakout_60d (+), Mom_60d (-), ATR_Ratio_14_60 (+), Mom_Accel_20_60 (+)
- **Macro / Relative:** IEF_DD_252d (+), GLD_SPY_Corr_20d (+), IEF_Mom_10d (+), GLD_vs_TLT_RS_60d (-), GLD_vs_SPY_RS_60d (-), VIX_Mom_20d (+)
- **Logic:** Asymmetric Thresholds. Enter Long when score > -0.15, Exit to Cash when score < -0.45. Hold period: 5 days.

## Performance Metrics (vs Buy & Hold GLD)
| Metric | Value |
|--------|-------|
| Annualized Return | 16.79% |
| Benchmark Return | 10.92% |
| Sharpe Ratio | 1.060 |
| Sortino Ratio | 1.284 |
| Calmar Ratio | 0.474 |
| Alpha | 8.44% |
| Beta | 0.765 |
| Max Drawdown | -35.42% |
| Market Exposure | 79.25% |
| Win Rate | 53.79% |
| Total Trades | 73 |

## Visualizations & S/B Trade Points
Green Triangles (▲) indicate Buy signals. Red Triangles (▼) indicate Sell (Go Cash) signals.
![Strategy Charts](chart.png)
