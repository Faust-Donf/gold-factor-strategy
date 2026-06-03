# GLD 黄金 ETF 因子挖掘与只做多择时回测设计规格

**日期**：2026-06-03  
**项目**：gold-factor-strategy  
**阶段**：Brainstorming Design Spec  
**状态**：等待用户审阅  

## 1. 背景与目标

本项目构建一个 **GLD 黄金 ETF 只做多择时策略原型**。目标不是单纯生成一条历史收益曲线，而是建立一条完整、可解释、可复现的量化研究链路：

```text
数据获取
→ 因子构造
→ 因子有效性检验
→ 因子筛选
→ 信号合成
→ 只做多回测
→ 样本外验证
→ 成本敏感性分析
→ HTML 报告与最新信号输出
```

第一版聚焦 GLD 日频策略，输出可阅读的研究 notebook、HTML 报告和最新持仓信号表。项目采用“因子研究驱动策略，规则固化落地”的路线：先通过 IC、分层收益、稳定性和相关性检验筛选有效因子，再把有效因子压缩为可解释的持有/空仓信号。

## 2. 范围

### 2.1 第一版包含

- 标的：GLD 黄金 ETF。
- 方向：只做多，不做空。
- 信号：持有 GLD 或空仓持现金。
- 频率：日频生成信号，加入最小持仓期与换手约束。
- 回测区间：GLD 上市以来。
- 数据源：Yahoo Finance + FRED。
- 因子路线：先构建技术因子 baseline，再加入宏观因子增强。
- 验证方式：固定训练/测试切分 + walk-forward 验证。
- 成本设定：主结果使用单边 5bp 成本，并做 0bp / 5bp / 10bp 敏感性测试。
- 交付形态：Notebook + HTML 报告 + 最新信号表。

### 2.2 第一版不包含

- 黄金期货连续合约换月。
- 杠杆策略。
- 多空策略。
- 高频交易。
- 自动实盘下单。
- 复杂机器学习黑箱模型。
- 多品种资产配置组合。
- Web UI 或数据库系统。

这些能力可以作为二期扩展，但不进入第一版实现范围。

## 3. 项目结构

建议项目结构如下：

```text
gold-factor-strategy/
  docs/
    superpowers/
      specs/
        2026-06-03-gold-factor-strategy-design.md

  notebooks/
    01_factor_research.ipynb
    02_strategy_backtest.ipynb

  src/
    gold_strategy/
      data/
        yahoo_loader.py
        fred_loader.py
        local_loader.py
        panel_builder.py

      features/
        technical.py
        macro.py
        transforms.py

      research/
        ic_analysis.py
        quantile_analysis.py
        factor_selection.py
        stability.py

      strategy/
        scoring.py
        signal.py
        rules.py

      backtest/
        engine.py
        metrics.py
        costs.py
        validation.py

      reporting/
        plots.py
        html_report.py
        signal_export.py

  reports/
    html/
    figures/
    signals/

  data/
    raw/
    processed/

  tests/
    test_features.py
    test_backtest.py
    test_metrics.py
    test_signal.py

  pyproject.toml
  README.md
```

设计原则是：核心逻辑放入 `src/gold_strategy/`，notebook 只负责研究展示和调用，避免形成不可维护的巨型 notebook。

## 4. 架构与模块边界

### 4.1 `data` 模块

职责：获取、缓存和对齐数据。

输入：symbol 列表、FRED series id、起止日期、缓存参数。  
输出：以 GLD 交易日为索引的日频数据面板。

主要文件：

- `yahoo_loader.py`：获取 GLD、SPY、VIX、UUP、TLT、IEF、USO 等市场数据。
- `fred_loader.py`：获取利率、实际利率、通胀预期等宏观数据。
- `local_loader.py`：预留本地 CSV / Wind / Bloomberg 替换入口。
- `panel_builder.py`：将不同来源数据统一为日频研究面板。

### 4.2 `features` 模块

职责：构造技术因子、宏观因子和通用变换。

主要文件：

- `technical.py`：动量、趋势、波动率、回撤、突破、成交量因子。
- `macro.py`：利率、实际利率、通胀预期、美元、风险偏好、大宗商品联动因子。
- `transforms.py`：rolling z-score、rolling rank、winsorize、滞后处理、方向调整。

### 4.3 `research` 模块

职责：检验因子有效性并筛选候选因子。

主要文件：

- `ic_analysis.py`：Pearson IC、Spearman IC、ICIR、滚动 IC。
- `quantile_analysis.py`：分位数组合、单调性、top-bottom spread。
- `factor_selection.py`：综合评分、相关性去冗余、因子筛选。
- `stability.py`：年度稳定性、滚动窗口稳定性、样本外稳定性。

### 4.4 `strategy` 模块

职责：将有效因子转成可执行信号。

主要文件：

- `scoring.py`：等权打分、等权投票、ICIR 加权。
- `signal.py`：raw signal 到 final signal 的转换。
- `rules.py`：最小持仓期、换手约束、阈值规则。

### 4.5 `backtest` 模块

职责：执行只做多回测并计算绩效指标。

主要文件：

- `engine.py`：回测主逻辑。
- `metrics.py`：年化收益、波动率、Sharpe、Sortino、Calmar、最大回撤、换手等。
- `costs.py`：0bp / 5bp / 10bp 成本模型。
- `validation.py`：训练/测试切分与 walk-forward 验证。

### 4.6 `reporting` 模块

职责：输出图表、HTML 报告和最新信号表。

主要文件：

- `plots.py`：净值曲线、回撤曲线、年度收益、IC 图、分层收益图。
- `html_report.py`：生成完整 HTML 报告。
- `signal_export.py`：导出最新信号 CSV / HTML。

## 5. 数据设计

### 5.1 数据源

第一版使用开源数据，后续保留替换为本地 CSV、Wind 或 Bloomberg 的能力。

| 类型 | 来源 | 示例 |
|---|---|---|
| GLD 价格 | Yahoo Finance | GLD OHLCV |
| 美股风险资产 | Yahoo Finance | SPY |
| 波动率 | Yahoo Finance | VIX |
| 美元代理 | Yahoo Finance | UUP 或 DXY 代理 |
| 美债利率 | FRED | 10Y / 2Y / 3M |
| 实际利率 | FRED | 10Y TIPS |
| 通胀预期 | FRED | 10Y breakeven inflation |
| 原油 | Yahoo Finance | WTI 或 USO |
| 债券 ETF | Yahoo Finance | TLT / IEF |

### 5.2 日频面板

全部数据统一到以 GLD 交易日为主索引的日频面板。市场类数据按交易日对齐，宏观类数据前向填充。所有因子必须只使用当日收盘及以前的信息。

面板字段示例：

```text
date
GLD_open
GLD_high
GLD_low
GLD_close
GLD_adj_close
GLD_volume
SPY_close
VIX_close
rate_10y
rate_2y
real_rate_10y
inflation_expectation_10y
usd_proxy
oil_proxy
```

### 5.3 前视偏差控制

- 信号在当日收盘后生成。
- 回测默认下一交易日成交。
- 策略收益使用 `position[t-1] * GLD_return[t]`。
- rolling 统计只使用历史窗口。
- 宏观数据第一版使用历史修正数据，但报告中必须披露该限制。

## 6. 因子体系

### 6.1 技术因子 baseline

技术因子只使用 GLD 自身价格和成交量，用于建立 baseline。

#### 动量因子

| 因子 | 含义 |
|---|---|
| `mom_5` | 过去 5 日收益 |
| `mom_10` | 过去 10 日收益 |
| `mom_20` | 过去 20 日收益 |
| `mom_60` | 过去 60 日收益 |
| `mom_120` | 过去 120 日收益 |
| `mom_252` | 过去 252 日收益 |

#### 趋势因子

| 因子 | 含义 |
|---|---|
| `ma_20_gap` | 收盘价相对 20 日均线偏离 |
| `ma_60_gap` | 收盘价相对 60 日均线偏离 |
| `ma_120_gap` | 收盘价相对 120 日均线偏离 |
| `ma_cross_20_60` | 20 日均线是否高于 60 日均线 |
| `ma_cross_60_120` | 60 日均线是否高于 120 日均线 |

#### 波动率因子

| 因子 | 含义 |
|---|---|
| `vol_20` | 过去 20 日年化波动率 |
| `vol_60` | 过去 60 日年化波动率 |
| `vol_ratio_20_60` | 短期波动率 / 中期波动率 |
| `realized_vol_z` | 波动率滚动 z-score |

#### 回撤与突破因子

| 因子 | 含义 |
|---|---|
| `drawdown_20` | 相对 20 日高点回撤 |
| `drawdown_60` | 相对 60 日高点回撤 |
| `breakout_20` | 是否突破 20 日高点 |
| `breakout_60` | 是否突破 60 日高点 |
| `distance_to_high_252` | 距离一年高点的位置 |

#### 成交量因子

| 因子 | 含义 |
|---|---|
| `volume_z_20` | 成交量 20 日 z-score |
| `volume_ratio_20_60` | 短期成交量 / 中期成交量 |

### 6.2 宏观因子增强

宏观因子用于检验黄金定价逻辑是否带来增量收益。

#### 利率类

| 因子 | 含义 |
|---|---|
| `rate_10y_level` | 10 年美债收益率水平 |
| `rate_10y_change_20` | 10 年利率 20 日变化 |
| `real_rate_10y_level` | 10 年实际利率 |
| `real_rate_10y_change_20` | 实际利率 20 日变化 |
| `yield_curve_10y_2y` | 10Y - 2Y 期限利差 |

#### 通胀预期类

| 因子 | 含义 |
|---|---|
| `breakeven_10y_level` | 10 年通胀预期 |
| `breakeven_10y_change_20` | 通胀预期 20 日变化 |
| `real_rate_minus_inflation` | 实际利率与通胀预期组合指标 |

#### 美元类

| 因子 | 含义 |
|---|---|
| `usd_mom_20` | 美元代理 20 日动量 |
| `usd_mom_60` | 美元代理 60 日动量 |
| `usd_z_252` | 美元相对一年分位水平 |

#### 风险偏好类

| 因子 | 含义 |
|---|---|
| `vix_level` | VIX 水平 |
| `vix_change_20` | VIX 20 日变化 |
| `spy_mom_20` | 美股 20 日动量 |
| `spy_drawdown_60` | 美股 60 日回撤 |

#### 大宗商品联动类

| 因子 | 含义 |
|---|---|
| `oil_mom_20` | 原油 20 日动量 |
| `commodity_proxy_mom` | 商品代理动量，后续可扩展 |

## 7. 因子研究方法

### 7.1 预测目标

同时评估三个 forward return horizon：

| 目标 | 含义 |
|---|---|
| `fwd_ret_5d` | 未来 5 日 GLD 收益 |
| `fwd_ret_10d` | 未来 10 日 GLD 收益 |
| `fwd_ret_20d` | 未来 20 日 GLD 收益 |

最终主策略选择表现最稳的 horizon，而不是只选择历史收益最高的窗口。

### 7.2 IC 分析

每个因子分别计算：

- Pearson IC。
- Spearman rank IC。
- IC 均值。
- IC 标准差。
- ICIR。
- IC 正值比例。
- 滚动 IC 稳定性。

重点判断因子在不同年份、不同市场阶段、不同 horizon 下是否稳定。

### 7.3 分层收益分析

对每个因子做分位数组合分析：

- 按因子值分成 5 组或 10 组。
- 观察未来 5 / 10 / 20 日平均收益。
- 检查单调性。
- 检查最高组与最低组收益差。
- 对只做多策略重点关注高分组是否显著优于全样本均值。

由于本策略只做多，long-short spread 只作为因子方向性辅助，不作为最终交易策略。

### 7.4 稳定性检验

因子稳定性至少从以下角度检查：

| 检验 | 说明 |
|---|---|
| 年度 IC | 每年单独计算 IC |
| 滚动窗口 IC | 例如 3 年滚动 |
| 分市场阶段 | 牛市、熊市、震荡期 |
| 样本外表现 | 训练期筛选，测试期验证 |
| 成本后表现 | 不只看毛收益 |

### 7.5 相关性与去冗余

为避免多个相似因子重复计分，需要：

- 计算因子相关矩阵。
- 对高度相关因子聚类。
- 每个相关性簇只保留一个代表因子。
- 优先保留解释性强、样本外稳定、换手低的因子。

## 8. 因子筛选规则

因子筛选采用多条件评分，不按单一历史收益排序。

一个因子进入候选池需要满足：

| 维度 | 要求 |
|---|---|
| 方向稳定 | IC 方向大体一致 |
| 分层有效 | 高分组未来收益更好 |
| 样本外不过度衰减 | 测试期不完全失效 |
| 相关性不过高 | 不与已有因子严重重复 |
| 解释合理 | 符合黄金定价逻辑 |
| 换手可控 | 不导致过度交易 |

因子评分表字段：

```text
factor_name
category
best_horizon
ic_mean
ic_ir
top_quantile_return
monotonicity_score
stability_score
correlation_group
selected_flag
reason
```

最终选出 5–10 个因子进入策略合成。

## 9. 策略信号设计

最终策略采用“多策略对比 + 组合策略加权/投票”的设计。

先比较三类策略：

1. 技术因子策略。
2. 宏观因子策略。
3. 技术 + 宏观组合策略。

再在组合策略内部使用等权投票或等权打分生成最终信号。

### 9.1 因子标准化

不同因子的方向和量纲不一致，需要统一处理：

- rolling z-score。
- rolling percentile rank。
- winsorize 极端值。
- 按因子方向调整符号。

方向示例：

- GLD 动量越强，越看多。
- 实际利率上行，越看空。
- 美元走强，越看空。
- 通胀预期上行，倾向看多，但以实证结果为准。
- 波动率过高时可能降低信号置信度。

### 9.2 投票法 baseline

每个有效因子生成一个方向判断：

```text
bullish = 1
neutral = 0
bearish = -1
```

综合分数：

```text
score = bullish_votes / total_votes
```

持仓规则：

```text
如果 score >= threshold，则持有 GLD
否则空仓
```

候选 threshold 为 0.5 / 0.6 / 0.7。最终只能选择一个主参数，不允许用测试集反复调参。

### 9.3 加权打分法

每个因子根据标准化值和权重贡献分数：

```text
factor_score = standardized_factor * factor_weight
total_score = sum(factor_score)
```

权重来源可以包括：

- 等权。
- ICIR。
- 稳定性评分。
- 分层收益评分。

第一版主策略使用等权投票或等权打分，ICIR 加权只作为补充分析，避免过拟合。

### 9.4 最小持仓期与换手约束

日频信号可能频繁跳变，因此加入交易约束：

- 每日计算 raw signal。
- 只有当距离上一次交易超过最小持仓期时，才允许换仓。
- 最小持仓期测试 5 / 10 / 20 个交易日。
- 主策略选择样本外表现稳定且换手不过高的参数。

信号输出字段：

```text
date
raw_score
raw_signal
final_signal
position
days_since_last_trade
trade_flag
reason
```

## 10. 回测设计

### 10.1 回测假设

| 项目 | 设定 |
|---|---|
| 标的 | GLD |
| 方向 | 只做多 |
| 空仓资产 | 现金，默认收益 0；可扩展 T-bill |
| 信号生成 | 当日收盘后生成 |
| 成交价格 | 下一交易日成交，第一版默认下一日收盘 |
| 交易成本 | 主结果单边 5bp |
| 成本敏感性 | 0bp / 5bp / 10bp |
| 杠杆 | 无 |
| 做空 | 无 |
| 仓位 | 0 或 1 |
| 分红 | 使用 adjusted close 处理 |

### 10.2 策略收益计算

核心公式：

```text
strategy_return[t] = position[t-1] * GLD_return[t] - transaction_cost[t]
```

交易成本：

```text
transaction_cost[t] = abs(position[t] - position[t-1]) * cost_rate
```

该设计避免使用当天信号赚当天收益。

### 10.3 绩效指标

输出完整指标表：

| 指标 | 含义 |
|---|---|
| Annual Return | 年化收益 |
| Annual Volatility | 年化波动 |
| Sharpe | 夏普比率 |
| Sortino | 下行风险调整收益 |
| Calmar | 年化收益 / 最大回撤 |
| Max Drawdown | 最大回撤 |
| Win Rate | 日收益胜率 |
| Exposure | 持仓时间占比 |
| Turnover | 年化换手 |
| Number of Trades | 交易次数 |
| Average Holding Days | 平均持仓天数 |
| Best Year / Worst Year | 最好/最差年份 |
| Excess Return vs GLD | 相对 GLD 超额 |
| Drawdown Reduction | 相对 GLD 回撤降低 |

核心评价标准：

- 是否降低最大回撤。
- 是否提高 Sharpe / Sortino / Calmar。
- 是否在样本外仍然有效。
- 是否在 5bp 和 10bp 成本下仍然有效。
- 是否相对 GLD buy-and-hold 有实际价值。

## 11. 验证设计

### 11.1 固定训练/测试切分

建议切分：

```text
训练期：GLD 上市后至 2016-12-31
测试期：2017-01-01 至最新数据
```

训练期用于：

- 因子筛选。
- 确定因子方向。
- 确定信号合成方法。
- 确定主 horizon。
- 确定主成本假设。

测试期用于：

- 严格样本外回测。
- 与 GLD buy-and-hold 比较。
- 检查过拟合。

### 11.2 Walk-forward 验证

模拟真实研究流程：

```text
使用过去 N 年数据筛因子/参数
在未来 M 个月或 1 年测试
窗口向前滚动
```

推荐配置：

```text
训练窗口：5 年
测试窗口：1 年
滚动频率：每年一次
```

输出字段：

```text
walk_forward_year
selected_factors
annual_return
sharpe
max_drawdown
turnover
outperformed_gld
```

### 11.3 稳健性检查

至少包含：

- 不同 horizon：5 / 10 / 20 日。
- 不同成本：0bp / 5bp / 10bp。
- 不同最小持仓期：5 / 10 / 20 日。
- 技术策略 vs 宏观策略 vs 组合策略。
- 等权打分 vs ICIR 加权。
- 样本内 vs 样本外。
- 年度收益拆分。
- 回撤阶段拆分。

## 12. 报告输出设计

### 12.1 Notebook

两个 notebook：

```text
01_factor_research.ipynb
02_strategy_backtest.ipynb
```

`01_factor_research.ipynb` 内容：

- 数据加载。
- 因子构造。
- 因子覆盖率检查。
- IC 分析。
- 分层收益。
- 因子相关矩阵。
- 因子筛选表。
- 技术因子 vs 宏观因子对比。

`02_strategy_backtest.ipynb` 内容：

- 策略信号生成。
- 策略净值曲线。
- 与 GLD buy-and-hold 对比。
- 回撤曲线。
- 年度收益。
- 成本敏感性。
- walk-forward 结果。
- 最新信号。

### 12.2 HTML 报告

HTML 报告结构：

```text
1. Executive Summary
2. Strategy Definition
3. Data and Factor Universe
4. Factor Research Results
5. Selected Factors
6. Signal Construction
7. Backtest Results
8. Out-of-Sample Validation
9. Cost Sensitivity
10. Latest Signal
11. Limitations and Next Steps
```

报告需要突出：

- 本项目做的是 GLD 只做多择时，不是预测金价本身。
- 策略是否降低回撤。
- 策略是否改善风险收益比。
- 宏观因子是否比纯技术策略有增益。
- 最新信号是持有还是空仓。

### 12.3 最新信号表

输出 CSV / HTML 表：

```text
date
target
latest_close
strategy_score
signal
position
last_trade_date
days_since_last_trade
recommended_action
reason
```

候选动作：

```text
HOLD_GLD
STAY_CASH
BUY_GLD
SELL_TO_CASH
NO_CHANGE
```

## 13. 错误处理与数据质量

### 13.1 数据缺失

- GLD 价格缺失：不能回测该日期。
- 宏观数据缺失：允许前向填充。
- 连续缺失超过阈值：报告中警告。
- 因子缺失：该因子在该日期不参与打分。

### 13.2 数据延迟

FRED 宏观数据可能存在发布日期延迟和历史修正。第一版默认使用历史修正后的日频数据，但报告限制中必须明确：

```text
第一版未严格处理宏观数据发布滞后，后续实盘化需要引入 point-in-time 数据。
```

### 13.3 异常值

- 对因子做 winsorize。
- 对极端收益做检查，但不随意删除。
- 对缺失和异常数据生成数据质量报告。

### 13.4 数据缓存

缓存目录：

```text
data/raw/
data/processed/
```

要求：

- 保存原始数据。
- 保存处理后面板。
- 报告记录数据更新时间。
- 支持 `force_refresh` 参数重新拉取。

## 14. 测试设计

### 14.1 因子测试

- 动量因子滞后正确。
- 均线因子没有未来数据。
- 波动率计算符合预期。
- z-score 只使用历史窗口。
- 宏观数据前向填充正确。

### 14.2 回测测试

- `position[t-1]` 对应 `return[t]`。
- 交易成本只在仓位变化时扣除。
- 只做多约束有效，position 只能是 0 或 1。
- 最小持仓期约束生效。
- 空仓时收益为 0。
- buy-and-hold 基准计算正确。

### 14.3 指标测试

- 年化收益。
- 年化波动。
- Sharpe。
- 最大回撤。
- Calmar。
- 换手率。
- 持仓天数。

### 14.4 报告测试

- HTML 报告可生成。
- 最新信号表字段完整。
- 关键图表文件存在。
- 核心指标不为空。

## 15. 技术栈

使用 Python 核心框架 + Notebook 展示。

主要依赖：

```text
pandas
numpy
yfinance
pandas-datareader
scipy
statsmodels
scikit-learn
matplotlib
plotly
jupyter
nbconvert
jinja2
pytest
```

项目管理推荐：

```text
uv + pyproject.toml
```

后续命令形态可以设计为：

```text
python -m gold_strategy.run_research
python -m gold_strategy.run_backtest
python -m gold_strategy.generate_report
python -m gold_strategy.latest_signal
```

## 16. 成功标准

### 16.1 研究成功标准

- 成功构造技术因子和宏观因子。
- 输出每个因子的 IC、分层收益、稳定性结果。
- 能解释最终入选因子的原因。
- 能比较技术策略、宏观策略、组合策略。

### 16.2 回测成功标准

- 能从 GLD 上市以来完整回测。
- 能输出 buy-and-hold 对比。
- 能执行 0bp / 5bp / 10bp 成本敏感性。
- 能执行训练/测试切分。
- 能执行 walk-forward 验证。
- 能输出最新持仓信号。

### 16.3 策略质量目标

组合策略在样本外应尽量满足：

- Sharpe 高于 GLD buy-and-hold。
- 最大回撤低于 GLD buy-and-hold。
- Calmar 高于 GLD buy-and-hold。
- 成本 10bp 下不完全失效。
- 换手率不过高。
- 策略表现不是只依赖单一历史阶段。

不强制要求年化收益一定高于 GLD，因为只做多择时策略可能通过降低暴露和回撤来提高风险收益比。

## 17. 主要风险与限制

### 17.1 过拟合风险

黄金样本约 20 年，对机器学习或复杂参数搜索来说仍然不大。需要避免：

- 反复调阈值。
- 选择历史最优窗口。
- 同时试太多因子后只展示最好的。
- 用测试集参与因子筛选。

### 17.2 宏观数据未来函数风险

FRED 数据可能存在发布日期滞后和历史修正。第一版研究可以接受，但必须在报告中说明。若要实盘化，需要 point-in-time 数据。

### 17.3 免费数据源质量风险

Yahoo Finance 免费数据可能存在：

- 调整价格问题。
- 偶发缺失。
- 下载失败。
- symbol 变更。
- 与专业数据源有差异。

因此数据层要支持后续替换。

### 17.4 交易现实风险

GLD 流动性较好，但第一版仍然只是研究原型，不考虑：

- 大资金冲击成本。
- 税务。
- 券商实际手续费差异。
- 现金收益。
- 融资融券。

## 18. 实施边界

第一版必须完成：

```text
数据层
技术因子
宏观因子
因子研究
因子筛选
信号合成
只做多回测
成本敏感性
样本外验证
walk-forward
HTML 报告
最新信号表
基础测试
```

第一版不做：

```text
机器学习预测模型
实时交易
期货策略
多品种组合
参数自动寻优平台
复杂 Web UI
数据库系统
定时任务系统
```

## 19. 后续扩展方向

二期可以扩展：

- IAU 或国内黄金 ETF。
- XAUUSD。
- COMEX GC 连续合约。
- 空仓现金收益或 T-bill 收益。
- regime detection。
- 贝叶斯或状态空间模型。
- point-in-time 宏观数据。
- 定时生成报告。

## 20. 决策记录

本轮 brainstorming 已确认：

| 决策 | 结论 |
|---|---|
| 项目类型 | 交易策略原型 |
| 标的 | 黄金 ETF，第一版使用 GLD |
| 交易方向 | 只做多 |
| 调仓频率 | 日频生成信号，加入最小持仓期 |
| 因子路线 | 先技术 baseline，再加入宏观因子 |
| 回测区间 | GLD 上市以来 |
| 目标 | 收益、风险调整收益、最大回撤、相对 GLD 综合评估 |
| 交易成本 | 主结果 5bp，附加 0bp / 5bp / 10bp 敏感性 |
| 数据源 | Yahoo Finance + FRED，预留本地数据替换 |
| 建模方法 | 先统计因子研究，再固化成规则 |
| 预测目标 | 同时评估未来 5 / 10 / 20 日收益 |
| 输出形式 | Notebook + HTML 报告 + 最新信号表 |
| 技术栈 | Python 核心框架 + Notebook 展示 |
| 验证方式 | 固定样本外验证 + walk-forward |
| 信号方式 | 技术、宏观、组合策略对比；组合策略内部加权/投票 |

## 21. 推荐实现顺序

后续 implementation plan 应按以下顺序拆分：

1. 初始化项目结构与 Python 环境。
2. 实现数据获取、缓存和日频面板构建。
3. 实现技术因子。
4. 实现宏观因子。
5. 实现 forward return 和因子研究函数。
6. 实现因子筛选与去冗余。
7. 实现信号合成与最小持仓期约束。
8. 实现只做多回测引擎。
9. 实现绩效指标与基准对比。
10. 实现训练/测试切分和 walk-forward 验证。
11. 实现图表、HTML 报告和最新信号表。
12. 补充测试。
13. 运行完整研究流程并产出报告。
