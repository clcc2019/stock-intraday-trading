---
name: stock-intraday-trading
description: 中国A股选股、日线分析、日内做T和回测技能。适用于A股/沪深京股票/沪深300/中证500/全A筛选、买入候选、趋势股、底部反弹、超跌低估、个股分析、T+0、K线图；用户未说明市场时默认按中国A股处理，不筛美股或港股。
---

# 中国 A 股选股与日内做 T

## 默认边界

- 默认市场是中国 A 股。用户说“股票、选股、筛几个适合买入、趋势股、超跌、做T”但未说明市场时，按 A 股处理。
- 不要把美股、港股、基金、ETF、期货、可转债混入 A 股候选；用户明确要求其他市场时，先说明本技能的数据源和脚本不适用。
- A 股做 T 默认指“已有底仓”的日内高抛低吸。无底仓时不能把当天买入当天卖出表述为可执行交易。
- 行情、技术面、基本面评分和回测以本地脚本为准。不要用常识、热度榜或泛互联网信息替代脚本结果。
- 新闻、公告、监管、行业政策等定性催化只能作为补充，不能替代选股和回测闸门。

## 核心框架

- 投资哲学：内功为本，招式为辅；基本面定方向，技术面把时机。
- 趋势原则：顺大势、逆小势；优先找均线多头、MA20 向上、回踩均线簇附近的股票。
- 钟摆模型：均线是价值中枢代理，价格偏离越大，回归潜力和趋势反转风险同时增加。
- 评分体系：基本面 50 分 + 技术面 50 分。
- 见顶过滤：见顶/出货评分 >=70 直接排除，>=50 降级观望，>=30 标记风险。
- 底部反弹：只找“基本面尚可 + 跌够了 + 出现底部信号”的候选，避免接飞刀。

默认赚钱闸门：
总收益 > 0，年化收益 >= 5%，最大回撤 <= 15%，交易次数 >= 6，盈亏比 >= 1.2，单笔期望 > 0，相对买入持有超额 >= 0。

## 意图识别

- “选股 / 筛选 / 推荐 / 买什么 / 适合买入”：
  先运行趋势选股，再对前 5-10 个候选运行回测闸门。
- “底部 / 反弹 / 超跌 / 低估 / 抄底 / 错杀 / 跌够了”：
  使用底部反弹策略。
- “全市场 / 全A股 / 不限定指数 / 多筛一点”：
  使用 `--index all`，并提醒会更慢。
- “沪深300 / 大盘蓝筹 / 快速筛选”：
  使用 `--index hs300`；更快可用 `--index sz50`。
- “中证500 / 中小盘”：
  使用 `--index zz500`。
- “做T / 日内 / 分时 / 今天能不能T”：
  使用日内做 T 脚本；必须按 A 股底仓约束解释。
- “回测 / 历史收益 / 能不能赚钱 / 稳不稳定”：
  使用回测脚本。
- “K线 / 图表 / 画图”：
  使用图表脚本。
- 单只股票代码或名称：
  使用日线分析脚本；若涉及买入结论，再运行回测脚本。

常见 A 股代码：贵州茅台 `600519`，比亚迪 `002594`，恒瑞医药 `600276`，宁德时代 `300750`。
如果用户只给股票名称且代码不确定，先用已有常识识别；仍不确定时简短询问，不要猜。

## 执行命令

在技能根目录运行命令。若当前目录不是技能根目录，先切到包含 `SKILL.md` 和 `scripts/` 的目录。

趋势选股，默认宽基池 `wide`（沪深300 + 中证500）：

```bash
python3 scripts/select_stocks.py --strategy trend --index wide --top 30
```

底部反弹选股：

```bash
python3 scripts/select_stocks.py --strategy bottom --index wide --top 30
```

全 A 股筛选：

```bash
python3 scripts/select_stocks.py --strategy trend --index all --top 50
python3 scripts/select_stocks.py --strategy bottom --index all --top 50
```

日线分析：

```bash
python3 scripts/analyze_stock_simple.py 股票代码
```

日内做 T 分析：

```bash
python3 scripts/analyze_intraday_t0.py 股票代码
```

回测 / 赚钱闸门：

```bash
python3 scripts/backtest_strategy.py 股票代码 --days 900 --position-pct 0.5
```

压力测试：

```bash
python3 scripts/backtest_strategy.py 股票代码 --days 1200 --position-pct 0.5 --slippage 0.001 --min-annual-return 8
```

策略 C 参数扫描：

```bash
python3 scripts/optimize_strategy_c.py --multi --days 900 --position-pct 0.5
```

图表：

```bash
python3 scripts/generate_chart_data.py 股票代码 --open
```

## 买入类推荐流程

用户要求“推荐、买什么、适合买入、能不能赚钱”时，必须执行完整流程：

1. 先运行选股脚本。默认 `--strategy trend --index wide --top 30`；用户提到低估/超跌时改用 `--strategy bottom`；用户要求全市场时加 `--index all`。
2. 从选股输出中取排名靠前且没有明显高位/见顶风险的 5-10 只候选。
3. 逐只运行 `backtest_strategy.py`。优先看策略 C，同时报告 A/B/C 的闸门情况。
4. 只有回测闸门通过且样本外表现没有明显恶化，才能列为“可交易候选”。
5. 闸门未通过但技术面或基本面较好的，只能列为“观察候选”，必须写明失败原因，例如年化不足、最大回撤过大、交易次数不足、样本外为负或跑输买入持有。
6. 如果没有股票通过闸门，明确说“本轮没有可交易候选”，不要为了满足请求硬给买入名单。

当前研究基线：优先查看策略 C；A/B 在 2024-03-11 至 2026-05-25 的代表股票回测中整体为负。策略输出只代表历史研究，不保证未来收益。

## 输出规范

- 先给结论分组：`可交易候选`、`观察候选`、`剔除/暂不碰`。
- 每只股票至少列出：代码、名称、当前价格、选股理由、关键风险、回测闸门状态。
- 买入相关表述必须使用“候选、观察、等待回踩、通过/未通过闸门”等概率性语言。
- 不要使用“必涨、稳赚、确定买入、无脑买、闭眼买”等确定性表达。
- 可简要总结脚本结果，但关键表格、闸门结果和失败原因必须保留。
- 如果用户只要快速答案，也要说明是否已跑脚本；没跑脚本时不能给买入结论。

## 数据源与依赖

- 行情优先使用 `stock-api`（腾讯/新浪/东方财富自动兜底），缺失时脚本会降级到 baostock/akshare。
- 股票列表和指数成分主要由 baostock 提供。
- 基本面评分依赖 akshare；如失败，只能说明基本面增强不可用。
- 实时行情、分时和盘口增强依赖 adata。

依赖缺失时，提示在技能根目录安装：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-fundamental.txt
python3 -m pip install -r requirements-realtime.txt
npm install
```

脚本失败时，展示错误并给出最小修复建议；不要用理论分析替代失败脚本。

## 参考资料

按需读取，不要一次性加载全部：

- `references/philosophy.md`：投资哲学、均线价值中枢、钟摆模型。
- `references/indicators.md`：MACD、KDJ、RSI、布林带等指标细节。
- `references/volume-price.md`：量价关系、放量滞涨、缩量企稳。
- `references/t0-strategies.md`：A 股底仓做 T 场景和分时策略。
- `references/market-environment.md`：市场环境判断。
- `references/research-log.md`：历史回测研究记录。

## 严格禁止

- 对 A 股筛选请求输出美股或港股候选。
- 脚本前写冗长说明，拖延执行。
- 没跑脚本就给“适合买入”的股票名单。
- 脚本失败后用理论、常识或网络热度补一份推荐。
- 把未通过样本外验证的技术信号说成确定性盈利机会。
