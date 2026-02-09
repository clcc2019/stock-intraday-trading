---
name: stock-intraday-trading
description: 股票趋势分析与日内做T。100分制评分（基本面50+技术面50），支持选股/做T/日线分析/回测/图表。
---

# 股票趋势分析与日内做T

## 投资哲学

### 两条公设
1. **价格围绕价值波动** — 均线是价值中枢，价格偏离后必然回归
2. **钟摆式过度波动** — 价格因贪婪恐惧过度摆动，偏离越远回归力越大

### 核心原则
- **内功为本招式为辅** — 基本面定方向，技术面把时机
- **顺大势逆小势** — 顺应大趋势，利用小回调进出
- **均线 = 价值中枢** — 均线方向=价值方向，价格与均线距离=钟摆位置

### 评分体系（100分）
- **基本面50分**：盈利(15)+成长(10)+健康(10)+估值(10)+资金(5)
- **技术面50分**：趋势(15)+钟摆(12.5)+强度(10)+量价(7.5)+指标(5)

### 均线层级
| MA250(年线) | MA120(半年) | MA60(季线) | MA20(月线/中枢) | MA5/10(短期情绪) |

## AI执行指令

### 1. 意图识别
- "选股/筛选/推荐" → 选股脚本
- "做T/日内/分时" → T+0脚本
- "回测/历史收益" → 回测脚本
- "图表/K线图" → 图表生成脚本
- 其他 → 日线分析脚本

### 2. 代码识别
常见：贵州茅台(600519)、比亚迪(002594)、恒瑞医药(600276)、宁德时代(300750)

### 3. 执行脚本

**选股**：
```bash
cd .cursor/skills/stock-intraday-trading && python3 scripts/select_stocks.py --index hs300
# 可选: --index zz500/sz50, --no-fundamental(跳过基本面), --sector 板块名
```

**做T分析**：
```bash
cd .cursor/skills/stock-intraday-trading && python3 scripts/analyze_intraday_t0.py 股票代码
```

**日线分析**：
```bash
cd .cursor/skills/stock-intraday-trading && python3 scripts/analyze_stock_simple.py 股票代码
```

**回测**：
```bash
cd .cursor/skills/stock-intraday-trading && python3 scripts/backtest_strategy.py 股票代码
```

**图表**：
```bash
cd .cursor/skills/stock-intraday-trading && python3 scripts/generate_chart_data.py 股票代码 --open
```

### 4. 展示结果
- 直接展示脚本完整输出
- 可用1-2句总结核心建议
- 提醒关注定性因素（管理层、行业格局）

### 严格禁止
- 脚本前的冗长说明
- "需要您验证"等推诿话术
- 脚本失败时的理论替代

## 错误处理

脚本失败时：显示错误信息，建议检查代码或网络，不输出理论分析。
