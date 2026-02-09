# 股票日内交易分析工具

基于「内功（基本面）+招式（技术面）」融合分析，提供趋势选股、日内做T、日线综合分析。

## ✨ 特性

- 🎯 **100分制评分**：基本面50分 + 技术面50分
- 📊 **多级别趋势分析**：周线/日线/分时多级别判断
- 🔄 **钟摆模型**：基于均线偏离度的买卖时机判断
- 🚀 **多数据源**：baostock（主）+ akshare（备），自动切换
- ⚡ **智能缓存**：5-10分钟TTL，响应速度提升1000倍
- 🛡️ **稳定可靠**：容错处理，避免限流

## 📦 安装

### 方法一：导入 Cursor Agent Skill（推荐）

1. **克隆或下载本项目**

```bash
git clone https://github.com/clcc2019/stock-intraday-trading.git
# 或直接下载 ZIP 解压
```

2. **复制到 Cursor Skills 目录**

```bash
# Linux/Mac
cp -r stock-intraday-trading ~/.cursor/skills/

# Windows
xcopy /E /I stock-intraday-trading %USERPROFILE%\.cursor\skills\stock-intraday-trading
```

3. **安装 Python 依赖**

```bash
pip install baostock pandas numpy akshare adata
```

4. **重启 Cursor**

重启后，Skill 会自动加载，可以直接通过自然语言调用。

### 方法二：命令行直接使用

```bash
# 克隆项目
git clone https://github.com/clcc2019/stock-intraday-trading.git
cd stock-intraday-trading

# 安装依赖
pip install baostock pandas numpy akshare adata

# 直接运行脚本
python3 scripts/analyze_stock_simple.py 600519
```

## 🚀 快速开始（Agent Skill 使用）

导入 Skill 后，可通过自然语言直接调用，无需记忆命令行参数。

### 1. 日线综合分析

直接询问 AI：
```
分析一下贵州茅台
600519 是否适合买入？
帮我看看比亚迪的技术面和基本面
```

AI 会自动执行日线分析脚本，输出：
- 多级别趋势分析（周线/日线）
- 钟摆位置（均线偏离度）
- 基本面评分（盈利/成长/健康/估值/资金）
- 综合买卖建议（100分制评分）

### 2. 趋势选股

直接询问 AI：
```
帮我从沪深300中选几只趋势向上的股票
筛选一些适合做T的标的
推荐一些基本面好且技术面强的股票
```

AI 会自动执行选股脚本，输出：
- 符合条件的股票列表（按综合得分排序）
- 技术面强度 + 基本面评分
- 最佳做T候选（回踩均线附近的标的）

### 3. 日内做T分析

直接询问 AI：
```
贵州茅台适合做T吗？
分析一下中国动力的做T机会
帮我看看今天能不能做T
```

AI 会自动执行做T分析脚本，输出：
- 多级别趋势判断（大势方向）
- 钟摆位置（当前是否适合做T）
- 日内交易机会和关键价位
- 风险提示

### 4. 命令行直接使用（可选）

如需直接运行脚本：

```bash
# 日线分析
python3 scripts/analyze_stock_simple.py 600519

# 选股
python3 scripts/select_stocks.py --index hs300 --top 10

# 做T分析
python3 scripts/analyze_intraday_t0.py 600519

# 策略回测
python3 scripts/backtest_strategy.py 600519

# 生成可视化图表
python3 scripts/generate_chart_data.py 600519 --open
```

## 📊 评分体系

### 基本面（50分）
- 盈利能力：15分（ROE、净利率、毛利率）
- 成长能力：10分（营收增长、利润增长）
- 财务健康：10分（资产负债率、流动比率、现金流）
- 估值水平：10分（PE、机构参与度）
- 资金面：5分（主力资金、股东户数）

### 技术面（50分）
- 趋势方向：15分（多头排列、MA斜率）
- 钟摆位置：12.5分（均线偏离度）
- 趋势强度：10分（高低点、成交量）
- 量价关系：7.5分（量比、放量）
- 辅助指标：5分（MACD、KDJ）

## 🎯 投资哲学

### 两条公设
1. **价格围绕价值波动** → 均线 = 价值中枢
2. **钟摆式过度波动** → 均线偏离度 = 钟摆位置

### 核心原则
- **顺大势**：只做趋势向上的股票（均线多头排列）
- **逆小势**：在钟摆回摆时买入（均线偏离度过大时）
- **内功为本**：基本面决定长期方向
- **招式为辅**：技术面把握买卖时机

## 🔧 性能优化

### 多数据源自动切换
- 主数据源：baostock（稳定、免费、不限流）— 历史K线
- 备用数据源：akshare（自动降级）— 历史K线备用
- 补充数据源：adata（实时行情、资金流向、分时行情、5档盘口）
- 智能重试和容错

### 智能缓存机制
- 技术面数据：5分钟缓存
- 基本面数据：10分钟缓存
- 缓存命中：1秒 → 1毫秒（1000倍提升）

### 查询优化
- 批量查询接口
- 轻量级评分模式
- 容错处理

## 📈 性能对比

| 操作 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 单股分析 | ~10秒 | ~8秒 | 20% |
| 重复查询 | ~10秒 | ~0.01秒 | 1000x |
| 选股300只 | ~30分钟 | ~15分钟 | 50% |
| API限流 | 频繁 | 极少 | 90%↓ |

## ⚠️ 免责声明

**本工具仅供学习研究和技术交流使用，不构成任何投资建议。**

1. **非投资建议**：所有分析结果均基于历史数据和数学模型自动生成
2. **投资风险**：股票投资具有高风险性，过往表现不代表未来收益
3. **数据准确性**：依赖第三方数据源，不保证数据准确性
4. **模型局限性**：任何量化模型都无法完全预测市场走势
5. **免责条款**：使用本工具造成的任何损失，开发者不承担责任

**投资有风险，入市需谨慎。**

## 📚 参考文档

- [投资哲学](references/philosophy.md) - 核心理论
- [技术指标详解](references/indicators.md)
- [做T策略详解](references/t0-strategies.md)
- [量价关系分析](references/volume-price.md)
- [市场环境分析](references/market-environment.md)
- [更新日志](CHANGELOG.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
