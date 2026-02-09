#!/usr/bin/env python3
"""
MACD/KDJ 策略历史回测系统

对比两种策略的历史收益：
  A) 核心信号策略：仅依据 MACD(8,17,9) / KDJ(6,3,3) 金叉死叉交叉信号
  B) 完整评分策略：复用 analyze_stock_simple.py 的 20 分制综合评分体系

用法：
  python3 backtest_strategy.py 600276              # 单只股票回测
  python3 backtest_strategy.py --multi              # 预设4只代表性股票对比
  python3 backtest_strategy.py 600519 002594 600276 # 多只自定义股票
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# 导入统一数据源和公共技术指标
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_source import DataSource
from technical import calculate_ma, calculate_macd, calculate_kdj, calculate_rsi, calculate_volume_ma

# ============================================================
# 常量
# ============================================================
INITIAL_CAPITAL = 100_000.0   # 初始资金 10 万
COMMISSION_RATE = 0.00025     # 佣金 万2.5（买卖都收）
STAMP_TAX_RATE = 0.0005       # 印花税 万5（仅卖出收）
BACKTEST_DAYS = 240           # 拉取约 240 个交易日数据（含指标预热）
SIGNAL_START_OFFSET = 60      # 前 60 天用于指标预热，不产生信号
STOP_LOSS_PCT = -3.0          # 止损线：-3%
TRAILING_ACTIVATE_PCT = 2.0   # 移动止盈激活线：+2%
TRAILING_STOP_PCT = 1.0       # 移动止盈保底线：+1%（回撤到此平仓）
MAX_HOLDING_DAYS = 20         # 最大持仓天数

PRESET_STOCKS = {
    '600519': '贵州茅台',
    '002594': '比亚迪',
    '600276': '恒瑞医药',
    '300750': '宁德时代',
}


# ============================================================
# 数据获取与指标计算
# ============================================================

def fetch_stock_data(stock_code, days=BACKTEST_DAYS):
    """获取历史日K线数据（使用统一 DataSource）"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = DataSource.get_stock_hist(
        stock_code=stock_code,
        start_date=start_date,
        end_date=end_date,
        adjust='qfq',
        period='daily'
    )
    return df if df is not None and not df.empty else None


def get_stock_name(stock_code):
    """尝试获取股票名称"""
    if stock_code in PRESET_STOCKS:
        return PRESET_STOCKS[stock_code]
    return stock_code


def calculate_indicators(df):
    """计算全部技术指标（使用公共模块）"""
    calculate_ma(df, windows=[5, 10, 20])
    calculate_macd(df)
    calculate_kdj(df)
    calculate_rsi(df)
    calculate_volume_ma(df, windows=[5])

    # MACD 背离（逐日滚动检测，回测专用）
    df['MACD_divergence'] = 'none'
    for idx in range(30, len(df)):
        window = df.iloc[idx - 30:idx + 1].copy()
        df.iloc[idx, df.columns.get_loc('MACD_divergence')] = _detect_divergence(window)

    return df


def _detect_divergence(window):
    """在给定窗口中检测 MACD 背离"""
    if len(window) < 7:
        return 'none'

    divergence = 'none'

    # 底背离
    price_lows = []
    for i in range(2, len(window) - 2):
        if (window.iloc[i]['收盘'] < window.iloc[i - 1]['收盘'] and
                window.iloc[i]['收盘'] < window.iloc[i - 2]['收盘'] and
                window.iloc[i]['收盘'] <= window.iloc[i + 1]['收盘'] and
                window.iloc[i]['收盘'] <= window.iloc[i + 2]['收盘']):
            price_lows.append((i, window.iloc[i]['收盘'], window.iloc[i]['DIF']))

    if len(price_lows) >= 2:
        last_low = price_lows[-1]
        prev_low = price_lows[-2]
        if last_low[1] < prev_low[1] and last_low[2] > prev_low[2]:
            divergence = 'bottom'

    # 顶背离
    price_highs = []
    for i in range(2, len(window) - 2):
        if (window.iloc[i]['收盘'] > window.iloc[i - 1]['收盘'] and
                window.iloc[i]['收盘'] > window.iloc[i - 2]['收盘'] and
                window.iloc[i]['收盘'] >= window.iloc[i + 1]['收盘'] and
                window.iloc[i]['收盘'] >= window.iloc[i + 2]['收盘']):
            price_highs.append((i, window.iloc[i]['收盘'], window.iloc[i]['DIF']))

    if len(price_highs) >= 2:
        last_high = price_highs[-1]
        prev_high = price_highs[-2]
        if last_high[1] > prev_high[1] and last_high[2] < prev_high[2]:
            divergence = 'top'

    return divergence


# ============================================================
# 策略 A：核心信号（MACD / KDJ 交叉）
# ============================================================

def strategy_a_signals(df):
    """
    Strategy A: MACD / KDJ 核心交叉信号
    返回 DataFrame，包含 signal 列 ('buy' / 'sell' / None) 及 reason 列
    """
    signals = pd.DataFrame(index=df.index, columns=['signal', 'reason'])
    signals['signal'] = None
    signals['reason'] = ''

    for i in range(1, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i - 1]

        buy_reasons = []
        sell_reasons = []

        # MACD 金叉
        if cur['DIF'] > cur['DEA'] and prev['DIF'] <= prev['DEA']:
            buy_reasons.append('MACD金叉')
        # MACD 死叉
        if cur['DIF'] < cur['DEA'] and prev['DIF'] >= prev['DEA']:
            sell_reasons.append('MACD死叉')

        # KDJ 低位金叉 (J < 30)
        if cur['K'] > cur['D'] and prev['K'] <= prev['D'] and cur['J'] < 30:
            buy_reasons.append(f'KDJ低位金叉(J={cur["J"]:.0f})')
        # KDJ 高位死叉 (J > 70)
        if cur['K'] < cur['D'] and prev['K'] >= prev['D'] and cur['J'] > 70:
            sell_reasons.append(f'KDJ高位死叉(J={cur["J"]:.0f})')

        # 双金叉共振
        macd_golden = cur['DIF'] > cur['DEA'] and prev['DIF'] <= prev['DEA']
        kdj_golden = cur['K'] > cur['D'] and prev['K'] <= prev['D']
        if macd_golden and kdj_golden:
            buy_reasons.append('MACD+KDJ双金叉共振')

        # 双死叉共振
        macd_death = cur['DIF'] < cur['DEA'] and prev['DIF'] >= prev['DEA']
        kdj_death = cur['K'] < cur['D'] and prev['K'] >= prev['D']
        if macd_death and kdj_death:
            sell_reasons.append('MACD+KDJ双死叉共振')

        # MA20 趋势守卫：价格低于MA20时阻止买入信号
        ma20 = cur['MA20'] if not np.isnan(cur['MA20']) else 0
        price_above_ma20 = cur['收盘'] > ma20 if ma20 > 0 else True

        if buy_reasons and not price_above_ma20:
            # 下跌趋势中阻止买入（双金叉共振除外，但降级为观望）
            if 'MACD+KDJ双金叉共振' not in buy_reasons:
                buy_reasons = []  # 清除买入信号
            else:
                buy_reasons.append('⚠️趋势偏弱')  # 保留但标记

        # 优先级：买入/卖出信号同时出现时取较强一侧
        if buy_reasons and not sell_reasons:
            signals.iloc[i, signals.columns.get_loc('signal')] = 'buy'
            signals.iloc[i, signals.columns.get_loc('reason')] = '+'.join(buy_reasons)
        elif sell_reasons and not buy_reasons:
            signals.iloc[i, signals.columns.get_loc('signal')] = 'sell'
            signals.iloc[i, signals.columns.get_loc('reason')] = '+'.join(sell_reasons)
        elif buy_reasons and sell_reasons:
            # 双金叉/双死叉优先，否则忽略矛盾信号
            if '双金叉共振' in ' '.join(buy_reasons):
                signals.iloc[i, signals.columns.get_loc('signal')] = 'buy'
                signals.iloc[i, signals.columns.get_loc('reason')] = '+'.join(buy_reasons)
            elif '双死叉共振' in ' '.join(sell_reasons):
                signals.iloc[i, signals.columns.get_loc('signal')] = 'sell'
                signals.iloc[i, signals.columns.get_loc('reason')] = '+'.join(sell_reasons)

    return signals


# ============================================================
# 策略 B：完整评分体系（复用 analyze_stock_simple.py 评分逻辑）
# ============================================================

def _score_day(df, idx):
    """
    对第 idx 行计算买入/卖出评分（复用 analyze_stock_simple.py 的 analyze() 逻辑）
    返回 (buy_score, sell_score, reason)
    """
    if idx < 2:
        return 0, 0, ''

    cur = df.iloc[idx]
    prev = df.iloc[idx - 1]
    prev2 = df.iloc[idx - 2]

    buy = 0
    sell = 0
    reasons = []

    # ── MACD (max 7) ──
    macd_buy = 0
    macd_sell = 0

    if cur['DIF'] > cur['DEA']:
        if prev['DIF'] <= prev['DEA']:
            macd_buy += 5
            reasons.append('MACD金叉')
        elif prev['DIF'] > prev['DEA'] and prev2['DIF'] <= prev2['DEA']:
            macd_buy += 4
            reasons.append('MACD金叉确认')
        else:
            macd_buy += 2
    else:
        if prev['DIF'] >= prev['DEA']:
            macd_sell += 5
            reasons.append('MACD死叉')
        elif prev['DIF'] < prev['DEA'] and prev2['DIF'] >= prev2['DEA']:
            macd_sell += 4
            reasons.append('MACD死叉确认')
        else:
            macd_sell += 2

    if cur['DIF'] > 0:
        macd_buy += 1
    elif cur['DIF'] < 0:
        if cur['DIF'] > cur['DEA'] and prev['DIF'] <= prev['DEA']:
            macd_buy += 2
        else:
            macd_sell += 1

    if cur['MACD'] > prev['MACD']:
        macd_buy += 1
    else:
        macd_sell += 1

    divergence = cur.get('MACD_divergence', 'none')
    if divergence == 'bottom':
        macd_buy += 3
        reasons.append('MACD底背离')
    elif divergence == 'top':
        macd_sell += 3
        reasons.append('MACD顶背离')

    buy += min(7, macd_buy)
    sell += min(7, macd_sell)

    # ── KDJ (max 7) ──
    kdj_buy = 0
    kdj_sell = 0
    j_val = cur['J']
    k_val = cur['K']
    d_val = cur['D']
    prev_k = prev['K']
    prev_d = prev['D']

    if k_val > d_val and prev_k <= prev_d:
        kdj_buy += 4
        reasons.append('KDJ金叉')
    elif k_val < d_val and prev_k >= prev_d:
        kdj_sell += 4
        reasons.append('KDJ死叉')
    elif k_val > d_val:
        kdj_buy += 1
    else:
        kdj_sell += 1

    if j_val < 0:
        kdj_buy += 3
    elif j_val < 20:
        kdj_buy += 3
        reasons.append(f'KDJ超卖J={j_val:.0f}')
    elif j_val > 100:
        kdj_sell += 3
    elif j_val > 80:
        kdj_sell += 3
        reasons.append(f'KDJ超买J={j_val:.0f}')
    elif j_val < 50:
        kdj_buy += 1
    else:
        kdj_sell += 1

    if j_val < 30 and k_val > d_val and prev_k <= prev_d:
        kdj_buy += 2
        reasons.append('KDJ低位金叉')
    elif j_val > 70 and k_val < d_val and prev_k >= prev_d:
        kdj_sell += 2
        reasons.append('KDJ高位死叉')

    buy += min(7, kdj_buy)
    sell += min(7, kdj_sell)

    # ── RSI (max 2) ──
    rsi = cur['RSI']
    if rsi < 30:
        buy += 2
    elif rsi > 70:
        sell += 2
    elif rsi < 45:
        buy += 1
    elif rsi > 55:
        sell += 1

    # ── MA (max 2) ──
    price = cur['收盘']
    if not np.isnan(cur['MA5']) and not np.isnan(cur['MA10']):
        if price > cur['MA5'] > cur['MA10']:
            buy += 2
        elif price < cur['MA5'] < cur['MA10']:
            sell += 2

    # ── Volume (max 2) ──
    vol_ma5 = cur['VOL_MA5']
    if vol_ma5 and vol_ma5 > 0:
        vol_ratio = cur['成交量'] / vol_ma5
        change_pct = ((cur['收盘'] - prev['收盘']) / prev['收盘']) * 100
        if vol_ratio > 1.5:
            if change_pct > 0:
                buy += 2
            else:
                sell += 2

    # ── MACD + KDJ 共振 (max 3) ──
    macd_golden = cur['DIF'] > cur['DEA'] and prev['DIF'] <= prev['DEA']
    macd_death = cur['DIF'] < cur['DEA'] and prev['DIF'] >= prev['DEA']
    kdj_golden = k_val > d_val and prev_k <= prev_d
    kdj_death = k_val < d_val and prev_k >= prev_d

    if macd_golden and kdj_golden:
        buy += 3
        reasons.append('双金叉共振')
    elif macd_death and kdj_death:
        sell += 3
        reasons.append('双死叉共振')
    elif cur['DIF'] > cur['DEA'] and kdj_golden and j_val < 30:
        buy += 2
    elif cur['DIF'] < cur['DEA'] and kdj_death and j_val > 70:
        sell += 2

    # ── MA20 趋势守卫（惩罚下跌趋势中的买入信号）──
    ma20 = cur['MA20']
    if not np.isnan(ma20):
        if price < ma20:
            sell += 2
            reasons.append('价格<MA20')
        # MA20 斜率检测（5日变化）
        if idx >= 5:
            ma20_prev5 = df.iloc[idx - 5]['MA20']
            if not np.isnan(ma20_prev5) and ma20 < ma20_prev5:
                sell += 1  # MA20 下行额外惩罚
                if price < ma20:
                    reasons.append('MA20下行')

    buy_score = min(20, buy)
    sell_score = min(20, sell)

    return buy_score, sell_score, '+'.join(reasons)


def strategy_b_signals(df):
    """
    Strategy B: 完整评分体系
    buy_score >= 10 → buy, sell_score >= 10 → sell
    """
    signals = pd.DataFrame(index=df.index, columns=['signal', 'reason'])
    signals['signal'] = None
    signals['reason'] = ''

    for i in range(2, len(df)):
        buy_s, sell_s, reason = _score_day(df, i)
        if buy_s >= 10 and buy_s > sell_s:
            signals.iloc[i, signals.columns.get_loc('signal')] = 'buy'
            signals.iloc[i, signals.columns.get_loc('reason')] = f'评分B{buy_s}/S{sell_s} {reason}'
        elif sell_s >= 10 and sell_s > buy_s:
            signals.iloc[i, signals.columns.get_loc('signal')] = 'sell'
            signals.iloc[i, signals.columns.get_loc('reason')] = f'评分B{buy_s}/S{sell_s} {reason}'

    return signals


# ============================================================
# 交易模拟引擎
# ============================================================

class TradeSimulator:
    """模拟交易执行器"""

    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0
        self.trades = []        # 已完成的完整交易 (买+卖)
        self.pending_buy = None  # 未平仓的买入记录
        self.equity_curve = []  # (date, equity)

    def _buy_cost(self, price, shares):
        return price * shares * COMMISSION_RATE

    def _sell_cost(self, price, shares):
        return price * shares * (COMMISSION_RATE + STAMP_TAX_RATE)

    def _close_position(self, sell_price, sell_date, sell_reason):
        """平仓辅助函数"""
        proceeds = sell_price * self.shares
        cost = self._sell_cost(sell_price, self.shares)
        self.cash += proceeds - cost
        pnl = (sell_price - self.pending_buy['buy_price']) / self.pending_buy['buy_price'] * 100
        self.trades.append({
            'buy_date': self.pending_buy['buy_date'],
            'buy_price': self.pending_buy['buy_price'],
            'sell_date': sell_date,
            'sell_price': sell_price,
            'shares': self.shares,
            'pnl_pct': pnl,
            'buy_reason': self.pending_buy['reason'],
            'sell_reason': sell_reason,
        })
        self.shares = 0
        self.pending_buy = None

    def execute_signals(self, df, signals, start_idx):
        """
        按信号执行交易。信号在 day i 产生，在 day i+1 的开盘价执行。
        包含止损、移动止盈、最大持仓天数等风控机制。
        start_idx: 信号开始有效的位置（跳过预热期）
        """
        self._peak_price = 0  # 持仓期间最高价（用于移动止盈）
        self._holding_days = 0  # 持仓天数

        for i in range(start_idx, len(df) - 1):
            exec_day = df.iloc[i + 1]
            exec_price = exec_day['开盘']
            exec_date = exec_day['日期']
            day_low = exec_day['最低']
            day_high = exec_day['最高']
            day_close = exec_day['收盘']

            # ── 风控检查（持仓中时，优先于信号处理）──
            if self.shares > 0 and self.pending_buy is not None:
                buy_price = self.pending_buy['buy_price']
                self._holding_days += 1

                # 更新持仓最高价
                if day_high > self._peak_price:
                    self._peak_price = day_high

                # 1) 止损检查：日内最低价触及-3%
                stop_loss_price = buy_price * (1 + STOP_LOSS_PCT / 100)
                if day_low <= stop_loss_price:
                    sell_at = stop_loss_price  # 以止损价成交
                    self._close_position(sell_at, exec_date, f'止损{STOP_LOSS_PCT}%')
                    self._peak_price = 0
                    self._holding_days = 0
                    equity = self.cash + self.shares * day_close
                    self.equity_curve.append((exec_date, equity))
                    continue

                # 2) 移动止盈：曾涨+2%后回落到仅+1%
                pnl_from_peak = (self._peak_price - buy_price) / buy_price * 100
                if pnl_from_peak >= TRAILING_ACTIVATE_PCT:
                    trailing_price = buy_price * (1 + TRAILING_STOP_PCT / 100)
                    if day_low <= trailing_price:
                        sell_at = trailing_price
                        self._close_position(sell_at, exec_date, f'移动止盈(峰值+{pnl_from_peak:.1f}%)')
                        self._peak_price = 0
                        self._holding_days = 0
                        equity = self.cash + self.shares * day_close
                        self.equity_curve.append((exec_date, equity))
                        continue

                # 3) 最大持仓天数
                if self._holding_days >= MAX_HOLDING_DAYS:
                    self._close_position(exec_price, exec_date, f'超时{MAX_HOLDING_DAYS}天')
                    self._peak_price = 0
                    self._holding_days = 0
                    equity = self.cash + self.shares * day_close
                    self.equity_curve.append((exec_date, equity))
                    continue

            # ── 信号处理 ──
            sig = signals.iloc[i]['signal']
            reason = signals.iloc[i]['reason']

            if sig == 'buy' and self.shares == 0:
                # 全仓买入
                max_shares = int(self.cash / (exec_price * (1 + COMMISSION_RATE)))
                max_shares = (max_shares // 100) * 100
                if max_shares <= 0:
                    equity = self.cash
                    self.equity_curve.append((exec_date, equity))
                    continue
                cost = self._buy_cost(exec_price, max_shares)
                self.cash -= exec_price * max_shares + cost
                self.shares = max_shares
                self.pending_buy = {
                    'buy_date': exec_date,
                    'buy_price': exec_price,
                    'shares': max_shares,
                    'reason': reason,
                }
                self._peak_price = day_high
                self._holding_days = 0

            elif sig == 'sell' and self.shares > 0 and self.pending_buy is not None:
                self._close_position(exec_price, exec_date, reason)
                self._peak_price = 0
                self._holding_days = 0

            # 记录每日权益
            equity = self.cash + self.shares * day_close
            self.equity_curve.append((exec_date, equity))

        # 补上最后一天的权益
        if len(df) > 0:
            last = df.iloc[-1]
            equity = self.cash + self.shares * last['收盘']
            if not self.equity_curve or self.equity_curve[-1][0] != last['日期']:
                self.equity_curve.append((last['日期'], equity))

    def get_metrics(self, df, start_idx):
        """计算回测绩效指标"""
        if not self.equity_curve:
            return {}

        equities = [e for _, e in self.equity_curve]
        final_equity = equities[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100

        # 年化收益
        first_date = pd.to_datetime(df.iloc[start_idx]['日期'])
        last_date = pd.to_datetime(df.iloc[-1]['日期'])
        days = (last_date - first_date).days
        if days > 0:
            annual_return = ((final_equity / self.initial_capital) ** (365 / days) - 1) * 100
        else:
            annual_return = 0

        # 最大回撤
        peak = equities[0]
        max_dd = 0
        for e in equities:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # 胜率 / 盈亏比
        wins = [t for t in self.trades if t['pnl_pct'] > 0]
        losses = [t for t in self.trades if t['pnl_pct'] <= 0]
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        total_profit = sum(t['pnl_pct'] for t in wins) if wins else 0
        total_loss = abs(sum(t['pnl_pct'] for t in losses)) if losses else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0

        # 平均持仓天数
        holding_days = []
        for t in self.trades:
            d1 = pd.to_datetime(t['buy_date'])
            d2 = pd.to_datetime(t['sell_date'])
            holding_days.append((d2 - d1).days)
        avg_holding = np.mean(holding_days) if holding_days else 0

        # 买入持有收益
        start_price = df.iloc[start_idx]['收盘']
        end_price = df.iloc[-1]['收盘']
        buy_hold_return = (end_price / start_price - 1) * 100

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'wins': len(wins),
            'losses': len(losses),
            'total_trades': len(self.trades),
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding,
            'buy_hold_return': buy_hold_return,
            'final_equity': final_equity,
        }


# ============================================================
# 回测执行与报告
# ============================================================

def run_backtest(stock_code, verbose=True):
    """对单只股票运行回测"""
    name = get_stock_name(stock_code)
    if verbose:
        print(f"\n📊 正在获取 {name}({stock_code}) 的历史数据...")

    df = fetch_stock_data(stock_code)
    if df is None or df.empty:
        print(f"❌ 无法获取 {stock_code} 的数据")
        return None

    if verbose:
        print(f"✅ 获取到 {len(df)} 条日线数据")
        print(f"   日期范围: {df.iloc[0]['日期']} ~ {df.iloc[-1]['日期']}")
        print("⏳ 正在计算技术指标...")

    df = calculate_indicators(df)

    # 确定信号起始位置（跳过预热期，同时确保至少有 6 个月的回测区间）
    start_idx = min(SIGNAL_START_OFFSET, max(0, len(df) - 130))
    backtest_start_date = df.iloc[start_idx]['日期']
    backtest_end_date = df.iloc[-1]['日期']

    if verbose:
        print(f"📅 回测区间: {backtest_start_date} ~ {backtest_end_date}")
        print("⏳ 正在生成交易信号...")

    # 策略 A
    sig_a = strategy_a_signals(df)
    sim_a = TradeSimulator()
    sim_a.execute_signals(df, sig_a, start_idx)
    metrics_a = sim_a.get_metrics(df, start_idx)

    # 策略 B
    sig_b = strategy_b_signals(df)
    sim_b = TradeSimulator()
    sim_b.execute_signals(df, sig_b, start_idx)
    metrics_b = sim_b.get_metrics(df, start_idx)

    result = {
        'code': stock_code,
        'name': name,
        'start_date': backtest_start_date,
        'end_date': backtest_end_date,
        'metrics_a': metrics_a,
        'metrics_b': metrics_b,
        'trades_a': sim_a.trades,
        'trades_b': sim_b.trades,
    }

    if verbose:
        print_single_report(result)

    return result


def print_single_report(result):
    """打印单只股票的回测报告"""
    name = result['name']
    code = result['code']
    ma = result['metrics_a']
    mb = result['metrics_b']

    print()
    print("=" * 70)
    print(f"📈 回测报告: {name}({code})")
    print(f"📅 区间: {result['start_date']} ~ {result['end_date']}")
    print("=" * 70)

    for label, m, trades in [
        ("策略A: MACD/KDJ 核心信号", ma, result['trades_a']),
        ("策略B: 完整评分体系(20分制)", mb, result['trades_b']),
    ]:
        if not m:
            print(f"\n--- {label} ---")
            print("  无有效数据")
            continue

        print(f"\n--- {label} ---")
        print(f"  总收益:     {m['total_return']:+.2f}%")
        print(f"  年化收益:   {m['annual_return']:+.2f}%")
        print(f"  最大回撤:   -{m['max_drawdown']:.2f}%")
        print(f"  胜率:       {m['win_rate']:.1f}% ({m['wins']}胜/{m['losses']}负)")
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "∞"
        print(f"  盈亏比:     {pf_str}")
        print(f"  交易次数:   {m['total_trades']}")
        print(f"  平均持仓:   {m['avg_holding_days']:.1f} 天")
        print(f"  期末资金:   ¥{m['final_equity']:,.0f}")
        print(f"  买入持有:   {m['buy_hold_return']:+.2f}%")

        # 交易明细（最多显示 10 条）
        if trades:
            print(f"\n  交易明细 (共{len(trades)}笔):")
            display_trades = trades[:10]
            for i, t in enumerate(display_trades, 1):
                emoji = '🟢' if t['pnl_pct'] > 0 else '🔴'
                print(f"    {emoji} #{i} 买:{t['buy_date']} ¥{t['buy_price']:.2f}"
                      f" → 卖:{t['sell_date']} ¥{t['sell_price']:.2f}"
                      f"  {t['pnl_pct']:+.2f}%"
                      f"  [{t['buy_reason']}]")
            if len(trades) > 10:
                print(f"    ... 省略 {len(trades) - 10} 笔交易")

    # 对比
    if ma and mb:
        print(f"\n--- 对比 ---")
        bh = ma['buy_hold_return']
        alpha_a = ma['total_return'] - bh
        alpha_b = mb['total_return'] - bh
        print(f"  策略A vs 买入持有: {alpha_a:+.2f}%")
        print(f"  策略B vs 买入持有: {alpha_b:+.2f}%")

        if ma['total_return'] > mb['total_return']:
            print(f"  最优策略: A (核心信号)")
        elif mb['total_return'] > ma['total_return']:
            print(f"  最优策略: B (完整评分)")
        else:
            print(f"  最优策略: 两者相当")

    print("=" * 70)


def print_multi_summary(results):
    """打印多只股票的汇总对比表"""
    print()
    print("=" * 76)
    print("📊 多股票回测汇总")
    print("=" * 76)

    header = f"{'股票':<12} | {'策略A':>8} | {'策略B':>8} | {'买入持有':>8} | {'最优':>4}"
    print(header)
    print("-" * 76)

    sum_a = []
    sum_b = []
    sum_bh = []

    for r in results:
        if r is None:
            continue
        ma = r['metrics_a']
        mb = r['metrics_b']
        if not ma or not mb:
            continue

        ra = ma['total_return']
        rb = mb['total_return']
        bh = ma['buy_hold_return']
        sum_a.append(ra)
        sum_b.append(rb)
        sum_bh.append(bh)

        best = 'A' if ra > rb else ('B' if rb > ra else '-')
        label = f"{r['name']}"
        print(f"{label:<12} | {ra:>+7.2f}% | {rb:>+7.2f}% | {bh:>+7.2f}% | {best:>4}")

    if sum_a:
        print("-" * 76)
        avg_a = np.mean(sum_a)
        avg_b = np.mean(sum_b)
        avg_bh = np.mean(sum_bh)
        best_avg = 'A' if avg_a > avg_b else ('B' if avg_b > avg_a else '-')
        print(f"{'平均':<12} | {avg_a:>+7.2f}% | {avg_b:>+7.2f}% | {avg_bh:>+7.2f}% | {best_avg:>4}")

    print("=" * 76)

    # 策略说明
    print("\n策略说明:")
    print("  A = MACD(8,17,9)/KDJ(6,3,3) 核心金叉死叉信号")
    print("  B = 完整评分体系(20分制，MACD+KDJ占70%权重)")
    print(f"  初始资金: ¥{INITIAL_CAPITAL:,.0f} | 佣金: {COMMISSION_RATE*10000:.1f}‱ | 印花税: {STAMP_TAX_RATE*10000:.1f}‱")
    print(f"  交易规则: 次日开盘价执行，满仓买入/满仓卖出，100股整数倍")


# ============================================================
# CLI 入口
# ============================================================

def main():
    args = sys.argv[1:]

    if not args:
        print("使用方法:")
        print("  python3 backtest_strategy.py 600276              # 单只股票")
        print("  python3 backtest_strategy.py --multi              # 预设4只代表性股票")
        print("  python3 backtest_strategy.py 600519 002594 600276 # 多只自定义股票")
        sys.exit(1)

    if args[0] == '--multi':
        codes = list(PRESET_STOCKS.keys())
    else:
        codes = args

    if len(codes) == 1:
        # 单只股票模式：详细报告
        run_backtest(codes[0], verbose=True)
    else:
        # 多只股票模式：逐个回测 + 汇总
        results = []
        for code in codes:
            r = run_backtest(code, verbose=True)
            results.append(r)
        print_multi_summary(results)


if __name__ == "__main__":
    main()
