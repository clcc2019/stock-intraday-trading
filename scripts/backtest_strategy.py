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
import argparse

warnings.filterwarnings('ignore')

# 导入统一数据源和公共技术指标
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from data_source import DataSource
    DATA_SOURCE_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    DataSource = None
    DATA_SOURCE_IMPORT_ERROR = exc
from technical import calculate_ma, calculate_macd, calculate_kdj, calculate_rsi, calculate_volume_ma

# ============================================================
# 常量
# ============================================================
INITIAL_CAPITAL = 100_000.0   # 初始资金 10 万
COMMISSION_RATE = 0.00025     # 含规费佣金 万2.5（买卖都收，通常已含经手费/证管费）
STAMP_TAX_RATE = 0.0005       # 印花税 万5（仅卖出收）
TRANSFER_FEE_RATE = 0.00001   # A股过户费 0.01‰（买卖都收）
SLIPPAGE_RATE = 0.0005        # 默认滑点 5bp，避免回测过度乐观
BACKTEST_DAYS = 900           # 默认约 3 年自然日数据（含指标预热）
SIGNAL_START_OFFSET = 60      # 前 60 天用于指标预热，不产生信号
STOP_LOSS_PCT = -3.0          # 止损线：-3%
TRAILING_ACTIVATE_PCT = 2.0   # 移动止盈激活线：+2%
TRAILING_STOP_PCT = 1.0       # 移动止盈保底线：+1%（回撤到此平仓）
MAX_HOLDING_DAYS = 20         # 最大持仓天数
MIN_PROFIT_TRADES = 6         # 盈利闸门要求的最低交易样本数
TREND_STOP_LOSS_PCT = -15.0
TREND_TRAILING_ACTIVATE_PCT = 100.0
TREND_TRAILING_STOP_PCT = 50.0
TREND_MAX_HOLDING_DAYS = 260

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
    if DataSource is None:
        raise RuntimeError(
            "缺少行情依赖，无法获取真实K线。请先安装 requirements.txt "
            "（建议: /opt/homebrew/bin/python3.12 -m pip install -r requirements.txt）。"
        ) from DATA_SOURCE_IMPORT_ERROR
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
    calculate_ma(df, windows=[5, 10, 20, 60, 120])
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
# 策略 C：MA60 中线趋势持有（顺大势，少做短线噪音）
# ============================================================

def strategy_c_signals(df):
    """
    Strategy C: MA60 中线趋势持有

    设计目的：
    - 修正 A/B 过度依赖 MACD/KDJ 导致的频繁卖飞和止损噪音。
    - 只在价格站上 MA60 且 MA60 斜率为正时参与。
    - 跌破 MA60 两日确认后退出，尽量保留强趋势。
    """
    signals = pd.DataFrame(index=df.index, columns=['signal', 'reason'])
    signals['signal'] = None
    signals['reason'] = ''

    for i in range(80, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i - 1]
        price = cur['收盘']
        ma60 = cur.get('MA60', np.nan)
        prev_ma60 = prev.get('MA60', np.nan)

        if np.isnan(ma60) or np.isnan(prev_ma60):
            continue

        ma60_prev20 = df.iloc[i - 20].get('MA60', np.nan) if i >= 20 else np.nan
        ma60_slope = 0
        if not np.isnan(ma60_prev20) and ma60_prev20 > 0:
            ma60_slope = (ma60 - ma60_prev20) / ma60_prev20 * 100

        dev_ma60 = (price - ma60) / ma60 * 100 if ma60 > 0 else 0

        enter = (
            price > ma60 and
            ma60_slope > 0 and
            dev_ma60 < 35
        )
        exit_trend = (
            price < ma60 and
            prev['收盘'] < prev_ma60
        )

        if enter:
            signals.iloc[i, signals.columns.get_loc('signal')] = 'buy'
            signals.iloc[i, signals.columns.get_loc('reason')] = f'MA60趋势持有(MA60斜率{ma60_slope:+.1f}%, 偏离{dev_ma60:+.1f}%)'
        elif exit_trend:
            signals.iloc[i, signals.columns.get_loc('signal')] = 'sell'
            signals.iloc[i, signals.columns.get_loc('reason')] = '连续跌破MA60，趋势破坏'

    return signals


# ============================================================
# 交易模拟引擎
# ============================================================

class TradeSimulator:
    """模拟交易执行器"""

    def __init__(
        self,
        initial_capital=INITIAL_CAPITAL,
        commission_rate=COMMISSION_RATE,
        stamp_tax_rate=STAMP_TAX_RATE,
        transfer_fee_rate=TRANSFER_FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
        position_pct=1.0,
        stop_loss_pct=STOP_LOSS_PCT,
        trailing_activate_pct=TRAILING_ACTIVATE_PCT,
        trailing_stop_pct=TRAILING_STOP_PCT,
        max_holding_days=MAX_HOLDING_DAYS,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.transfer_fee_rate = transfer_fee_rate
        self.slippage_rate = slippage_rate
        self.position_pct = max(0.0, min(1.0, position_pct))
        self.stop_loss_pct = stop_loss_pct
        self.trailing_activate_pct = trailing_activate_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_holding_days = max_holding_days
        self.trades = []        # 已完成的完整交易 (买+卖)
        self.pending_buy = None  # 未平仓的买入记录
        self.equity_curve = []   # (date, equity)
        self.exposure_curve = [] # 每个权益记录点是否持仓

    def _buy_cost(self, price, shares):
        return price * shares * (self.commission_rate + self.transfer_fee_rate)

    def _sell_cost(self, price, shares):
        return price * shares * (self.commission_rate + self.stamp_tax_rate + self.transfer_fee_rate)

    def _buy_price(self, raw_price):
        return raw_price * (1 + self.slippage_rate)

    def _sell_price(self, raw_price):
        return raw_price * (1 - self.slippage_rate)

    def _append_equity(self, date, equity):
        self.equity_curve.append((date, equity))
        self.exposure_curve.append(self.shares > 0)

    def _close_position(self, raw_sell_price, sell_date, sell_reason):
        """平仓辅助函数"""
        sell_price = self._sell_price(raw_sell_price)
        shares = self.shares
        proceeds = sell_price * self.shares
        sell_cost = self._sell_cost(sell_price, self.shares)
        self.cash += proceeds - sell_cost

        buy_price = self.pending_buy['buy_price']
        buy_cost = self.pending_buy['buy_cost']
        entry_value = buy_price * shares
        gross_pnl = proceeds - entry_value
        net_pnl = proceeds - sell_cost - entry_value - buy_cost
        invested = entry_value + buy_cost
        pnl = net_pnl / invested * 100 if invested > 0 else 0.0
        gross_pnl_pct = gross_pnl / entry_value * 100 if entry_value > 0 else 0.0

        self.trades.append({
            'buy_date': self.pending_buy['buy_date'],
            'buy_price': buy_price,
            'sell_date': sell_date,
            'sell_price': sell_price,
            'shares': shares,
            'pnl_pct': pnl,
            'gross_pnl_pct': gross_pnl_pct,
            'net_pnl': net_pnl,
            'cost': buy_cost + sell_cost,
            'buy_reason': self.pending_buy['reason'],
            'sell_reason': sell_reason,
        })
        self.shares = 0
        self.pending_buy = None

    def execute_signals(self, df, signals, start_idx, end_idx=None):
        """
        按信号执行交易。信号在 day i 产生，在 day i+1 的开盘价执行。
        包含止损、移动止盈、最大持仓天数等风控机制。
        start_idx: 信号开始有效的位置（跳过预热期）
        end_idx: 最后一个信号位置（不含），用于样本内/样本外分段验证
        """
        self._peak_price = 0  # 持仓期间最高价（用于移动止盈）
        self._holding_days = 0  # 持仓天数
        if end_idx is None:
            end_idx = len(df) - 1
        end_idx = min(end_idx, len(df) - 1)

        for i in range(start_idx, end_idx):
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
                stop_loss_price = buy_price * (1 + self.stop_loss_pct / 100)
                if day_low <= stop_loss_price:
                    # 缺口低开时不能假设能按止损价成交，按开盘价更保守。
                    sell_at = exec_price if exec_price < stop_loss_price else stop_loss_price
                    self._close_position(sell_at, exec_date, f'止损{self.stop_loss_pct}%')
                    self._peak_price = 0
                    self._holding_days = 0
                    equity = self.cash + self.shares * day_close
                    self._append_equity(exec_date, equity)
                    continue

                # 2) 移动止盈：曾涨+2%后回落到仅+1%
                pnl_from_peak = (self._peak_price - buy_price) / buy_price * 100
                if pnl_from_peak >= self.trailing_activate_pct:
                    trailing_price = buy_price * (1 + self.trailing_stop_pct / 100)
                    if day_low <= trailing_price:
                        sell_at = exec_price if exec_price < trailing_price else trailing_price
                        self._close_position(sell_at, exec_date, f'移动止盈(峰值+{pnl_from_peak:.1f}%)')
                        self._peak_price = 0
                        self._holding_days = 0
                        equity = self.cash + self.shares * day_close
                        self._append_equity(exec_date, equity)
                        continue

                # 3) 最大持仓天数
                if self._holding_days >= self.max_holding_days:
                    self._close_position(exec_price, exec_date, f'超时{self.max_holding_days}天')
                    self._peak_price = 0
                    self._holding_days = 0
                    equity = self.cash + self.shares * day_close
                    self._append_equity(exec_date, equity)
                    continue

            # ── 信号处理 ──
            sig = signals.iloc[i]['signal']
            reason = signals.iloc[i]['reason']

            if sig == 'buy' and self.shares == 0:
                buy_price = self._buy_price(exec_price)
                target_cash = self.cash * self.position_pct
                buy_fee_rate = self.commission_rate + self.transfer_fee_rate
                max_shares = int(target_cash / (buy_price * (1 + buy_fee_rate)))
                max_shares = (max_shares // 100) * 100
                if max_shares <= 0:
                    equity = self.cash
                    self._append_equity(exec_date, equity)
                    continue
                cost = self._buy_cost(buy_price, max_shares)
                self.cash -= buy_price * max_shares + cost
                self.shares = max_shares
                self.pending_buy = {
                    'buy_date': exec_date,
                    'buy_price': buy_price,
                    'buy_cost': cost,
                    'shares': max_shares,
                    'reason': reason,
                }
                self._peak_price = max(day_high, buy_price)
                self._holding_days = 0

            elif sig == 'sell' and self.shares > 0 and self.pending_buy is not None:
                self._close_position(exec_price, exec_date, reason)
                self._peak_price = 0
                self._holding_days = 0

            # 记录每日权益
            equity = self.cash + self.shares * day_close
            self._append_equity(exec_date, equity)

        # 补上最后一天的权益
        if len(df) > 0:
            last_pos = min(end_idx, len(df) - 1)
            last = df.iloc[last_pos]
            equity = self.cash + self.shares * last['收盘']
            if not self.equity_curve or self.equity_curve[-1][0] != last['日期']:
                self._append_equity(last['日期'], equity)

    def get_metrics(self, df, start_idx, end_idx=None):
        """计算回测绩效指标"""
        if not self.equity_curve:
            return {}
        if end_idx is None:
            end_idx = len(df) - 1
        end_idx = min(end_idx, len(df) - 1)

        equities = [e for _, e in self.equity_curve]
        final_equity = equities[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100

        # 年化收益
        first_date = pd.to_datetime(df.iloc[start_idx]['日期'])
        last_date = pd.to_datetime(df.iloc[end_idx]['日期'])
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

        # 胜率 / 盈亏比（用扣费后净收益计算）
        wins = [t for t in self.trades if t['pnl_pct'] > 0]
        losses = [t for t in self.trades if t['pnl_pct'] <= 0]
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        total_profit = sum(t['net_pnl'] for t in wins) if wins else 0
        total_loss = abs(sum(t['net_pnl'] for t in losses)) if losses else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        trade_pcts = [t['pnl_pct'] for t in self.trades]
        avg_trade_pct = float(np.mean(trade_pcts)) if trade_pcts else 0.0
        avg_win_pct = float(np.mean([t['pnl_pct'] for t in wins])) if wins else 0.0
        avg_loss_pct = float(np.mean([t['pnl_pct'] for t in losses])) if losses else 0.0

        # 平均持仓天数
        holding_days = []
        for t in self.trades:
            d1 = pd.to_datetime(t['buy_date'])
            d2 = pd.to_datetime(t['sell_date'])
            holding_days.append((d2 - d1).days)
        avg_holding = np.mean(holding_days) if holding_days else 0

        # 买入持有收益
        start_price = df.iloc[start_idx]['收盘']
        end_price = df.iloc[end_idx]['收盘']
        buy_hold_return = (end_price / start_price - 1) * 100
        alpha_vs_buy_hold = total_return - buy_hold_return

        # 日度波动与夏普（无风险利率按0处理，只用于横向比较）
        equity_series = pd.Series(equities, dtype=float)
        daily_returns = equity_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        exposure_pct = np.mean(self.exposure_curve) * 100 if self.exposure_curve else 0.0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'wins': len(wins),
            'losses': len(losses),
            'total_trades': len(self.trades),
            'profit_factor': profit_factor,
            'avg_trade_pct': avg_trade_pct,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'avg_holding_days': avg_holding,
            'buy_hold_return': buy_hold_return,
            'alpha_vs_buy_hold': alpha_vs_buy_hold,
            'sharpe': sharpe,
            'exposure_pct': exposure_pct,
            'final_equity': final_equity,
        }


# ============================================================
# 回测执行与报告
# ============================================================

def evaluate_profit_gate(
    metrics,
    min_trades=MIN_PROFIT_TRADES,
    min_annual_return=5.0,
    max_drawdown=15.0,
    min_profit_factor=1.2,
    min_alpha=0.0,
):
    """
    赚钱前置闸门：不是预测未来，而是要求历史样本中至少证明正期望。
    未通过时脚本仍会输出数据，但不能把该策略当作可实盘策略。
    """
    if not metrics:
        return {'passed': False, 'failed': ['无有效回测指标']}

    checks = [
        (metrics['total_return'] > 0, f"总收益需>0%，当前{metrics['total_return']:+.2f}%"),
        (metrics['annual_return'] >= min_annual_return,
         f"年化收益需>={min_annual_return:.1f}%，当前{metrics['annual_return']:+.2f}%"),
        (metrics['max_drawdown'] <= max_drawdown,
         f"最大回撤需<={max_drawdown:.1f}%，当前{metrics['max_drawdown']:.2f}%"),
        (metrics['total_trades'] >= min_trades,
         f"交易样本需>={min_trades}笔，当前{metrics['total_trades']}笔"),
        (metrics['profit_factor'] >= min_profit_factor,
         f"盈亏比需>={min_profit_factor:.2f}，当前{metrics['profit_factor']:.2f}"),
        (metrics['avg_trade_pct'] > 0,
         f"单笔期望需>0%，当前{metrics['avg_trade_pct']:+.2f}%"),
        (metrics['alpha_vs_buy_hold'] >= min_alpha,
         f"相对买入持有超额需>={min_alpha:+.1f}%，当前{metrics['alpha_vs_buy_hold']:+.2f}%"),
    ]

    failed = [message for ok, message in checks if not ok]
    return {'passed': len(failed) == 0, 'failed': failed}


def _simulate_strategy(df, signals, start_idx, simulator_kwargs=None, end_idx=None):
    sim = TradeSimulator(**(simulator_kwargs or {}))
    sim.execute_signals(df, signals, start_idx, end_idx=end_idx)
    metrics = sim.get_metrics(df, start_idx, end_idx=end_idx)
    return metrics, sim.trades


def _trend_simulator_kwargs(base_kwargs=None):
    kwargs = dict(base_kwargs or {})
    kwargs.setdefault('stop_loss_pct', TREND_STOP_LOSS_PCT)
    kwargs.setdefault('trailing_activate_pct', TREND_TRAILING_ACTIVATE_PCT)
    kwargs.setdefault('trailing_stop_pct', TREND_TRAILING_STOP_PCT)
    kwargs.setdefault('max_holding_days', TREND_MAX_HOLDING_DAYS)
    return kwargs


def _get_oos_split_idx(df, start_idx, oos_ratio):
    """返回样本外起点信号位置；样本过短时返回 None。"""
    if oos_ratio <= 0 or oos_ratio >= 0.8:
        return None
    signal_end_idx = len(df) - 1
    backtest_bars = signal_end_idx - start_idx
    if backtest_bars < 120:
        return None
    split_idx = int(signal_end_idx - backtest_bars * oos_ratio)
    split_idx = max(start_idx + 60, split_idx)
    if signal_end_idx - split_idx < 40:
        return None
    return split_idx


def run_backtest(
    stock_code,
    verbose=True,
    days=BACKTEST_DAYS,
    simulator_kwargs=None,
    gate_kwargs=None,
    oos_ratio=0.33,
):
    """对单只股票运行回测"""
    name = get_stock_name(stock_code)
    if verbose:
        print(f"\n📊 正在获取 {name}({stock_code}) 的历史数据...")

    try:
        df = fetch_stock_data(stock_code, days=days)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return None
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
    metrics_a, trades_a = _simulate_strategy(df, sig_a, start_idx, simulator_kwargs)

    # 策略 B
    sig_b = strategy_b_signals(df)
    metrics_b, trades_b = _simulate_strategy(df, sig_b, start_idx, simulator_kwargs)

    # 策略 C
    sig_c = strategy_c_signals(df)
    trend_kwargs = _trend_simulator_kwargs(simulator_kwargs)
    metrics_c, trades_c = _simulate_strategy(df, sig_c, start_idx, trend_kwargs)

    gate_kwargs = gate_kwargs or {}
    gate_a = evaluate_profit_gate(metrics_a, **gate_kwargs)
    gate_b = evaluate_profit_gate(metrics_b, **gate_kwargs)
    gate_c = evaluate_profit_gate(metrics_c, **gate_kwargs)

    oos = {}
    split_idx = _get_oos_split_idx(df, start_idx, oos_ratio)
    if split_idx is not None:
        metrics_a_is, _ = _simulate_strategy(df, sig_a, start_idx, simulator_kwargs, end_idx=split_idx)
        metrics_b_is, _ = _simulate_strategy(df, sig_b, start_idx, simulator_kwargs, end_idx=split_idx)
        metrics_c_is, _ = _simulate_strategy(df, sig_c, start_idx, trend_kwargs, end_idx=split_idx)
        metrics_a_oos, trades_a_oos = _simulate_strategy(df, sig_a, split_idx, simulator_kwargs)
        metrics_b_oos, trades_b_oos = _simulate_strategy(df, sig_b, split_idx, simulator_kwargs)
        metrics_c_oos, trades_c_oos = _simulate_strategy(df, sig_c, split_idx, trend_kwargs)
        oos = {
            'split_date': df.iloc[split_idx]['日期'],
            'metrics_a_is': metrics_a_is,
            'metrics_b_is': metrics_b_is,
            'metrics_c_is': metrics_c_is,
            'metrics_a_oos': metrics_a_oos,
            'metrics_b_oos': metrics_b_oos,
            'metrics_c_oos': metrics_c_oos,
            'trades_a_oos': trades_a_oos,
            'trades_b_oos': trades_b_oos,
            'trades_c_oos': trades_c_oos,
            'gate_a_oos': evaluate_profit_gate(metrics_a_oos, **gate_kwargs),
            'gate_b_oos': evaluate_profit_gate(metrics_b_oos, **gate_kwargs),
            'gate_c_oos': evaluate_profit_gate(metrics_c_oos, **gate_kwargs),
        }

    result = {
        'code': stock_code,
        'name': name,
        'start_date': backtest_start_date,
        'end_date': backtest_end_date,
        'days': days,
        'metrics_a': metrics_a,
        'metrics_b': metrics_b,
        'metrics_c': metrics_c,
        'gate_a': gate_a,
        'gate_b': gate_b,
        'gate_c': gate_c,
        'trades_a': trades_a,
        'trades_b': trades_b,
        'trades_c': trades_c,
        'oos': oos,
        'simulator_kwargs': simulator_kwargs or {},
        'trend_simulator_kwargs': trend_kwargs,
        'gate_kwargs': gate_kwargs,
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
    mc = result.get('metrics_c')

    print()
    print("=" * 70)
    print(f"📈 回测报告: {name}({code})")
    print(f"📅 区间: {result['start_date']} ~ {result['end_date']}")
    print("=" * 70)

    cost = result.get('simulator_kwargs', {})
    commission = cost.get('commission_rate', COMMISSION_RATE)
    stamp_tax = cost.get('stamp_tax_rate', STAMP_TAX_RATE)
    transfer_fee = cost.get('transfer_fee_rate', TRANSFER_FEE_RATE)
    slippage = cost.get('slippage_rate', SLIPPAGE_RATE)
    position_pct = cost.get('position_pct', 1.0)
    print(f"💰 成本假设: 佣金{commission*10000:.2f}‱ | 印花税{stamp_tax*10000:.2f}‱ | "
          f"过户费{transfer_fee*10000:.2f}‱ | 滑点{slippage*10000:.2f}bp | 单票仓位{position_pct*100:.0f}%")
    trend = result.get('trend_simulator_kwargs', {})
    print(f"🧭 策略C风控: 止损{trend.get('stop_loss_pct', TREND_STOP_LOSS_PCT):+.0f}% | "
          f"最长持仓{trend.get('max_holding_days', TREND_MAX_HOLDING_DAYS)}天")

    for label, m, trades, gate in [
        ("策略A: MACD/KDJ 核心信号", ma, result['trades_a'], result.get('gate_a')),
        ("策略B: 完整评分体系(20分制)", mb, result['trades_b'], result.get('gate_b')),
        ("策略C: MA60 中线趋势持有", mc, result.get('trades_c', []), result.get('gate_c')),
    ]:
        if not m:
            print(f"\n--- {label} ---")
            print("  无有效数据")
            continue

        print(f"\n--- {label} ---")
        print(f"  总收益:     {m['total_return']:+.2f}%")
        print(f"  年化收益:   {m['annual_return']:+.2f}%")
        print(f"  最大回撤:   -{m['max_drawdown']:.2f}%")
        print(f"  买入持有:   {m['buy_hold_return']:+.2f}%")
        print(f"  超额收益:   {m['alpha_vs_buy_hold']:+.2f}%")
        print(f"  胜率:       {m['win_rate']:.1f}% ({m['wins']}胜/{m['losses']}负)")
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "∞"
        print(f"  盈亏比:     {pf_str}")
        print(f"  单笔期望:   {m['avg_trade_pct']:+.2f}%")
        print(f"  夏普(粗略): {m['sharpe']:+.2f}")
        print(f"  交易次数:   {m['total_trades']}")
        print(f"  平均持仓:   {m['avg_holding_days']:.1f} 天")
        print(f"  持仓暴露:   {m['exposure_pct']:.1f}%")
        print(f"  期末资金:   ¥{m['final_equity']:,.0f}")

        if gate:
            status = "通过" if gate['passed'] else "未通过"
            print(f"  赚钱闸门:   {status}")
            if not gate['passed']:
                for reason in gate['failed'][:4]:
                    print(f"              - {reason}")

        # 交易明细（最多显示 10 条）
        if trades:
            print(f"\n  交易明细 (共{len(trades)}笔):")
            display_trades = trades[:10]
            for i, t in enumerate(display_trades, 1):
                emoji = '🟢' if t['pnl_pct'] > 0 else '🔴'
                print(f"    {emoji} #{i} 买:{t['buy_date']} ¥{t['buy_price']:.2f}"
                      f" → 卖:{t['sell_date']} ¥{t['sell_price']:.2f}"
                      f"  {t['pnl_pct']:+.2f}%"
                      f"  成本¥{t.get('cost', 0):.0f}"
                      f"  [{t['buy_reason']}]")
            if len(trades) > 10:
                print(f"    ... 省略 {len(trades) - 10} 笔交易")

    # 对比
    if ma and mb and mc:
        print(f"\n--- 对比 ---")
        rows = [
            ('A', '核心信号', ma),
            ('B', '完整评分', mb),
            ('C', 'MA60趋势持有', mc),
        ]
        for key, _, metrics in rows:
            print(f"  策略{key} vs 买入持有: {metrics['alpha_vs_buy_hold']:+.2f}%")
        best = max(rows, key=lambda x: x[2]['total_return'])
        print(f"  最优策略: {best[0]} ({best[1]})")

    oos = result.get('oos') or {}
    if oos:
        print(f"\n--- 样本外验证 ---")
        print(f"  切分日期: {oos['split_date']}（之前为样本内，之后为样本外）")
        for label, m, gate in [
            ("策略A", oos.get('metrics_a_oos'), oos.get('gate_a_oos')),
            ("策略B", oos.get('metrics_b_oos'), oos.get('gate_b_oos')),
            ("策略C", oos.get('metrics_c_oos'), oos.get('gate_c_oos')),
        ]:
            if not m:
                continue
            status = "通过" if gate and gate['passed'] else "未通过"
            print(f"  {label}: 总收益{m['total_return']:+.2f}% | 年化{m['annual_return']:+.2f}% | "
                  f"回撤-{m['max_drawdown']:.2f}% | 超额{m['alpha_vs_buy_hold']:+.2f}% | "
                  f"交易{m['total_trades']}笔 | 闸门{status}")

    print("=" * 70)


def print_multi_summary(results):
    """打印多只股票的汇总对比表"""
    print()
    print("=" * 100)
    print("📊 多股票回测汇总")
    print("=" * 100)

    header = (f"{'股票':<12} | {'策略A':>8} | {'A闸门':>4} | "
              f"{'策略B':>8} | {'B闸门':>4} | {'策略C':>8} | {'C闸门':>4} | "
              f"{'买入持有':>8} | {'最优':>4}")
    print(header)
    print("-" * 100)

    sum_a = []
    sum_b = []
    sum_c = []
    sum_bh = []
    pass_a = 0
    pass_b = 0
    pass_c = 0
    pass_a_oos = 0
    pass_b_oos = 0
    pass_c_oos = 0
    oos_count = 0

    for r in results:
        if r is None:
            continue
        ma = r['metrics_a']
        mb = r['metrics_b']
        mc = r.get('metrics_c')
        if not ma or not mb or not mc:
            continue

        ra = ma['total_return']
        rb = mb['total_return']
        rc = mc['total_return']
        bh = ma['buy_hold_return']
        sum_a.append(ra)
        sum_b.append(rb)
        sum_c.append(rc)
        sum_bh.append(bh)
        gate_a = r.get('gate_a', {}).get('passed', False)
        gate_b = r.get('gate_b', {}).get('passed', False)
        gate_c = r.get('gate_c', {}).get('passed', False)
        pass_a += 1 if gate_a else 0
        pass_b += 1 if gate_b else 0
        pass_c += 1 if gate_c else 0
        oos = r.get('oos') or {}
        if oos:
            oos_count += 1
            pass_a_oos += 1 if oos.get('gate_a_oos', {}).get('passed', False) else 0
            pass_b_oos += 1 if oos.get('gate_b_oos', {}).get('passed', False) else 0
            pass_c_oos += 1 if oos.get('gate_c_oos', {}).get('passed', False) else 0

        best = max([('A', ra), ('B', rb), ('C', rc)], key=lambda x: x[1])[0]
        label = f"{r['name']}"
        print(f"{label:<12} | {ra:>+7.2f}% | {'过' if gate_a else '否':>4} | "
              f"{rb:>+7.2f}% | {'过' if gate_b else '否':>4} | "
              f"{rc:>+7.2f}% | {'过' if gate_c else '否':>4} | {bh:>+7.2f}% | {best:>4}")

    if sum_a:
        print("-" * 100)
        avg_a = np.mean(sum_a)
        avg_b = np.mean(sum_b)
        avg_c = np.mean(sum_c)
        avg_bh = np.mean(sum_bh)
        best_avg = max([('A', avg_a), ('B', avg_b), ('C', avg_c)], key=lambda x: x[1])[0]
        print(f"{'平均':<12} | {avg_a:>+7.2f}% | {pass_a}/{len(sum_a):<2} | "
              f"{avg_b:>+7.2f}% | {pass_b}/{len(sum_b):<2} | "
              f"{avg_c:>+7.2f}% | {pass_c}/{len(sum_c):<2} | {avg_bh:>+7.2f}% | {best_avg:>4}")
        if oos_count:
            print(f"样本外闸门通过: 策略A {pass_a_oos}/{oos_count} | "
                  f"策略B {pass_b_oos}/{oos_count} | 策略C {pass_c_oos}/{oos_count}")

    print("=" * 100)

    # 策略说明
    print("\n策略说明:")
    print("  A = MACD(8,17,9)/KDJ(6,3,3) 核心金叉死叉信号")
    print("  B = 完整评分体系(20分制，MACD+KDJ占70%权重)")
    print("  C = MA60 中线趋势持有（宽止损、长持仓，用于减少短线噪音）")
    print(f"  交易规则: 次日开盘价执行，按设置仓位买卖，100股整数倍")
    print(f"  闸门含义: 总收益/年化/回撤/交易样本/盈亏比/单笔期望/相对买入持有同时达标才算通过")


def run_self_test():
    """离线自检：不依赖外部行情源，验证交易成本、滑点、风控和闸门逻辑。"""
    rows = []
    start = pd.Timestamp('2025-01-01')
    for i in range(80):
        date = (start + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        base = 10 + i * 0.08
        rows.append({
            '日期': date,
            '开盘': base,
            '最高': base * 1.03,
            '最低': base * 0.99,
            '收盘': base * 1.015,
            '成交量': 1000000 + i * 1000,
        })
    df = pd.DataFrame(rows)
    signals = pd.DataFrame(index=df.index, columns=['signal', 'reason'])
    signals['signal'] = None
    signals['reason'] = ''
    for idx in [5, 16, 27, 38, 49, 60]:
        signals.iloc[idx, signals.columns.get_loc('signal')] = 'buy'
        signals.iloc[idx, signals.columns.get_loc('reason')] = 'self-test buy'
        signals.iloc[idx + 4, signals.columns.get_loc('signal')] = 'sell'
        signals.iloc[idx + 4, signals.columns.get_loc('reason')] = 'self-test sell'

    sim = TradeSimulator(slippage_rate=SLIPPAGE_RATE, position_pct=1.0)
    sim.execute_signals(df, signals, start_idx=0)
    metrics = sim.get_metrics(df, start_idx=0)
    gate = evaluate_profit_gate(metrics, min_trades=6, min_annual_return=1.0,
                                max_drawdown=20.0, min_profit_factor=1.0, min_alpha=-100.0)

    assert metrics['total_trades'] == 6, f"expected 6 trades, got {metrics['total_trades']}"
    assert all(t['cost'] > 0 for t in sim.trades), "all trades should include transaction costs"
    assert metrics['total_return'] > 0, "synthetic rising market should be profitable"
    assert gate['passed'], f"self-test gate should pass: {gate['failed']}"

    print("✅ self-test passed")
    print(f"   trades={metrics['total_trades']} return={metrics['total_return']:+.2f}% "
          f"pf={metrics['profit_factor']:.2f} avg_trade={metrics['avg_trade_pct']:+.2f}%")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='MACD/KDJ 与评分策略回测。默认扣佣金、印花税、过户费与滑点，并输出赚钱闸门。',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('codes', nargs='*', help='股票代码，如 600519 002594')
    parser.add_argument('--multi', action='store_true', help='使用预设4只代表股票')
    parser.add_argument('--days', type=int, default=BACKTEST_DAYS, help='获取最近N个自然日数据，含指标预热')
    parser.add_argument('--initial-capital', type=float, default=INITIAL_CAPITAL, help='初始资金')
    parser.add_argument('--commission', type=float, default=COMMISSION_RATE, help='佣金率，小数格式，如0.00025')
    parser.add_argument('--stamp-tax', type=float, default=STAMP_TAX_RATE, help='卖出印花税率，小数格式')
    parser.add_argument('--transfer-fee', type=float, default=TRANSFER_FEE_RATE, help='买卖过户费率，小数格式')
    parser.add_argument('--slippage', type=float, default=SLIPPAGE_RATE, help='单边滑点率，小数格式，如0.0005=5bp')
    parser.add_argument('--position-pct', type=float, default=1.0, help='单次买入使用现金比例，1=满仓')
    parser.add_argument('--min-trades', type=int, default=MIN_PROFIT_TRADES, help='赚钱闸门最低交易笔数')
    parser.add_argument('--min-annual-return', type=float, default=5.0, help='赚钱闸门最低年化收益率')
    parser.add_argument('--max-drawdown', type=float, default=15.0, help='赚钱闸门最大回撤上限')
    parser.add_argument('--min-profit-factor', type=float, default=1.2, help='赚钱闸门最低盈亏比')
    parser.add_argument('--min-alpha', type=float, default=0.0, help='赚钱闸门最低相对买入持有超额收益')
    parser.add_argument('--self-test', action='store_true', help='运行离线自检，不访问行情源')
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.codes and not args.multi:
        parser.print_help()
        sys.exit(1)

    if args.multi:
        codes = list(PRESET_STOCKS.keys())
    else:
        codes = args.codes

    simulator_kwargs = {
        'initial_capital': args.initial_capital,
        'commission_rate': args.commission,
        'stamp_tax_rate': args.stamp_tax,
        'transfer_fee_rate': args.transfer_fee,
        'slippage_rate': args.slippage,
        'position_pct': args.position_pct,
    }
    gate_kwargs = {
        'min_trades': args.min_trades,
        'min_annual_return': args.min_annual_return,
        'max_drawdown': args.max_drawdown,
        'min_profit_factor': args.min_profit_factor,
        'min_alpha': args.min_alpha,
    }

    if len(codes) == 1:
        # 单只股票模式：详细报告
        run_backtest(codes[0], verbose=True, days=args.days,
                     simulator_kwargs=simulator_kwargs, gate_kwargs=gate_kwargs)
    else:
        # 多只股票模式：逐个回测 + 汇总
        results = []
        for code in codes:
            r = run_backtest(code, verbose=True, days=args.days,
                             simulator_kwargs=simulator_kwargs, gate_kwargs=gate_kwargs)
            results.append(r)
        print_multi_summary(results)


if __name__ == "__main__":
    main()
