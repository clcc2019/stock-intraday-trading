#!/usr/bin/env python3
"""
策略 C 参数扫描工具

目标不是找到“神奇参数”，而是把 MA60 中线趋势持有策略的下一轮优化
变成可复现的研究流程：同一组成本、同一组样本外切分、同一组赚钱闸门。

示例：
  python3 scripts/optimize_strategy_c.py --multi
  python3 scripts/optimize_strategy_c.py 600519 002594 600276 300750 --top 10
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_strategy as bt
from technical import calculate_ma


INDEX_MAP = {
    'hs300': ('沪深300', ['sh.000300']),
    'sz50': ('上证50', ['sh.000016']),
    'core': ('核心指数(沪深300+上证50)', ['sh.000300', 'sh.000016']),
}


PARAM_GRID = [
    # ma_window, slope_lookback, min_slope, max_dev, entry_confirm_days,
    # entry_buffer, exit_confirm_days, stop_loss, momentum_lookback, min_momentum
    (60, 20, 0.0, 25, 2, 1.0, 2, -12.0, 60, 0.0),
    (60, 20, 0.0, 25, 2, 1.0, 2, -12.0, 60, 10.0),
    (60, 20, 0.0, 25, 2, 1.0, 2, -12.0, 60, 20.0),
    (60, 20, 0.0, 35, 2, 1.0, 2, -15.0, 60, 0.0),
    (60, 20, 0.0, 35, 2, 1.0, 2, -15.0, 60, 10.0),
    (60, 20, 0.2, 25, 2, 1.0, 2, -12.0, 60, 0.0),
    (60, 20, 0.2, 25, 2, 1.0, 2, -12.0, 60, 10.0),
    (60, 30, 0.0, 25, 3, 1.0, 3, -12.0, 60, 0.0),
    (60, 30, 0.0, 25, 3, 1.0, 3, -12.0, 60, 10.0),
    (60, 30, 0.2, 25, 3, 1.0, 3, -12.0, 60, 10.0),
    (120, 30, 0.0, 25, 2, 0.0, 2, -12.0, 120, 0.0),
    (120, 40, 0.0, 25, 3, 1.0, 3, -12.0, 120, 10.0),
]


def make_trend_signals(
    df,
    ma_window,
    slope_lookback,
    min_slope,
    max_dev,
    entry_confirm_days,
    entry_buffer,
    exit_confirm_days,
    momentum_lookback,
    min_momentum,
):
    """生成可参数化趋势持有信号。"""
    calculate_ma(df, windows=sorted({5, 10, 20, 60, 120, ma_window}))

    ma_col = f'MA{ma_window}'
    signals = pd.DataFrame(index=df.index, columns=['signal', 'reason'])
    signals['signal'] = None
    signals['reason'] = ''

    start = max(ma_window + slope_lookback, momentum_lookback, 80)
    for i in range(start, len(df)):
        cur = df.iloc[i]
        ma = cur.get(ma_col, np.nan)
        if np.isnan(ma) or ma <= 0:
            continue

        prev_ma = df.iloc[i - slope_lookback].get(ma_col, np.nan)
        if np.isnan(prev_ma) or prev_ma <= 0:
            continue

        price = cur['收盘']
        slope = (ma - prev_ma) / prev_ma * 100
        dev = (price - ma) / ma * 100
        momentum = 0.0
        if momentum_lookback > 0 and i >= momentum_lookback:
            ref_price = df.iloc[i - momentum_lookback]['收盘']
            if ref_price > 0:
                momentum = (price - ref_price) / ref_price * 100

        enter = (
            price > ma * (1 + entry_buffer / 100) and
            slope >= min_slope and
            dev <= max_dev and
            momentum >= min_momentum
        )

        if enter and entry_confirm_days > 1:
            recent = df.iloc[i - entry_confirm_days + 1:i + 1]
            for _, row in recent.iterrows():
                row_ma = row.get(ma_col, np.nan)
                if np.isnan(row_ma) or row['收盘'] <= row_ma * (1 + entry_buffer / 100):
                    enter = False
                    break

        exit_trend = True
        if i - exit_confirm_days + 1 < 0:
            exit_trend = False
        else:
            recent = df.iloc[i - exit_confirm_days + 1:i + 1]
            for _, row in recent.iterrows():
                row_ma = row.get(ma_col, np.nan)
                if np.isnan(row_ma) or row['收盘'] >= row_ma:
                    exit_trend = False
                    break

        if enter:
            signals.iloc[i, signals.columns.get_loc('signal')] = 'buy'
            signals.iloc[i, signals.columns.get_loc('reason')] = (
                f'MA{ma_window}趋势持有(斜率{slope:+.1f}%, 偏离{dev:+.1f}%, 动量{momentum:+.1f}%, 确认{entry_confirm_days}d)'
            )
        elif exit_trend:
            signals.iloc[i, signals.columns.get_loc('signal')] = 'sell'
            signals.iloc[i, signals.columns.get_loc('reason')] = f'连续{exit_confirm_days}日跌破MA{ma_window}'

    return signals


def get_index_codes(index_key, limit=None):
    key = index_key.lower()
    if key not in INDEX_MAP:
        raise ValueError(f"不支持的指数: {index_key}，支持: {', '.join(INDEX_MAP)}")
    name, index_codes = INDEX_MAP[key]
    codes = {}
    for index_code in index_codes:
        df = bt.DataSource.get_index_stocks(index_code)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                codes[row['代码']] = row['名称']
    result = list(codes.keys())
    if limit:
        result = result[:limit]
    print(f"指数股票池: {name}，加载 {len(result)} 只")
    return result


def load_data(codes, days):
    """预加载每只股票数据，避免每组参数重复访问行情源。"""
    data = {}
    for code in codes:
        df = bt.fetch_stock_data(code, days=days)
        if df is not None and not df.empty:
            data[code] = bt.calculate_indicators(df)
    return data


def score_params(data_by_code, params, position_pct, gate_kwargs):
    (
        ma_window,
        slope_lookback,
        min_slope,
        max_dev,
        entry_confirm_days,
        entry_buffer,
        exit_confirm_days,
        stop_loss,
        momentum_lookback,
        min_momentum,
    ) = params
    full_returns = []
    oos_returns = []
    full_alpha = []
    oos_alpha = []
    full_pass = 0
    oos_pass = 0
    total_trades = 0

    sim_kwargs = {
        'position_pct': position_pct,
        'stop_loss_pct': stop_loss,
        'trailing_activate_pct': bt.TREND_TRAILING_ACTIVATE_PCT,
        'trailing_stop_pct': bt.TREND_TRAILING_STOP_PCT,
        'max_holding_days': bt.TREND_MAX_HOLDING_DAYS,
    }

    for _, df_src in data_by_code.items():
        df = df_src.copy()
        signals = make_trend_signals(
            df,
            ma_window,
            slope_lookback,
            min_slope,
            max_dev,
            entry_confirm_days,
            entry_buffer,
            exit_confirm_days,
            momentum_lookback,
            min_momentum,
        )
        start_idx = min(bt.SIGNAL_START_OFFSET, max(0, len(df) - 130))
        split_idx = bt._get_oos_split_idx(df, start_idx, 0.33)

        metrics, _ = bt._simulate_strategy(df, signals, start_idx, sim_kwargs)
        gate = bt.evaluate_profit_gate(metrics, **gate_kwargs)
        full_returns.append(metrics['total_return'])
        full_alpha.append(metrics['alpha_vs_buy_hold'])
        total_trades += metrics['total_trades']
        full_pass += 1 if gate['passed'] else 0

        if split_idx is not None:
            oos_metrics, _ = bt._simulate_strategy(df, signals, split_idx, sim_kwargs)
            oos_gate = bt.evaluate_profit_gate(oos_metrics, **gate_kwargs)
            oos_returns.append(oos_metrics['total_return'])
            oos_alpha.append(oos_metrics['alpha_vs_buy_hold'])
            oos_pass += 1 if oos_gate['passed'] else 0

    if not full_returns:
        return None

    return {
        'params': params,
        'full_avg': float(np.mean(full_returns)),
        'oos_avg': float(np.mean(oos_returns)) if oos_returns else 0.0,
        'full_alpha_avg': float(np.mean(full_alpha)),
        'oos_alpha_avg': float(np.mean(oos_alpha)) if oos_alpha else 0.0,
        'full_pass': full_pass,
        'oos_pass': oos_pass,
        'n': len(full_returns),
        'oos_n': len(oos_returns),
        'trades': total_trades,
    }


def format_params(params):
    (
        ma_window,
        slope_lookback,
        min_slope,
        max_dev,
        entry_confirm_days,
        entry_buffer,
        exit_confirm_days,
        stop_loss,
        momentum_lookback,
        min_momentum,
    ) = params
    return (
        f"MA{ma_window} lookback={slope_lookback} min_slope={min_slope:.1f} "
        f"max_dev={max_dev:.0f}% entry={entry_confirm_days}d+{entry_buffer:.0f}% "
        f"mom{momentum_lookback}>={min_momentum:.0f}% exit={exit_confirm_days}d stop={stop_loss:.0f}%"
    )


def main():
    parser = argparse.ArgumentParser(description='扫描策略C参数并报告全样本/样本外表现')
    parser.add_argument('codes', nargs='*', help='股票代码列表')
    parser.add_argument('--multi', action='store_true', help='使用预设4只代表股票')
    parser.add_argument('--index', choices=sorted(INDEX_MAP), help='使用指数成分股股票池')
    parser.add_argument('--limit', type=int, help='限制指数股票池前N只，便于快速验证')
    parser.add_argument('--days', type=int, default=bt.BACKTEST_DAYS, help='获取最近N个自然日数据')
    parser.add_argument('--position-pct', type=float, default=0.5, help='单票仓位')
    parser.add_argument('--top', type=int, default=10, help='显示前N组参数')
    parser.add_argument('--min-trades', type=int, default=bt.MIN_PROFIT_TRADES)
    parser.add_argument('--min-annual-return', type=float, default=5.0)
    parser.add_argument('--max-drawdown', type=float, default=15.0)
    parser.add_argument('--min-profit-factor', type=float, default=1.2)
    parser.add_argument('--min-alpha', type=float, default=0.0)
    args = parser.parse_args()

    if args.index:
        codes = get_index_codes(args.index, args.limit)
    elif args.multi:
        codes = list(bt.PRESET_STOCKS.keys())
    else:
        codes = args.codes
    if not codes:
        parser.error('请提供股票代码，或使用 --multi')

    gate_kwargs = {
        'min_trades': args.min_trades,
        'min_annual_return': args.min_annual_return,
        'max_drawdown': args.max_drawdown,
        'min_profit_factor': args.min_profit_factor,
        'min_alpha': args.min_alpha,
    }

    data_by_code = load_data(codes, args.days)
    if not data_by_code:
        raise SystemExit('无法获取任何股票数据')

    rows = []
    for params in PARAM_GRID:
        row = score_params(data_by_code, params, args.position_pct, gate_kwargs)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda r: (r['oos_pass'], r['oos_avg'], r['full_pass'], r['full_avg']), reverse=True)

    print("\n策略C参数扫描")
    print(f"股票数: {len(codes)} | 样本天数: {args.days} | 仓位: {args.position_pct:.0%}")
    print("-" * 118)
    print(f"{'排名':<4} {'全样本':>8} {'样本外':>8} {'全闸门':>6} {'外闸门':>6} {'外超额':>8} {'交易':>5} 参数")
    print("-" * 118)
    for i, row in enumerate(rows[:args.top], 1):
        print(
            f"{i:<4} {row['full_avg']:>+7.2f}% {row['oos_avg']:>+7.2f}% "
            f"{row['full_pass']:>2}/{row['n']:<3} {row['oos_pass']:>2}/{row['oos_n']:<3} "
            f"{row['oos_alpha_avg']:>+7.2f}% {row['trades']:>5} {format_params(row['params'])}"
        )
    print("-" * 118)
    print("排序优先样本外闸门和样本外收益。结果只能作为下一轮候选，不代表未来收益。")


if __name__ == '__main__':
    main()
