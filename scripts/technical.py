#!/usr/bin/env python3
"""
公共技术指标计算模块

提供统一的技术指标计算函数，避免各脚本重复实现。
所有函数均接收 DataFrame 并原地添加列，返回 DataFrame。

包含：
- 均线（MA）+ 斜率
- MACD(8,17,9)
- KDJ(6,3,3)
- RSI(14)
- 成交量均线
- 高低点递增/递减检测
- 均线排列分析
- 钟摆位置分析（均线偏离度）
- 趋势强度评分
"""

import pandas as pd
import numpy as np


# ============================================================
# 技术指标计算
# ============================================================

def calculate_ma(df, windows=None, slope_period=5):
    """
    计算多级别均线 + 斜率

    参数:
        df: DataFrame，需包含 '收盘' 列
        windows: 均线窗口列表，默认 [5,10,20,60,120,250]
        slope_period: 斜率计算周期（默认5日变化率）

    返回:
        df（原地修改），新增 MA5/MA10/... 及 MA5_slope/MA10_slope/... 列
    """
    if windows is None:
        windows = [5, 10, 20, 60, 120, 250]

    for w in windows:
        col = f'MA{w}'
        if len(df) >= w:
            df[col] = df['收盘'].rolling(window=w).mean()
        # 长期均线数据不足时不创建列

    # 斜率（仅对短中期均线计算）
    slope_windows = [w for w in windows if w <= 60 and f'MA{w}' in df.columns]
    for w in slope_windows:
        col = f'MA{w}'
        df[f'{col}_slope'] = (df[col] - df[col].shift(slope_period)) / df[col].shift(slope_period) * 100

    return df


def calculate_macd(df, fast=8, slow=17, signal=9):
    """
    计算 MACD

    参数:
        df: DataFrame，需包含 '收盘' 列
        fast/slow/signal: MACD 参数，默认 (8,17,9)

    返回:
        df（原地修改），新增 DIF/DEA/MACD 列
    """
    exp_fast = df['收盘'].ewm(span=fast, adjust=False).mean()
    exp_slow = df['收盘'].ewm(span=slow, adjust=False).mean()
    df['DIF'] = exp_fast - exp_slow
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    return df


def calculate_kdj(df, n=6, m1=3, m2=3):
    """
    计算 KDJ

    参数:
        df: DataFrame，需包含 '收盘'/'最高'/'最低' 列
        n/m1/m2: KDJ 参数，默认 (6,3,3)

    返回:
        df（原地修改），新增 RSV/K/D/J 列
    """
    low_n = df['最低'].rolling(window=n).min()
    high_n = df['最高'].rolling(window=n).max()
    df['RSV'] = (df['收盘'] - low_n) / (high_n - low_n) * 100
    df['K'] = df['RSV'].ewm(com=m1 - 1, adjust=False).mean()
    df['D'] = df['K'].ewm(com=m2 - 1, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df


def calculate_rsi(df, period=14):
    """
    计算 RSI

    参数:
        df: DataFrame，需包含 '收盘' 列
        period: RSI 周期，默认 14

    返回:
        df（原地修改），新增 RSI 列
    """
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_volume_ma(df, windows=None):
    """
    计算成交量均线

    参数:
        df: DataFrame，需包含 '成交量' 列
        windows: 均线窗口列表，默认 [5, 20]

    返回:
        df（原地修改），新增 VOL_MA5/VOL_MA20 列
    """
    if windows is None:
        windows = [5, 20]

    for w in windows:
        df[f'VOL_MA{w}'] = df['成交量'].rolling(window=w).mean()

    return df


def calculate_all_indicators(df):
    """
    一次性计算所有技术指标（便捷函数）

    参数:
        df: DataFrame，需包含 '收盘'/'最高'/'最低'/'成交量' 列

    返回:
        df（原地修改）
    """
    calculate_ma(df)
    calculate_macd(df)
    calculate_kdj(df)
    calculate_rsi(df)
    calculate_volume_ma(df)
    return df


# ============================================================
# 分析函数
# ============================================================

def detect_highs_lows(df, window=20):
    """
    检测近期高低点递增/递减

    参数:
        df: DataFrame，需包含 '最高'/'最低' 列
        window: 检测窗口（默认近20根K线）

    返回:
        dict: {
            'highs': list,          # 高点值列表
            'lows': list,           # 低点值列表
            'highs_rising': bool,   # 高点是否递增
            'lows_rising': bool,    # 低点是否递增
            'highs_falling': bool,  # 高点是否递减
            'lows_falling': bool,   # 低点是否递减
        }
    """
    recent = df.tail(window)
    highs = []
    lows = []

    for i in range(2, len(recent) - 2):
        row = recent.iloc[i]
        prev1 = recent.iloc[i - 1]
        prev2 = recent.iloc[i - 2]
        next1 = recent.iloc[i + 1]
        next2 = recent.iloc[i + 2]

        if (row['最高'] >= prev1['最高'] and row['最高'] >= prev2['最高'] and
                row['最高'] >= next1['最高'] and row['最高'] >= next2['最高']):
            highs.append(row['最高'])

        if (row['最低'] <= prev1['最低'] and row['最低'] <= prev2['最低'] and
                row['最低'] <= next1['最低'] and row['最低'] <= next2['最低']):
            lows.append(row['最低'])

    highs_rising = len(highs) >= 2 and highs[-1] > highs[0]
    lows_rising = len(lows) >= 2 and lows[-1] > lows[0]
    highs_falling = len(highs) >= 2 and highs[-1] < highs[0]
    lows_falling = len(lows) >= 2 and lows[-1] < lows[0]

    return {
        'highs': highs,
        'lows': lows,
        'highs_rising': highs_rising,
        'lows_rising': lows_rising,
        'highs_falling': highs_falling,
        'lows_falling': lows_falling,
    }


def _safe_ma(latest, col):
    """安全获取均线值，NaN 返回 None"""
    val = latest.get(col, np.nan)
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


def analyze_ma_alignment(latest, price):
    """
    均线排列分析

    参数:
        latest: 最新一行数据（Series）
        price: 当前价格

    返回:
        dict: {
            'desc': str,       # 排列描述（如 '完美多头(MA5>10>20>60>120)'）
            'score': int,      # 排列得分（-3 ~ +3）
            'ma_values': dict, # 各均线值 {'MA5': x, 'MA10': x, ...}
        }
    """
    ma5 = _safe_ma(latest, 'MA5')
    ma10 = _safe_ma(latest, 'MA10')
    ma20 = _safe_ma(latest, 'MA20')
    ma60 = _safe_ma(latest, 'MA60')
    ma120 = _safe_ma(latest, 'MA120')
    ma250 = _safe_ma(latest, 'MA250')

    ma_values = {
        'MA5': ma5, 'MA10': ma10, 'MA20': ma20,
        'MA60': ma60, 'MA120': ma120, 'MA250': ma250,
    }

    # 短期均线缺失则无法判断
    if ma5 is None or ma10 is None or ma20 is None:
        return {'desc': '数据不足', 'score': 0, 'ma_values': ma_values}

    if ma60 is not None and ma5 > ma10 > ma20 > ma60:
        if ma120 is not None and price > ma120:
            desc = '完美多头(MA5>10>20>60, 价格>MA120)'
        else:
            desc = '强势多头(MA5>10>20>60)'
        score = 3
    elif ma5 > ma10 > ma20:
        desc = '多头(MA5>10>20)'
        score = 2
    elif ma5 > ma10:
        desc = '偏多(MA5>10)'
        score = 1
    elif ma60 is not None and ma5 < ma10 < ma20 < ma60:
        desc = '空头(MA5<10<20<60)'
        score = -3
    elif ma5 < ma10 < ma20:
        desc = '偏空(MA5<10<20)'
        score = -2
    elif ma5 < ma10:
        desc = '偏空(MA5<10)'
        score = -1
    else:
        desc = '震荡缠绕'
        score = 0

    return {'desc': desc, 'score': score, 'ma_values': ma_values}


def calculate_pendulum(price, ma_values):
    """
    钟摆位置分析（均线偏离度）

    参数:
        price: 当前价格
        ma_values: dict, 各均线值 {'MA20': x, 'MA60': x, ...}

    返回:
        dict: {
            'MA20': {'value': x, 'deviation': x, 'phase': str},
            'MA60': {...},
            ...
            'overall': str,  # 综合判断
        }
    """
    result = {}

    for ma_name in ['MA20', 'MA60', 'MA120', 'MA250']:
        ma_val = ma_values.get(ma_name)
        if ma_val is None or ma_val == 0:
            result[ma_name] = {'value': None, 'deviation': None, 'phase': '数据不足'}
            continue

        dev = (price - ma_val) / ma_val * 100

        if dev > 15:
            phase = '极度偏高（绳子极紧，回归压力大）'
        elif dev > 10:
            phase = '偏高（注意回归压力）'
        elif dev > 5:
            phase = '略高'
        elif dev > -2:
            phase = '中枢附近（适合做T）'
        elif dev > -5:
            phase = '略低'
        elif dev > -10:
            phase = '偏低（回归动力增强）'
        else:
            phase = '极度偏低（绳子极紧，反弹动力大）'

        result[ma_name] = {'value': ma_val, 'deviation': dev, 'phase': phase}

    # 综合钟摆阶段
    dev_ma20 = result.get('MA20', {}).get('deviation')
    if dev_ma20 is not None:
        if abs(dev_ma20) <= 3:
            overall = '钟摆在中枢附近，适合做T'
        elif dev_ma20 > 5:
            overall = '钟摆偏高，做T以卖出为主'
        elif dev_ma20 < -5:
            overall = '钟摆偏低，做T以买入为主'
        else:
            overall = '钟摆在中间区域'
    else:
        overall = '数据不足'

    result['overall'] = overall
    return result


def calculate_trend_strength(latest, df, price):
    """
    趋势强度评分（0-10）

    参数:
        latest: 最新一行数据（Series）
        df: 完整 DataFrame（需含均线列）
        price: 当前价格

    返回:
        dict: {
            'score': int,          # 0-10
            'details': list[str],  # 评分明细
        }
    """
    strength = 0
    details = []

    # 1. 均线排列（0-3分）
    alignment = analyze_ma_alignment(latest, price)
    align_score = max(0, alignment['score'])
    strength += align_score
    if align_score >= 2:
        details.append(f"均线{alignment['desc']}")

    # 2. 高低点递增（0-2分）
    hl = detect_highs_lows(df)
    if hl['highs_rising']:
        strength += 1
        details.append('高点递增')
    if hl['lows_rising']:
        strength += 1
        details.append('低点递增')

    # 3. MA20 斜率（0-2分）
    ma20_slope = latest.get('MA20_slope', 0)
    if isinstance(ma20_slope, float) and np.isnan(ma20_slope):
        ma20_slope = 0
    if ma20_slope > 3:
        strength += 2
        details.append(f'MA20加速上行({ma20_slope:+.1f}%)')
    elif ma20_slope > 0.5:
        strength += 1
        details.append(f'MA20上行({ma20_slope:+.1f}%)')
    elif ma20_slope < -2:
        details.append(f'MA20下行({ma20_slope:+.1f}%)')

    # 4. 近20日涨幅（0-2分）
    if len(df) >= 20:
        price_20d_ago = df.iloc[-20]['收盘']
        change_20d = (price - price_20d_ago) / price_20d_ago * 100
        if change_20d > 5:
            strength += 2
            details.append(f'20日强势(+{change_20d:.1f}%)')
        elif change_20d > 0:
            strength += 1
            details.append(f'20日偏强(+{change_20d:.1f}%)')

    # 5. 价格在 MA120 以上（0-1分）
    ma120 = _safe_ma(latest, 'MA120')
    if ma120 is not None and price > ma120:
        strength += 1

    strength = min(10, strength)
    return {'score': strength, 'details': details}
