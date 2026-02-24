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


def calculate_bollinger(df, window=20, num_std=2):
    """
    计算布林带

    参数:
        df: DataFrame，需包含 '收盘' 列
        window: 中轨周期（默认20）
        num_std: 标准差倍数（默认2）

    返回:
        df（原地修改），新增 BOLL_MID/BOLL_UPPER/BOLL_LOWER 列
    """
    if len(df) < window:
        return df
    df['BOLL_MID'] = df['收盘'].rolling(window=window).mean()
    rolling_std = df['收盘'].rolling(window=window).std()
    df['BOLL_UPPER'] = df['BOLL_MID'] + num_std * rolling_std
    df['BOLL_LOWER'] = df['BOLL_MID'] - num_std * rolling_std
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
    calculate_bollinger(df)
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
    多级别钟摆位置分析（均线偏离度）

    参数:
        price: 当前价格
        ma_values: dict, 各均线值 {'MA5': x, 'MA10': x, 'MA20': x, 'MA60': x, ...}

    返回:
        dict: {
            'MA5': {'value': x, 'deviation': x, 'phase': str},
            'MA10': {...},
            'MA20': {...},
            'MA60': {...},
            ...
            'short_term': str,   # 短期钟摆（MA5/MA10）
            'mid_term': str,     # 中期钟摆（MA20）
            'overall': str,      # 综合判断
        }
    """
    result = {}

    # 短期均线用更小的阈值
    short_thresholds = {'MA5': (3, 5, 8), 'MA10': (4, 6, 10)}  # (略高, 偏高, 极度)
    # 中长期均线用标准阈值
    long_thresholds = {'MA20': (5, 8, 12), 'MA60': (8, 15, 20), 'MA120': (10, 15, 25), 'MA250': (15, 20, 30)}

    all_thresholds = {**short_thresholds, **long_thresholds}

    for ma_name in ['MA5', 'MA10', 'MA20', 'MA60', 'MA120', 'MA250']:
        ma_val = ma_values.get(ma_name)
        if ma_val is None or ma_val == 0:
            result[ma_name] = {'value': None, 'deviation': None, 'phase': '数据不足'}
            continue

        dev = (price - ma_val) / ma_val * 100
        t_high, t_very_high, t_extreme = all_thresholds.get(ma_name, (5, 10, 15))

        if dev > t_extreme:
            phase = '极度偏高（绳子极紧，回归压力大）'
        elif dev > t_very_high:
            phase = '偏高（注意回归压力）'
        elif dev > t_high:
            phase = '略高'
        elif dev > -2:
            phase = '中枢附近（适合做T）'
        elif dev > -t_high:
            phase = '略低'
        elif dev > -t_very_high:
            phase = '偏低（回归动力增强）'
        else:
            phase = '极度偏低（绳子极紧，反弹动力大）'

        result[ma_name] = {'value': ma_val, 'deviation': dev, 'phase': phase}

    # 短期钟摆判断（MA5/MA10联合）
    dev_ma5 = result.get('MA5', {}).get('deviation')
    dev_ma10 = result.get('MA10', {}).get('deviation')
    if dev_ma5 is not None and dev_ma10 is not None:
        if dev_ma5 <= 1 and dev_ma10 <= 2:
            short_term = '短期均线收敛（安全）'
        elif dev_ma5 > 5 or dev_ma10 > 4:
            short_term = '短期过热（注意回调风险）'
        elif dev_ma5 < -3 or dev_ma10 < -3:
            short_term = '短期超跌（反弹动力强）'
        else:
            short_term = '短期中性'
    else:
        short_term = '数据不足'
    result['short_term'] = short_term

    # 中期钟摆判断（MA20）
    dev_ma20 = result.get('MA20', {}).get('deviation')
    if dev_ma20 is not None:
        if abs(dev_ma20) <= 3:
            mid_term = '钟摆在中枢附近，适合做T'
        elif dev_ma20 > 8:
            mid_term = '钟摆严重偏高，追高风险大'
        elif dev_ma20 > 5:
            mid_term = '钟摆偏高，做T以卖出为主'
        elif dev_ma20 < -8:
            mid_term = '钟摆严重偏低，反弹动力大'
        elif dev_ma20 < -5:
            mid_term = '钟摆偏低，做T以买入为主'
        else:
            mid_term = '钟摆在中间区域'
    else:
        mid_term = '数据不足'
    result['mid_term'] = mid_term

    # 综合判断（短期+中期联合）
    if dev_ma20 is not None and dev_ma5 is not None:
        if abs(dev_ma20) <= 3 and dev_ma5 <= 1:
            overall = '均线簇收敛，最佳买点'
        elif abs(dev_ma20) <= 3:
            overall = '接近中枢，可考虑做T'
        elif dev_ma20 > 5 and dev_ma5 > 3:
            overall = '多级别过热，建议等待回踩'
        elif dev_ma20 < -5 and dev_ma5 < -2:
            overall = '多级别超跌，关注反弹机会'
        else:
            overall = '钟摆在中间区域'
    elif dev_ma20 is not None:
        if abs(dev_ma20) <= 3:
            overall = '钟摆在中枢附近'
        elif dev_ma20 > 5:
            overall = '钟摆偏高，建议等待回踩'
        elif dev_ma20 < -5:
            overall = '钟摆偏低，关注反弹'
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


def detect_topping_signals(df, price=None):
    """
    检测行情见顶/主力出货信号

    核心场景：MA20仍在向上（看起来趋势还好），但短期已经开始转弱。
    需要识别"趋势末端"的典型特征，避免在行情结束时还误判为买入机会。

    检测维度（每个维度独立打分，总分越高见顶概率越大）：
    1. 短期均线拐头：MA5 连续下行，MA5下穿MA10（短线走弱）
    2. 连续阴线：近N日阴线比例高，说明卖压持续
    3. 量价背离：价格创新高但成交量萎缩（拉高出货的经典特征）
    4. 高位放量滞涨：大量成交但价格不涨（主力对倒出货）
    5. MACD顶背离：价格创新高但DIF走低
    6. 高位长上影线：说明上方抛压重
    7. MA5/MA10 死叉且远离MA20：短期趋势已经反转
    8. 高位缩量回落：从近期高点回落+成交量萎缩+连续下跌

    参数:
        df: DataFrame，需包含 '收盘'/'开盘'/'最高'/'最低'/'成交量' 以及均线列
        price: 当前价格（可选，默认取最新收盘价）

    返回:
        dict: {
            'score': int,           # 见顶信号强度（0-100），>= 50 需要警惕，>= 70 高概率见顶
            'signals': list[str],   # 触发的信号列表
            'level': str,           # '安全' / '注意' / '警惕' / '危险'
            'is_topping': bool,     # 是否判定为见顶（score >= 50）
            'details': dict,        # 各维度详细数据
        }
    """
    if len(df) < 30:
        return {'score': 0, 'signals': [], 'level': '数据不足', 'is_topping': False, 'details': {}}

    latest = df.iloc[-1]
    if price is None:
        price = latest['收盘']

    score = 0
    signals = []
    details = {}

    # ------------------------------------------------------------------
    # 1. 短期均线拐头下行（最大20分）
    # MA5 连续下行 = 短线资金撤离
    # ------------------------------------------------------------------
    ma5_vals = df['MA5'].tail(5).tolist()
    ma5_consecutive_down = 0
    for i in range(len(ma5_vals) - 1, 0, -1):
        if not (np.isnan(ma5_vals[i]) or np.isnan(ma5_vals[i-1])):
            if ma5_vals[i] < ma5_vals[i-1]:
                ma5_consecutive_down += 1
            else:
                break

    ma5_turning = ma5_consecutive_down >= 2
    ma5_score = 0
    if ma5_consecutive_down >= 3:
        ma5_score = 20
        signals.append(f'MA5连续{ma5_consecutive_down}日下行（短线资金持续撤离）')
    elif ma5_consecutive_down >= 2:
        ma5_score = 12
        signals.append(f'MA5连续{ma5_consecutive_down}日下行')

    # MA5下穿MA10（短线死叉）
    ma5 = _safe_ma(latest, 'MA5')
    ma10 = _safe_ma(latest, 'MA10')
    prev = df.iloc[-2]
    if ma5 is not None and ma10 is not None:
        prev_ma5 = _safe_ma(prev, 'MA5')
        prev_ma10 = _safe_ma(prev, 'MA10')
        if prev_ma5 is not None and prev_ma10 is not None:
            if ma5 < ma10 and prev_ma5 >= prev_ma10:
                ma5_score = min(20, ma5_score + 10)
                signals.append('MA5下穿MA10（短线死叉）')
            elif ma5 < ma10:
                ma5_score = min(20, ma5_score + 5)

    score += ma5_score
    details['ma5_turning'] = {'consecutive_down': ma5_consecutive_down, 'score': ma5_score}

    # ------------------------------------------------------------------
    # 2. 连续阴线（最大15分）
    # 近5日阴线比例高 = 卖压持续
    # ------------------------------------------------------------------
    recent_5 = df.tail(5)
    down_days = sum(1 for _, row in recent_5.iterrows() if row['收盘'] < row['开盘'])
    down_pct_days = sum(1 for _, row in recent_5.iterrows()
                        if row['收盘'] < df.iloc[max(0, df.index.get_loc(row.name) - 1)]['收盘'])

    consecutive_down = 0
    for i in range(len(df) - 1, max(len(df) - 6, 0), -1):
        if df.iloc[i]['收盘'] < df.iloc[i]['开盘']:
            consecutive_down += 1
        else:
            break

    candle_score = 0
    if consecutive_down >= 4:
        candle_score = 15
        signals.append(f'连续{consecutive_down}根阴线（卖压极强）')
    elif consecutive_down >= 3:
        candle_score = 10
        signals.append(f'连续{consecutive_down}根阴线（持续卖压）')
    elif down_days >= 4:
        candle_score = 8
        signals.append(f'近5日{down_days}阴（卖压偏重）')

    score += candle_score
    details['candle'] = {'consecutive_down': consecutive_down, 'down_5d': down_days, 'score': candle_score}

    # ------------------------------------------------------------------
    # 3. 量价背离（最大20分）
    # 价格在近期高位，但成交量逐步萎缩 = 上涨动能衰竭
    # ------------------------------------------------------------------
    vp_score = 0

    recent_20 = df.tail(20)
    high_20d = recent_20['最高'].max()
    drawdown_from_high = (high_20d - price) / high_20d * 100
    price_near_high = drawdown_from_high < 5  # 放宽到距20日最高点5%以内

    if price_near_high and len(df) >= 20:
        vol_first_half = df.iloc[-20:-10]['成交量'].mean()
        vol_second_half = df.iloc[-10:]['成交量'].mean()
        if vol_first_half > 0:
            vol_ratio = vol_second_half / vol_first_half
            if vol_ratio < 0.6:
                vp_score = 20
                signals.append(f'高位量价背离（量能萎缩{(1-vol_ratio)*100:.0f}%，拉高出货风险）')
            elif vol_ratio < 0.75:
                vp_score = 12
                signals.append(f'量能逐步萎缩（后半段量比前半段缩{(1-vol_ratio)*100:.0f}%）')
            details['volume_divergence'] = {'vol_ratio': vol_ratio}

    # 缩量创新高
    if len(df) >= 5:
        recent_high = df.tail(5)['最高'].max()
        prev_high = df.tail(20).head(15)['最高'].max() if len(df) >= 20 else 0
        recent_vol = df.tail(5)['成交量'].mean()
        prev_vol = df.tail(20).head(15)['成交量'].mean() if len(df) >= 20 else recent_vol
        if recent_high > prev_high and prev_vol > 0 and recent_vol / prev_vol < 0.7:
            vp_score = min(20, vp_score + 8)
            if '量价背离' not in ' '.join(signals):
                signals.append('缩量创新高（上涨动能不足）')

    score += vp_score
    details['volume_price'] = {'score': vp_score, 'near_high': price_near_high}

    # ------------------------------------------------------------------
    # 4. 高位放量滞涨 / 放量下跌（最大15分）
    # 放量但价格不涨甚至下跌 = 主力对倒出货
    # ------------------------------------------------------------------
    stagnation_score = 0
    if 'VOL_MA5' in latest and latest['VOL_MA5'] > 0:
        vol_ratio = latest['成交量'] / latest['VOL_MA5']
        today_change = (latest['收盘'] - latest['开盘']) / latest['开盘'] * 100

        # 放量下跌
        if vol_ratio > 1.5 and today_change < -1:
            stagnation_score = 15
            signals.append(f'放量下跌（量比{vol_ratio:.1f}，跌{today_change:.1f}%，主力出货特征）')
        elif vol_ratio > 1.3 and today_change < -0.5:
            stagnation_score = 10
            signals.append(f'放量阴线（量比{vol_ratio:.1f}，跌{today_change:.1f}%）')

        # 近3日放量滞涨
        if stagnation_score == 0 and len(df) >= 3:
            recent_3 = df.tail(3)
            avg_vol_3d = recent_3['成交量'].mean()
            avg_vol_20d = df.tail(20)['成交量'].mean() if len(df) >= 20 else avg_vol_3d
            price_change_3d = (price - df.iloc[-4]['收盘']) / df.iloc[-4]['收盘'] * 100 if len(df) >= 4 else 0
            if avg_vol_20d > 0 and avg_vol_3d / avg_vol_20d > 1.3 and abs(price_change_3d) < 1:
                stagnation_score = 10
                signals.append(f'近3日放量滞涨（量增价不涨，资金分歧）')

    score += stagnation_score
    details['stagnation'] = {'score': stagnation_score}

    # ------------------------------------------------------------------
    # 5. MACD顶背离（最大15分）
    # 价格创新高但DIF没有同步新高 = 动量衰减
    # ------------------------------------------------------------------
    divergence_score = 0
    if 'DIF' in df.columns and len(df) >= 20:
        # 找近20日价格高点和DIF高点
        recent = df.tail(20)
        price_highs = []
        dif_at_price_highs = []

        for i in range(2, len(recent) - 2):
            row = recent.iloc[i]
            if (row['最高'] >= recent.iloc[i-1]['最高'] and
                row['最高'] >= recent.iloc[i-2]['最高'] and
                row['最高'] >= recent.iloc[i+1]['最高'] and
                row['最高'] >= recent.iloc[i+2]['最高']):
                price_highs.append(row['最高'])
                dif_at_price_highs.append(row['DIF'])

        if len(price_highs) >= 2:
            if price_highs[-1] >= price_highs[-2] and dif_at_price_highs[-1] < dif_at_price_highs[-2]:
                divergence_score = 15
                signals.append('MACD顶背离（价格新高但动量走弱，趋势可能反转）')

    score += divergence_score
    details['macd_divergence'] = {'score': divergence_score}

    # ------------------------------------------------------------------
    # 6. 高位长上影线（最大10分）
    # 上影线长说明上方抛压重，是见顶的典型K线特征
    # ------------------------------------------------------------------
    shadow_score = 0
    upper_shadow = latest['最高'] - max(latest['收盘'], latest['开盘'])
    body = abs(latest['收盘'] - latest['开盘'])
    total_range = latest['最高'] - latest['最低']

    if total_range > 0:
        shadow_ratio = upper_shadow / total_range
        # 近3日出现长上影线
        long_shadow_count = 0
        for i in range(-3, 0):
            if abs(i) <= len(df):
                row = df.iloc[i]
                r_total = row['最高'] - row['最低']
                if r_total > 0:
                    r_shadow = (row['最高'] - max(row['收盘'], row['开盘'])) / r_total
                    if r_shadow > 0.5:
                        long_shadow_count += 1

        if long_shadow_count >= 2:
            shadow_score = 10
            signals.append(f'近3日出现{long_shadow_count}根长上影线（上方抛压重）')
        elif shadow_ratio > 0.6 and total_range / price * 100 > 1:
            shadow_score = 6
            signals.append('今日长上影线（上方承压）')

    score += shadow_score
    details['upper_shadow'] = {'score': shadow_score}

    # ------------------------------------------------------------------
    # 7. MA5/MA10走平或死叉 + 远离MA20（最大5分）
    # 短线已转弱但中线还在上行 = 趋势末端分歧
    # ------------------------------------------------------------------
    diverge_score = 0
    ma20 = _safe_ma(latest, 'MA20')
    if ma5 is not None and ma10 is not None and ma20 is not None and ma20 > 0:
        dev_ma20 = (price - ma20) / ma20 * 100
        if ma5 < ma10 and dev_ma20 > 5:
            diverge_score = 5
            signals.append(f'短线死叉但远离MA20({dev_ma20:+.1f}%)，趋势末端分歧')
        elif ma5_turning and dev_ma20 > 3:
            diverge_score = 3

    score += diverge_score
    details['short_long_diverge'] = {'score': diverge_score}

    # ------------------------------------------------------------------
    # 8. 高位回落 + 缩量（最大15分）
    # 从近期高点明显回落且成交量萎缩 = 资金撤退、行情转弱
    # 这是"缩量滞涨后连跌"这类场景的关键检测维度
    # ------------------------------------------------------------------
    retreat_score = 0
    if len(df) >= 10:
        high_10d = df.tail(10)['最高'].max()
        retreat_pct = (high_10d - price) / high_10d * 100

        # 近3日成交量 vs 近20日均量（或近10日均量）
        recent_3_vol = df.tail(3)['成交量'].mean()
        ref_vol = df.tail(20)['成交量'].mean() if len(df) >= 20 else df.tail(10)['成交量'].mean()

        vol_shrink = (recent_3_vol / ref_vol) if ref_vol > 0 else 1.0

        # 近3日连续下跌（收盘价逐日下降）
        recent_closes = df.tail(4)['收盘'].tolist()  # 取4个点看3天的趋势
        consecutive_decline = 0
        for i in range(len(recent_closes) - 1, 0, -1):
            if recent_closes[i] < recent_closes[i-1]:
                consecutive_decline += 1
            else:
                break

        if retreat_pct >= 3 and vol_shrink < 0.6 and consecutive_decline >= 2:
            retreat_score = 15
            signals.append(f'高位缩量回落（距高点-{retreat_pct:.1f}%，量能萎缩至{vol_shrink*100:.0f}%，连跌{consecutive_decline}日）')
        elif retreat_pct >= 2 and vol_shrink < 0.7 and consecutive_decline >= 2:
            retreat_score = 10
            signals.append(f'高位缩量回落（距高点-{retreat_pct:.1f}%，量能缩至{vol_shrink*100:.0f}%）')
        elif retreat_pct >= 3 and consecutive_decline >= 2:
            retreat_score = 8
            signals.append(f'高位连续回落（距高点-{retreat_pct:.1f}%，连跌{consecutive_decline}日）')
        elif retreat_pct >= 2 and vol_shrink < 0.6:
            retreat_score = 6
            signals.append(f'高位缩量（距高点-{retreat_pct:.1f}%，量能萎缩至{vol_shrink*100:.0f}%）')

    score += retreat_score
    details['high_retreat'] = {'score': retreat_score}

    # ------------------------------------------------------------------
    # 综合判定
    # ------------------------------------------------------------------
    score = min(100, score)

    if score >= 70:
        level = '危险'
    elif score >= 50:
        level = '警惕'
    elif score >= 30:
        level = '注意'
    else:
        level = '安全'

    return {
        'score': score,
        'signals': signals,
        'level': level,
        'is_topping': score >= 50,
        'details': details,
    }


def detect_bottoming_signals(df, price=None):
    """
    检测底部反弹信号

    核心场景：股价经历较大跌幅，技术指标超卖，出现止跌企稳或反转迹象。
    用于识别"跌透了"的股票——基本面好但被错杀，即将触底反弹。

    检测维度（每个维度独立打分，总分越高底部反弹概率越大）：
    1. RSI超卖（最大15分）
    2. KDJ超卖/低位金叉（最大15分）
    3. MACD底背离（最大20分）
    4. 缩量企稳/地量见地价（最大15分）
    5. 布林带下轨支撑（最大10分）
    6. 均线低位金叉/粘合（最大15分）
    7. 底部反转K线形态（最大10分）

    参数:
        df: DataFrame，需包含技术指标列（调用前应先 calculate_all_indicators）
        price: 当前价格（可选，默认取最新收盘价）

    返回:
        dict: {
            'score': int,           # 底部信号强度 0-100
            'signals': list[str],   # 触发的信号列表
            'level': str,           # '无信号' / '弱信号' / '中等' / '强信号'
            'is_bottoming': bool,   # 是否判定为底部（score >= 40）
            'details': dict,        # 各维度详细数据
        }
    """
    if len(df) < 30:
        return {'score': 0, 'signals': [], 'level': '数据不足', 'is_bottoming': False, 'details': {}}

    latest = df.iloc[-1]
    if price is None:
        price = latest['收盘']

    score = 0
    signals = []
    details = {}

    # ------------------------------------------------------------------
    # 1. RSI 超卖（最大15分）
    # RSI < 30 = 超卖，< 20 = 极度超卖
    # ------------------------------------------------------------------
    rsi_score = 0
    rsi_val = latest.get('RSI', np.nan)
    if not (isinstance(rsi_val, float) and np.isnan(rsi_val)):
        if rsi_val < 20:
            rsi_score = 15
            signals.append(f'RSI极度超卖（{rsi_val:.1f}，反弹动力极强）')
        elif rsi_val < 25:
            rsi_score = 12
            signals.append(f'RSI强超卖（{rsi_val:.1f}）')
        elif rsi_val < 30:
            rsi_score = 8
            signals.append(f'RSI超卖（{rsi_val:.1f}）')
        elif rsi_val < 35:
            rsi_score = 4
            signals.append(f'RSI偏低（{rsi_val:.1f}）')

    score += rsi_score
    details['rsi'] = {'value': float(rsi_val) if not (isinstance(rsi_val, float) and np.isnan(rsi_val)) else None, 'score': rsi_score}

    # ------------------------------------------------------------------
    # 2. KDJ 超卖 / 低位金叉（最大15分）
    # K/D < 20 = 超卖区，J < 0 = 极度超卖
    # 低位金叉（K上穿D）= 反转信号
    # ------------------------------------------------------------------
    kdj_score = 0
    k_val = latest.get('K', np.nan)
    d_val = latest.get('D', np.nan)
    j_val = latest.get('J', np.nan)

    k_valid = not (isinstance(k_val, float) and np.isnan(k_val))
    d_valid = not (isinstance(d_val, float) and np.isnan(d_val))
    j_valid = not (isinstance(j_val, float) and np.isnan(j_val))

    if k_valid and d_valid:
        # 超卖区判断
        if k_val < 20 and d_val < 20:
            kdj_score += 8
            if j_valid and j_val < 0:
                kdj_score += 4
                signals.append(f'KDJ极度超卖（K={k_val:.0f} D={d_val:.0f} J={j_val:.0f}）')
            else:
                signals.append(f'KDJ超卖区（K={k_val:.0f} D={d_val:.0f}）')
        elif k_val < 30:
            kdj_score += 4
            signals.append(f'KDJ偏低（K={k_val:.0f}）')

        # 低位金叉检测
        if len(df) >= 2:
            prev = df.iloc[-2]
            prev_k = prev.get('K', np.nan)
            prev_d = prev.get('D', np.nan)
            if not (isinstance(prev_k, float) and np.isnan(prev_k)):
                if not (isinstance(prev_d, float) and np.isnan(prev_d)):
                    if k_val > d_val and prev_k <= prev_d and k_val < 30:
                        kdj_score = min(15, kdj_score + 7)
                        signals.append('KDJ低位金叉（K上穿D，反转信号）')

    kdj_score = min(15, kdj_score)
    score += kdj_score
    details['kdj'] = {'k': float(k_val) if k_valid else None, 'd': float(d_val) if d_valid else None, 'j': float(j_val) if j_valid else None, 'score': kdj_score}

    # ------------------------------------------------------------------
    # 3. MACD 底背离（最大20分）
    # 价格创新低但 DIF 没有同步新低 = 下跌动能衰竭，强反转信号
    # ------------------------------------------------------------------
    macd_bottom_score = 0
    if 'DIF' in df.columns and len(df) >= 20:
        recent = df.tail(30) if len(df) >= 30 else df.tail(20)
        price_lows = []
        dif_at_price_lows = []

        for i in range(2, len(recent) - 2):
            row = recent.iloc[i]
            if (row['最低'] <= recent.iloc[i-1]['最低'] and
                row['最低'] <= recent.iloc[i-2]['最低'] and
                row['最低'] <= recent.iloc[i+1]['最低'] and
                row['最低'] <= recent.iloc[i+2]['最低']):
                price_lows.append(row['最低'])
                dif_at_price_lows.append(row['DIF'])

        if len(price_lows) >= 2:
            if price_lows[-1] <= price_lows[-2] and dif_at_price_lows[-1] > dif_at_price_lows[-2]:
                macd_bottom_score = 20
                signals.append('MACD底背离（价格新低但动量回升，强反转信号）')

        # DIF 从负值区域开始上行（弱底部信号）
        if macd_bottom_score == 0 and len(df) >= 5:
            dif_vals = df['DIF'].tail(5).tolist()
            dif_rising = 0
            for i in range(len(dif_vals) - 1, 0, -1):
                if not (np.isnan(dif_vals[i]) or np.isnan(dif_vals[i-1])):
                    if dif_vals[i] > dif_vals[i-1]:
                        dif_rising += 1
                    else:
                        break
            if dif_rising >= 3 and dif_vals[-1] < 0:
                macd_bottom_score = 8
                signals.append(f'DIF负值区连续{dif_rising}日回升（下跌动能衰竭）')
            elif dif_rising >= 2 and dif_vals[-1] < 0:
                macd_bottom_score = 5

        # MACD柱由负转正或负值缩小
        if 'MACD' in df.columns and len(df) >= 3:
            macd_vals = df['MACD'].tail(3).tolist()
            if all(not np.isnan(v) for v in macd_vals):
                if macd_vals[-2] < 0 and macd_vals[-1] >= 0:
                    macd_bottom_score = min(20, macd_bottom_score + 6)
                    signals.append('MACD柱由负转正（多头力量恢复）')
                elif macd_vals[-1] < 0 and macd_vals[-1] > macd_vals[-2] > macd_vals[-3]:
                    macd_bottom_score = min(20, macd_bottom_score + 4)

    score += macd_bottom_score
    details['macd_divergence'] = {'score': macd_bottom_score}

    # ------------------------------------------------------------------
    # 4. 缩量企稳 / 地量见地价（最大15分）
    # 成交量极度萎缩 + 价格止跌 = 抛压枯竭，底部特征
    # ------------------------------------------------------------------
    volume_score = 0
    if len(df) >= 20:
        vol_20d_mean = df.tail(20)['成交量'].mean()
        vol_5d_mean = df.tail(5)['成交量'].mean()
        vol_3d_mean = df.tail(3)['成交量'].mean()
        vol_today = latest['成交量']

        vol_ratio_5d = vol_5d_mean / vol_20d_mean if vol_20d_mean > 0 else 1.0
        vol_ratio_3d = vol_3d_mean / vol_20d_mean if vol_20d_mean > 0 else 1.0

        # 价格止跌判断（近3日跌幅收窄或走平）
        recent_3_closes = df.tail(3)['收盘'].tolist()
        price_stable = True
        if len(recent_3_closes) >= 2:
            max_drop = 0
            for i in range(1, len(recent_3_closes)):
                chg = (recent_3_closes[i] - recent_3_closes[i-1]) / recent_3_closes[i-1] * 100
                max_drop = min(max_drop, chg)
            price_stable = max_drop > -1.5  # 最大单日跌幅 < 1.5%

        # 近5日有阳线（止跌信号）
        recent_5 = df.tail(5)
        up_days = sum(1 for _, row in recent_5.iterrows() if row['收盘'] > row['开盘'])

        if vol_ratio_3d < 0.4 and price_stable:
            volume_score = 15
            signals.append(f'地量企稳（量能萎缩至{vol_ratio_3d*100:.0f}%，抛压枯竭）')
        elif vol_ratio_5d < 0.5 and price_stable:
            volume_score = 12
            signals.append(f'缩量止跌（5日均量仅{vol_ratio_5d*100:.0f}%，惜售信号）')
        elif vol_ratio_5d < 0.6 and up_days >= 2:
            volume_score = 8
            signals.append(f'缩量后出现阳线（量缩至{vol_ratio_5d*100:.0f}%，{up_days}阳）')
        elif vol_ratio_5d < 0.7 and price_stable:
            volume_score = 5

        # 放量企稳加分（缩量后突然放量上涨 = 底部确认）
        if vol_today > 0 and vol_20d_mean > 0:
            if vol_today / vol_20d_mean > 1.5 and latest['收盘'] > latest['开盘']:
                prev_5_vol_ratio = df.tail(6).head(5)['成交量'].mean() / vol_20d_mean if vol_20d_mean > 0 else 1
                if prev_5_vol_ratio < 0.7:
                    volume_score = min(15, volume_score + 5)
                    signals.append('缩量后放量阳线（底部确认信号）')

    score += volume_score
    details['volume'] = {'score': volume_score}

    # ------------------------------------------------------------------
    # 5. 布林带下轨支撑（最大10分）
    # 价格触及或跌破下轨后反弹 = 超跌反弹
    # ------------------------------------------------------------------
    boll_score = 0
    boll_lower = latest.get('BOLL_LOWER', np.nan)
    boll_mid = latest.get('BOLL_MID', np.nan)

    if not (isinstance(boll_lower, float) and np.isnan(boll_lower)):
        if price <= boll_lower:
            boll_score = 10
            signals.append(f'价格跌破布林带下轨（极度超跌，¥{price:.2f} < 下轨¥{boll_lower:.2f}）')
        elif boll_lower > 0:
            dist_to_lower = (price - boll_lower) / boll_lower * 100
            if dist_to_lower < 1:
                boll_score = 8
                signals.append(f'价格逼近布林带下轨（距下轨仅{dist_to_lower:.1f}%）')
            elif dist_to_lower < 2:
                boll_score = 5

        # 触及下轨后反弹（昨天在下轨，今天回升）
        if len(df) >= 2 and boll_score < 8:
            prev = df.iloc[-2]
            prev_lower = prev.get('BOLL_LOWER', np.nan)
            if not (isinstance(prev_lower, float) and np.isnan(prev_lower)):
                if prev['收盘'] <= prev_lower and price > boll_lower:
                    boll_score = min(10, boll_score + 6)
                    signals.append('布林带下轨反弹（昨日触底今日回升）')

    score += boll_score
    details['bollinger'] = {'score': boll_score, 'lower': float(boll_lower) if not (isinstance(boll_lower, float) and np.isnan(boll_lower)) else None}

    # ------------------------------------------------------------------
    # 6. 均线低位金叉 / 均线粘合（最大15分）
    # MA5上穿MA10（低位金叉）= 短期反转
    # 均线粘合 = 即将变盘
    # ------------------------------------------------------------------
    ma_cross_score = 0
    ma5 = _safe_ma(latest, 'MA5')
    ma10 = _safe_ma(latest, 'MA10')
    ma20 = _safe_ma(latest, 'MA20')

    if ma5 is not None and ma10 is not None:
        # MA5 上穿 MA10（低位金叉）
        if len(df) >= 2:
            prev = df.iloc[-2]
            prev_ma5 = _safe_ma(prev, 'MA5')
            prev_ma10 = _safe_ma(prev, 'MA10')
            if prev_ma5 is not None and prev_ma10 is not None:
                if ma5 > ma10 and prev_ma5 <= prev_ma10:
                    # 确认是"低位"金叉（价格在 MA20 以下或接近）
                    if ma20 is not None and price <= ma20 * 1.02:
                        ma_cross_score = 12
                        signals.append('MA5低位金叉MA10（短期反转信号）')
                    else:
                        ma_cross_score = 6
                        signals.append('MA5金叉MA10')

        # MA5 连续回升（从下行转上行）
        if ma_cross_score == 0 and len(df) >= 5:
            ma5_vals = df['MA5'].tail(5).tolist()
            ma5_up = 0
            for i in range(len(ma5_vals) - 1, 0, -1):
                if not (np.isnan(ma5_vals[i]) or np.isnan(ma5_vals[i-1])):
                    if ma5_vals[i] > ma5_vals[i-1]:
                        ma5_up += 1
                    else:
                        break
            if ma5_up >= 3 and price < ma20 if ma20 else False:
                ma_cross_score = 8
                signals.append(f'MA5连续{ma5_up}日回升（短线企稳）')
            elif ma5_up >= 2:
                ma_cross_score = 4

    # 均线粘合检测（MA5/MA10/MA20 间距很小 = 即将变盘）
    if ma5 is not None and ma10 is not None and ma20 is not None and ma20 > 0:
        spread = (max(ma5, ma10, ma20) - min(ma5, ma10, ma20)) / ma20 * 100
        if spread < 1.5:
            ma_cross_score = min(15, ma_cross_score + 5)
            signals.append(f'均线粘合（MA5/10/20间距{spread:.1f}%，即将变盘）')
        elif spread < 2.5:
            ma_cross_score = min(15, ma_cross_score + 3)

    ma_cross_score = min(15, ma_cross_score)
    score += ma_cross_score
    details['ma_cross'] = {'score': ma_cross_score}

    # ------------------------------------------------------------------
    # 7. 底部反转K线形态（最大10分）
    # 长下影线、十字星、早晨之星等底部K线
    # ------------------------------------------------------------------
    candle_score = 0
    body = abs(latest['收盘'] - latest['开盘'])
    total_range = latest['最高'] - latest['最低']
    lower_shadow = min(latest['收盘'], latest['开盘']) - latest['最低']

    if total_range > 0:
        lower_shadow_ratio = lower_shadow / total_range

        # 长下影线（下方有支撑）
        long_lower_shadow_count = 0
        for i in range(-3, 0):
            if abs(i) <= len(df):
                row = df.iloc[i]
                r_total = row['最高'] - row['最低']
                if r_total > 0:
                    r_lower_shadow = (min(row['收盘'], row['开盘']) - row['最低']) / r_total
                    if r_lower_shadow > 0.5:
                        long_lower_shadow_count += 1

        if long_lower_shadow_count >= 2:
            candle_score = 10
            signals.append(f'近3日出现{long_lower_shadow_count}根长下影线（下方支撑强）')
        elif lower_shadow_ratio > 0.6 and total_range / price * 100 > 1:
            candle_score = 6
            signals.append('今日长下影线（探底回升）')

        # 十字星（在低位出现 = 多空转换）
        if candle_score < 6:
            body_ratio = body / total_range if total_range > 0 else 0
            if body_ratio < 0.15 and total_range / price * 100 > 1.5:
                rsi_low = isinstance(rsi_val, (int, float)) and not np.isnan(rsi_val) and rsi_val < 40
                if rsi_low:
                    candle_score = max(candle_score, 5)
                    signals.append('低位十字星（多空力量转换）')

    # 早晨之星形态（3根K线：大阴 + 小十字 + 大阳）
    if candle_score < 8 and len(df) >= 3:
        d3 = df.iloc[-3]  # 第一天：大阴线
        d2 = df.iloc[-2]  # 第二天：小K线
        d1 = latest       # 第三天：大阳线

        d3_body = d3['开盘'] - d3['收盘']  # 阴线 body > 0
        d3_range = d3['最高'] - d3['最低']
        d2_body = abs(d2['收盘'] - d2['开盘'])
        d2_range = d2['最高'] - d2['最低']
        d1_body = d1['收盘'] - d1['开盘']  # 阳线 body > 0

        if (d3_range > 0 and d2_range > 0 and
            d3_body / d3_range > 0.5 and d3_body > 0 and  # 大阴线
            d2_body / d2_range < 0.3 and                    # 小K线
            d1_body > 0 and d1_body / total_range > 0.5):   # 大阳线
            candle_score = min(10, candle_score + 8)
            signals.append('早晨之星形态（大阴+十字+大阳，经典底部反转）')

    score += candle_score
    details['candle'] = {'score': candle_score}

    # ------------------------------------------------------------------
    # 综合判定
    # ------------------------------------------------------------------
    score = min(100, score)

    if score >= 60:
        level = '强信号'
    elif score >= 40:
        level = '中等'
    elif score >= 25:
        level = '弱信号'
    else:
        level = '无信号'

    return {
        'score': score,
        'signals': signals,
        'level': level,
        'is_bottoming': score >= 40,
        'details': details,
    }
