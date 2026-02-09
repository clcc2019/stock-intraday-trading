#!/usr/bin/env python3
"""
图表数据生成脚本
生成 stock_data.json 供固定 HTML 页面渲染

用法:
  python3 generate_chart_data.py 600276          # 分析恒瑞医药
  python3 generate_chart_data.py 600519 --days 120  # 最近120个交易日
  python3 generate_chart_data.py 002594 --open    # 生成后自动打开浏览器

输出:
  chart/stock_data.json — K线、均线、MACD、KDJ、买卖信号等全量数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os
import warnings
import argparse
import webbrowser
import http.server
import threading

warnings.filterwarnings('ignore')

# 导入统一数据源和公共技术指标
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_source import DataSource
from technical import calculate_all_indicators


STOCK_NAME_PRESET = {
    '600519': '贵州茅台', '002594': '比亚迪', '600276': '恒瑞医药',
    '300750': '宁德时代', '000858': '五粮液', '601318': '中国平安',
    '600036': '招商银行', '000333': '美的集团', '600900': '长江电力',
    '601012': '隆基绿能', '002475': '立讯精密', '300059': '东方财富',
    '600893': '航发动力', '600482': '中国动力', '002028': '思源电气',
    '002415': '海康威视', '600406': '国电南瑞', '601872': '招商轮船',
}


def fetch_daily_data(stock_code, days=400):
    """获取日线数据（使用统一 DataSource）"""
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
    """获取股票名称"""
    if stock_code in STOCK_NAME_PRESET:
        return STOCK_NAME_PRESET[stock_code]
    return f'股票{stock_code}'


def calc_indicators(df):
    """计算全部技术指标（使用公共模块）"""
    return calculate_all_indicators(df)


# ============================================================
# 买卖信号检测
# ============================================================
def detect_signals(df):
    """检测买入/卖出信号点"""
    buy_points = []
    sell_points = []

    if len(df) < 30:
        return buy_points, sell_points

    for i in range(20, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i - 1]
        date = str(cur['日期'])[:10]
        price = cur['收盘']

        ma5  = cur.get('MA5', np.nan)
        ma10 = cur.get('MA10', np.nan)
        ma20 = cur.get('MA20', np.nan)
        ma60 = cur.get('MA60', np.nan)
        ma20_slope = cur.get('MA20_slope', 0)
        if isinstance(ma20_slope, float) and np.isnan(ma20_slope):
            ma20_slope = 0

        dif  = cur.get('DIF', np.nan)
        dea  = cur.get('DEA', np.nan)
        p_dif = prev.get('DIF', np.nan)
        p_dea = prev.get('DEA', np.nan)
        k_val = cur.get('K', np.nan)
        d_val = cur.get('D', np.nan)
        p_k   = prev.get('K', np.nan)
        p_d   = prev.get('D', np.nan)
        j_val = cur.get('J', np.nan)

        # --- 买入信号 ---
        # 1. 趋势+均线：价格回踩MA20且MA20上行
        if (not np.isnan(ma20) and not np.isnan(ma60)
                and ma20_slope > 0
                and abs(price - ma20) / ma20 * 100 < 2
                and price > ma60
                and prev['收盘'] <= ma20 * 1.01):
            buy_points.append({'date': date, 'price': round(price, 2), 'reason': '回踩MA20'})
            continue

        # 2. MACD金叉 + 趋势向上
        if (not np.isnan(dif) and not np.isnan(dea)
                and dif > dea and p_dif <= p_dea
                and not np.isnan(ma20) and ma20_slope > 0):
            buy_points.append({'date': date, 'price': round(price, 2), 'reason': 'MACD金叉'})
            continue

        # 3. KDJ低位金叉(J<30)
        if (not np.isnan(k_val) and not np.isnan(d_val)
                and k_val > d_val and p_k <= p_d
                and j_val < 30):
            buy_points.append({'date': date, 'price': round(price, 2), 'reason': 'KDJ低位金叉'})
            continue

        # 4. 钟摆超卖反弹：偏离MA20超过-8%
        if (not np.isnan(ma20) and (price - ma20) / ma20 * 100 < -8
                and price > prev['收盘']):
            buy_points.append({'date': date, 'price': round(price, 2), 'reason': '钟摆超卖'})
            continue

        # --- 卖出信号 ---
        # 1. MACD死叉 + 趋势转弱
        if (not np.isnan(dif) and not np.isnan(dea)
                and dif < dea and p_dif >= p_dea
                and ma20_slope < 0):
            sell_points.append({'date': date, 'price': round(price, 2), 'reason': 'MACD死叉'})
            continue

        # 2. KDJ高位死叉(J>80)
        if (not np.isnan(k_val) and not np.isnan(d_val)
                and k_val < d_val and p_k >= p_d
                and j_val > 80):
            sell_points.append({'date': date, 'price': round(price, 2), 'reason': 'KDJ高位死叉'})
            continue

        # 3. 钟摆超买：偏离MA20超过+10%
        if (not np.isnan(ma20) and (price - ma20) / ma20 * 100 > 10
                and price < prev['收盘']):
            sell_points.append({'date': date, 'price': round(price, 2), 'reason': '钟摆超买'})
            continue

        # 4. 跌破MA20且MA20开始下行
        if (not np.isnan(ma20) and price < ma20
                and prev['收盘'] >= ma20 and ma20_slope < -0.5):
            sell_points.append({'date': date, 'price': round(price, 2), 'reason': '跌破MA20'})
            continue

    return buy_points, sell_points


# ============================================================
# 评分体系（与 analyze_stock_simple.py 一致）
# ============================================================
def calc_scores(df):
    """计算综合评分"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest['收盘']

    scores = {}

    # --- 趋势方向 (满分6) ---
    ma5, ma10, ma20, ma60 = latest['MA5'], latest['MA10'], latest['MA20'], latest['MA60']
    ma120 = latest.get('MA120', np.nan)

    t_buy, t_sell = 0, 0
    t_desc = []

    perfect = ma5 > ma10 > ma20 > ma60
    if perfect:
        t_buy += 3
        t_desc.append('完美多头排列')
    elif ma5 > ma10 > ma20:
        t_buy += 2
        t_desc.append('强势多头')
    elif ma5 > ma10:
        t_buy += 1
        t_desc.append('短期偏多')
    elif ma5 < ma10 < ma20 < ma60:
        t_sell += 3
        t_desc.append('完美空头排列')
    elif ma5 < ma10 < ma20:
        t_sell += 2
        t_desc.append('弱势空头')
    else:
        t_desc.append('均线交织')

    # 高低点
    if len(df) >= 40:
        highs_20 = df['最高'].iloc[-20:].max()
        highs_40 = df['最高'].iloc[-40:-20].max()
        lows_20 = df['最低'].iloc[-20:].min()
        lows_40 = df['最低'].iloc[-40:-20].min()
        if highs_20 > highs_40 and lows_20 > lows_40:
            t_buy += 2
            t_desc.append('高低点递增')
        elif highs_20 < highs_40 and lows_20 < lows_40:
            t_sell += 2
            t_desc.append('高低点递减')

    # MA120方向
    if not np.isnan(ma120):
        if price > ma120:
            t_buy += 1
            t_desc.append('站上MA120')
        else:
            t_sell += 1
            t_desc.append('低于MA120')

    scores['trend'] = {
        'score': min(6, t_buy), 'max': 6,
        'desc': ' / '.join(t_desc) if t_desc else '中性',
        'buy': min(6, t_buy), 'sell': min(6, t_sell),
    }

    # --- 钟摆位置 (满分5) ---
    dev_ma20 = (price - ma20) / ma20 * 100
    dev_ma60 = (price - ma60) / ma60 * 100 if ma60 > 0 else 0
    dev_ma120 = (price - ma120) / ma120 * 100 if not np.isnan(ma120) and ma120 > 0 else None

    p_buy, p_sell = 0, 0
    p_desc = []

    if abs(dev_ma20) <= 3:
        p_buy += 2
        p_desc.append(f'MA20附近({dev_ma20:+.1f}%)')
    elif dev_ma20 > 10:
        p_sell += 2
        p_desc.append(f'远离MA20({dev_ma20:+.1f}%)')
    elif dev_ma20 > 5:
        p_sell += 1
        p_desc.append(f'偏离MA20({dev_ma20:+.1f}%)')
    elif dev_ma20 < -8:
        p_buy += 2
        p_desc.append(f'超卖({dev_ma20:+.1f}%)')
    elif dev_ma20 < -5:
        p_buy += 1
        p_desc.append(f'偏低({dev_ma20:+.1f}%)')

    if dev_ma60 > 15:
        p_sell += 2
        p_desc.append(f'远离MA60({dev_ma60:+.1f}%)')
    elif -3 <= dev_ma60 <= 5:
        p_buy += 1
        p_desc.append(f'MA60附近({dev_ma60:+.1f}%)')
    elif dev_ma60 < -10:
        p_buy += 2
        p_desc.append(f'超卖MA60({dev_ma60:+.1f}%)')

    scores['pendulum'] = {
        'score': min(5, p_buy), 'max': 5,
        'desc': ' / '.join(p_desc) if p_desc else '中性',
        'buy': min(5, p_buy), 'sell': min(5, p_sell),
    }

    # --- 趋势强度 (满分4) ---
    ma20_slope = latest.get('MA20_slope', 0)
    if isinstance(ma20_slope, float) and np.isnan(ma20_slope):
        ma20_slope = 0

    s_buy, s_sell = 0, 0
    s_desc = []
    if ma20_slope > 2:
        s_buy += 2; s_desc.append(f'MA20加速上行({ma20_slope:+.1f}%)')
    elif ma20_slope > 0:
        s_buy += 1; s_desc.append(f'MA20上行({ma20_slope:+.1f}%)')
    elif ma20_slope < -2:
        s_sell += 2; s_desc.append(f'MA20加速下行({ma20_slope:+.1f}%)')
    elif ma20_slope < 0:
        s_sell += 1; s_desc.append(f'MA20下行({ma20_slope:+.1f}%)')

    price_20d_ago = df.iloc[-20]['收盘'] if len(df) >= 20 else price
    change_20d = (price - price_20d_ago) / price_20d_ago * 100
    if change_20d > 10:
        s_buy += 2; s_desc.append(f'20日强势(+{change_20d:.1f}%)')
    elif change_20d > 3:
        s_buy += 1; s_desc.append(f'20日偏强(+{change_20d:.1f}%)')
    elif change_20d < -10:
        s_sell += 2; s_desc.append(f'20日弱势({change_20d:+.1f}%)')
    elif change_20d < -3:
        s_sell += 1; s_desc.append(f'20日偏弱({change_20d:+.1f}%)')

    scores['strength'] = {
        'score': min(4, s_buy), 'max': 4,
        'desc': ' / '.join(s_desc) if s_desc else '中性',
        'buy': min(4, s_buy), 'sell': min(4, s_sell),
    }

    # --- 量价关系 (满分3) ---
    vol_ratio = latest['成交量'] / latest['VOL_MA5'] if latest['VOL_MA5'] > 0 else 1
    change_pct = (price - prev['收盘']) / prev['收盘'] * 100

    v_buy, v_sell = 0, 0
    v_desc = []
    if vol_ratio > 1.5 and change_pct > 0:
        v_buy += 2; v_desc.append(f'放量上涨(量比{vol_ratio:.1f})')
    elif vol_ratio > 1.5 and change_pct < 0:
        v_sell += 2; v_desc.append(f'放量下跌(量比{vol_ratio:.1f})')
    elif vol_ratio < 0.5 and change_pct < 0:
        v_buy += 1; v_desc.append(f'缩量止跌(量比{vol_ratio:.1f})')
    else:
        v_desc.append(f'量比{vol_ratio:.1f}')

    scores['volume'] = {
        'score': min(3, v_buy), 'max': 3,
        'desc': ' / '.join(v_desc) if v_desc else '中性',
        'buy': min(3, v_buy), 'sell': min(3, v_sell),
    }

    # --- 传统指标 (满分2) ---
    l_buy, l_sell = 0, 0
    l_desc = []
    macd_bull = latest['DIF'] > latest['DEA']
    if macd_bull and prev['DIF'] <= prev['DEA']:
        l_buy += 1; l_desc.append('MACD金叉')
    elif not macd_bull and prev['DIF'] >= prev['DEA']:
        l_sell += 1; l_desc.append('MACD死叉')
    elif macd_bull:
        l_desc.append('MACD多头')
    else:
        l_desc.append('MACD空头')

    j_val = latest['J']
    k_val, d_val = latest['K'], latest['D']
    if k_val > d_val and prev['K'] <= prev['D'] and j_val < 30:
        l_buy += 1; l_desc.append('KDJ低位金叉')
    elif k_val < d_val and prev['K'] >= prev['D'] and j_val > 70:
        l_sell += 1; l_desc.append('KDJ高位死叉')
    elif j_val > 80:
        l_desc.append(f'KDJ超买J={j_val:.0f}')
    elif j_val < 20:
        l_desc.append(f'KDJ超卖J={j_val:.0f}')

    scores['traditional'] = {
        'score': min(2, l_buy), 'max': 2,
        'desc': ' / '.join(l_desc) if l_desc else '中性',
        'buy': min(2, l_buy), 'sell': min(2, l_sell),
    }

    # 综合
    total_buy = sum(s['buy'] for s in scores.values())
    total_sell = sum(s['sell'] for s in scores.values())

    is_uptrend = scores['trend']['buy'] >= 3
    is_downtrend = scores['trend']['sell'] >= 3

    if total_buy >= 14 and is_uptrend:
        rec = '🟢 强烈买入'
    elif total_buy >= 10 and is_uptrend:
        rec = '🟢 买入'
    elif total_sell >= 14 and is_downtrend:
        rec = '🔴 强烈卖出'
    elif total_sell >= 10:
        rec = '🔴 卖出'
    elif total_buy >= 7:
        rec = '🟡 可考虑买入'
    elif total_sell >= 7:
        rec = '🟠 可考虑卖出'
    else:
        rec = '⚪ 观望'

    if is_downtrend and '买入' in rec:
        rec = '⚪ 观望(趋势向下)'

    scores['total_buy'] = total_buy
    scores['total_sell'] = total_sell
    scores['recommendation'] = rec

    # 钟摆位置百分比（0=极度超卖, 50=中性, 100=极度超买）
    pend_pct = 50 + dev_ma20 * 2.5  # 简化映射
    pend_pct = max(0, min(100, pend_pct))

    return scores, {
        'position_pct': round(pend_pct, 1),
        'dev_ma20': round(dev_ma20, 2),
        'dev_ma60': round(dev_ma60, 2),
        'dev_ma120': round(dev_ma120, 2) if dev_ma120 is not None else None,
    }


# ============================================================
# 核心信号提取
# ============================================================
def extract_key_signals(df, scores):
    """提取当前关键信号"""
    signals = []
    latest = df.iloc[-1]
    price = latest['收盘']
    ma20 = latest['MA20']
    ma60 = latest['MA60']

    dev20 = (price - ma20) / ma20 * 100

    # 趋势信号
    if scores['trend']['buy'] >= 4:
        signals.append({'type': 'buy', 'text': f"趋势向上: {scores['trend']['desc']}"})
    elif scores['trend']['sell'] >= 4:
        signals.append({'type': 'sell', 'text': f"趋势向下: {scores['trend']['desc']}"})

    # 钟摆信号
    if dev20 > 10:
        signals.append({'type': 'sell', 'text': f'钟摆偏高: 偏离MA20 {dev20:+.1f}%，回归压力大'})
    elif dev20 < -8:
        signals.append({'type': 'buy', 'text': f'钟摆超卖: 偏离MA20 {dev20:+.1f}%，反弹动力大'})
    elif abs(dev20) <= 3:
        signals.append({'type': 'info', 'text': f'钟摆中性: 价格在MA20附近({dev20:+.1f}%)'})

    # MACD
    if 'MACD金叉' in scores['traditional']['desc']:
        signals.append({'type': 'buy', 'text': 'MACD金叉（辅助参考）'})
    elif 'MACD死叉' in scores['traditional']['desc']:
        signals.append({'type': 'sell', 'text': 'MACD死叉（辅助参考）'})

    # 量价
    if scores['volume']['buy'] >= 2:
        signals.append({'type': 'buy', 'text': f"量价配合: {scores['volume']['desc']}"})
    elif scores['volume']['sell'] >= 2:
        signals.append({'type': 'sell', 'text': f"量价异常: {scores['volume']['desc']}"})

    return signals


# ============================================================
# JSON 组装
# ============================================================
def build_json(df, stock_code, stock_name, display_days=120):
    """组装完整 JSON 数据"""
    # 截取显示区间
    df_display = df.tail(display_days).copy().reset_index(drop=True)

    dates = [str(d)[:10] for d in df_display['日期']]

    # OHLC: [open, close, low, high] — ECharts candlestick 格式
    ohlc = []
    for _, row in df_display.iterrows():
        ohlc.append([
            round(row['开盘'], 2), round(row['收盘'], 2),
            round(row['最低'], 2), round(row['最高'], 2),
        ])

    volumes = [int(row['成交量']) for _, row in df_display.iterrows()]

    # 均线
    ma_keys = ['MA5', 'MA10', 'MA20', 'MA60', 'MA120']
    ma_data = {}
    for mk in ma_keys:
        if mk in df_display.columns:
            vals = df_display[mk].tolist()
            ma_data[mk] = [round(v, 2) if not np.isnan(v) else None for v in vals]

    # MACD
    macd_data = {
        'dif': [round(v, 4) if not np.isnan(v) else None for v in df_display['DIF']],
        'dea': [round(v, 4) if not np.isnan(v) else None for v in df_display['DEA']],
        'hist': [round(v, 4) if not np.isnan(v) else None for v in df_display['MACD']],
    }

    # KDJ
    kdj_data = {
        'k': [round(v, 2) if not np.isnan(v) else None for v in df_display['K']],
        'd': [round(v, 2) if not np.isnan(v) else None for v in df_display['D']],
        'j': [round(v, 2) if not np.isnan(v) else None for v in df_display['J']],
    }

    # 买卖信号
    buy_pts, sell_pts = detect_signals(df)
    # 只保留显示区间内的信号
    min_date = dates[0] if dates else ''
    buy_pts = [p for p in buy_pts if p['date'] >= min_date]
    sell_pts = [p for p in sell_pts if p['date'] >= min_date]

    # 评分
    scores, pendulum = calc_scores(df)
    key_signals = extract_key_signals(df, scores)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest['收盘']
    change_pct = (price - prev['收盘']) / prev['收盘'] * 100

    # 关键价位
    ma20 = latest['MA20']
    ma60 = latest['MA60']
    ma120 = latest.get('MA120', np.nan)

    support = ma20
    if not np.isnan(ma60) and ma60 < support:
        support = ma60

    # 信息面板
    info_panels = [
        {
            'icon': '📍', 'title': '关键价位',
            'rows': [
                ['当前价格', f"¥{price:.2f}"],
                ['MA20(价值中枢)', f"¥{ma20:.2f}"],
                ['MA60(中期中枢)', f"¥{ma60:.2f}"],
                ['MA120(长期中枢)', f"¥{ma120:.2f}" if not np.isnan(ma120) else '--'],
                ['支撑位', f"¥{support:.2f}"],
                ['建议止损', f"¥{price * 0.97:.2f}"],
            ]
        },
        {
            'icon': '📊', 'title': '技术状态',
            'rows': [
                ['MACD(DIF/DEA)', f"{latest['DIF']:.3f} / {latest['DEA']:.3f}"],
                ['KDJ(K/D/J)', f"{latest['K']:.1f} / {latest['D']:.1f} / {latest['J']:.1f}"],
                ['RSI(14)', f"{latest['RSI']:.1f}" if not np.isnan(latest['RSI']) else '--'],
                ['MA20斜率', f"{latest.get('MA20_slope', 0):+.1f}%"],
                ['20日涨幅', f"{change_pct * 20:+.1f}%" if False else f"{((price - df.iloc[-20]['收盘']) / df.iloc[-20]['收盘'] * 100):+.1f}%" if len(df) >= 20 else '--'],
            ]
        },
    ]

    result = {
        'info': {
            'name': stock_name,
            'code': stock_code,
            'price': round(price, 2),
            'change_pct': round(change_pct, 2),
        },
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'kline': {
            'dates': dates,
            'ohlc': ohlc,
            'volumes': volumes,
            'ma': ma_data,
            'macd': macd_data,
            'kdj': kdj_data,
        },
        'scores': scores,
        'pendulum': pendulum,
        'signals': {
            'buy_points': buy_pts,
            'sell_points': sell_pts,
            'key_signals': key_signals,
        },
        'info_panels': info_panels,
    }

    return result


# ============================================================
# 本地 HTTP 服务（解决 file:// 跨域问题）
# ============================================================
def serve_and_open(chart_dir, port=8686):
    """启动本地HTTP服务并打开浏览器"""
    handler = http.server.SimpleHTTPRequestHandler

    class QuietHandler(handler):
        def log_message(self, format, *args):
            pass  # 静默日志

    os.chdir(chart_dir)
    server = http.server.HTTPServer(('127.0.0.1', port), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f'http://127.0.0.1:{port}/stock-chart.html'
    print(f"🌐 本地服务已启动: {url}")
    webbrowser.open(url)
    print("   按 Ctrl+C 停止服务")

    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()
        print("\n✅ 服务已停止")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='生成股票图表数据')
    parser.add_argument('code', type=str, help='股票代码，如 600276')
    parser.add_argument('--days', type=int, default=120, help='显示天数（默认120）')
    parser.add_argument('--open', action='store_true', help='生成后自动打开浏览器')

    args = parser.parse_args()
    stock_code = args.code.strip()

    # 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    skill_dir = os.path.dirname(script_dir)
    chart_dir = os.path.join(skill_dir, 'chart')
    os.makedirs(chart_dir, exist_ok=True)
    json_path = os.path.join(chart_dir, 'stock_data.json')

    print(f"\n📊 正在生成 {stock_code} 的图表数据...")

    # 获取数据
    stock_name = get_stock_name(stock_code)
    print(f"   股票名称: {stock_name}")

    df = fetch_daily_data(stock_code, days=max(400, args.days + 280))
    if df is None or df.empty:
        print("❌ 无法获取股票数据，请检查代码或网络")
        sys.exit(1)
    print(f"   获取到 {len(df)} 条日线数据")

    # 计算指标
    print("⏳ 计算技术指标...")
    df = calc_indicators(df)

    # 组装JSON
    print("📦 组装图表数据...")
    data = build_json(df, stock_code, stock_name, display_days=args.days)

    # 写入文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 数据已生成: chart/stock_data.json")
    print(f"   K线数据: {len(data['kline']['dates'])} 天")
    print(f"   买入信号: {len(data['signals']['buy_points'])} 个")
    print(f"   卖出信号: {len(data['signals']['sell_points'])} 个")
    print(f"   综合评分: 买{data['scores']['total_buy']} / 卖{data['scores']['total_sell']}")
    print(f"   操作建议: {data['scores']['recommendation']}")

    # 打开浏览器
    if args.open:
        serve_and_open(chart_dir)
    else:
        print(f"\n💡 查看图表:")
        print(f"   方法1: cd {chart_dir} && python3 -m http.server 8686")
        print(f"          然后打开 http://127.0.0.1:8686/stock-chart.html")
        print(f"   方法2: python3 scripts/generate_chart_data.py {stock_code} --open")


if __name__ == '__main__':
    main()
