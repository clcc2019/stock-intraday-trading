#!/usr/bin/env python3
"""
股票日内T+0做T分析工具
基于「顺大势逆小势」投资哲学

核心理念：
- 公设一：价格围绕价值波动（均线=价值中枢）
- 公设二：钟摆式过度波动（偏离越远，回归力越大）
- 核心原则：顺大势（周线/日线趋势方向），逆小势（日内分时回调/反弹）
- 均线=玄铁重剑，MACD/KDJ仅作可选参考
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# 导入数据源适配层和公共技术指标
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_source import DataSource
from technical import (
    calculate_ma, calculate_macd, calculate_kdj, calculate_volume_ma,
    detect_highs_lows, analyze_ma_alignment, calculate_pendulum, _safe_ma,
)


class IntradayT0Analyzer:
    """日内T+0做T分析器 — 基于趋势+均线+钟摆模型"""

    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.df_daily = None   # 日K线数据
        self.df_weekly = None  # 周K线数据
        self.df_minute = None  # 分时数据
        self.data = {}
        self.market_data = {}

    def fetch_data(self):
        """获取股票数据（使用 baostock，扩展至300+天，支持MA120/MA250）"""
        try:
            print(f"📊 正在获取 {self.stock_code} 的数据...")

            # 1. 获取日K线数据（300+天，计算MA120/MA250）
            end_date = datetime.now()
            start_date = end_date - timedelta(days=400)

            self.df_daily = DataSource.get_stock_hist(
                stock_code=self.stock_code,
                start_date=start_date,
                end_date=end_date,
                adjust='qfq',
                period='daily'
            )

            if self.df_daily is None or self.df_daily.empty:
                print(f"❌ 无法获取日K线数据")
                return False

            # 2. 获取周K线数据（判断周级别趋势）
            try:
                self.df_weekly = DataSource.get_stock_hist(
                    stock_code=self.stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    adjust='qfq',
                    period='weekly'
                )
            except:
                self.df_weekly = None

            # 3. 获取实时分时数据（今日5分钟数据）
            try:
                today = datetime.now().strftime('%Y-%m-%d')
                self.df_minute = DataSource.get_stock_hist_minute(
                    stock_code=self.stock_code,
                    start_date=today,
                    end_date=today,
                    adjust='qfq',
                    period='5'
                )

                if self.df_minute is not None and not self.df_minute.empty:
                    print(f"✅ 获取到 {len(self.df_minute)} 条分时数据")
                else:
                    print("⚠️ 今日暂无分时数据（可能未开盘或已收盘）")
                    self.df_minute = None
            except Exception as e:
                print(f"⚠️ 分时数据获取失败: {e}")
                self.df_minute = None

            # 4. 基本信息
            latest_daily = self.df_daily.iloc[-1]

            self.data = {
                'name': f'股票{self.stock_code}',
                'current_price': latest_daily['收盘'],
                'change_pct': ((latest_daily['收盘'] - self.df_daily.iloc[-2]['收盘']) / self.df_daily.iloc[-2]['收盘']) * 100,
                'high': latest_daily['最高'],
                'low': latest_daily['最低'],
                'open': latest_daily['开盘'],
                'volume': latest_daily['成交量'],
            }

            # 如果有分时数据，更新为最新价格
            if self.df_minute is not None and not self.df_minute.empty:
                latest_minute = self.df_minute.iloc[-1]
                self.data['current_price'] = latest_minute['收盘']
                self.data['high'] = self.df_minute['最高'].max()
                self.data['low'] = self.df_minute['最低'].min()
                self.data['open'] = self.df_minute.iloc[0]['开盘']
                self.data['change_pct'] = ((self.data['current_price'] - self.data['open']) / self.data['open']) * 100

            # baostock 数据中已包含股票代码，名称暂时保持默认
            pass

            # 计算技术指标
            self.calculate_indicators()

            # 获取市场数据
            self.fetch_market_data()

            return True

        except Exception as e:
            print(f"❌ 数据获取失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def fetch_market_data(self):
        """获取市场数据（使用 baostock）"""
        try:
            sz_df = DataSource.get_stock_hist('000001', period='daily')
            if sz_df is not None and not sz_df.empty and len(sz_df) >= 2:
                latest_sz = sz_df.iloc[-1]
                prev_sz = sz_df.iloc[-2]
                self.market_data['上证指数'] = {
                    'price': latest_sz['收盘'],
                    'change_pct': ((latest_sz['收盘'] - prev_sz['收盘']) / prev_sz['收盘']) * 100
                }
        except:
            pass

    def calculate_indicators(self):
        """计算技术指标 — 使用公共模块"""
        df = self.df_daily

        # 日线指标（均线、MACD、KDJ、成交量均线）
        calculate_ma(df)
        calculate_macd(df)
        calculate_kdj(df)
        calculate_volume_ma(df)

        # 周线均线
        if self.df_weekly is not None and not self.df_weekly.empty:
            calculate_ma(self.df_weekly, windows=[5, 10, 20])
            for w in [5, 10, 20]:
                if f'MA{w}' in self.df_weekly.columns:
                    self.df_weekly[f'W_MA{w}'] = self.df_weekly[f'MA{w}']

        # 分时指标
        if self.df_minute is not None and not self.df_minute.empty and len(self.df_minute) >= 5:
            # 分时均价线（VWAP）— 做T的核心参考线
            self.df_minute['VWAP'] = (self.df_minute['成交额'].cumsum() / self.df_minute['成交量'].cumsum())

            # 分时量能
            self.df_minute['VOL_MA'] = self.df_minute['成交量'].rolling(window=5).mean()

    def analyze_multi_timeframe_trend(self):
        """多级别趋势分析 — 核心分析方法"""
        latest = self.df_daily.iloc[-1]
        price = self.data['current_price']
        result = {}

        # === 日线趋势分析（使用公共模块）===
        alignment_info = analyze_ma_alignment(latest, price)
        ma_alignment = alignment_info['desc']
        alignment_score = alignment_info['score']
        ma_vals = alignment_info['ma_values']
        ma5 = ma_vals['MA5']
        ma10 = ma_vals['MA10']
        ma20 = ma_vals['MA20']
        ma60 = ma_vals['MA60']
        ma120 = ma_vals['MA120']
        ma250 = ma_vals['MA250']

        # 均线方向（斜率）
        ma20_slope = latest.get('MA20_slope', 0)
        if np.isnan(ma20_slope):
            ma20_slope = 0
        ma60_slope = latest.get('MA60_slope', 0)
        if np.isnan(ma60_slope):
            ma60_slope = 0

        if ma20_slope > 1:
            ma20_dir = '↑ 加速上升'
        elif ma20_slope > 0:
            ma20_dir = '↗ 缓慢上升'
        elif ma20_slope > -1:
            ma20_dir = '→ 走平'
        else:
            ma20_dir = '↓ 下行'

        # 趋势定义（高低点递增/递减，使用公共模块）
        hl = detect_highs_lows(self.df_daily)
        highs_rising = hl['highs_rising']
        lows_rising = hl['lows_rising']
        highs_falling = hl['highs_falling']
        lows_falling = hl['lows_falling']

        if highs_rising and lows_rising:
            trend_def = '标准上升趋势'
            trend_def_score = 2
        elif highs_rising or lows_rising:
            trend_def = '疑似上升趋势'
            trend_def_score = 1
        elif highs_falling and lows_falling:
            trend_def = '标准下降趋势'
            trend_def_score = -2
        elif highs_falling or lows_falling:
            trend_def = '疑似下降趋势'
            trend_def_score = -1
        else:
            trend_def = '震荡整理'
            trend_def_score = 0

        # === 周线趋势（如有）===
        weekly_trend = '数据不足'
        weekly_score = 0
        if self.df_weekly is not None and not self.df_weekly.empty and len(self.df_weekly) >= 10:
            wl = self.df_weekly.iloc[-1]
            w_ma5 = wl.get('W_MA5', np.nan)
            w_ma10 = wl.get('W_MA10', np.nan)
            w_ma20 = wl.get('W_MA20', np.nan)
            if not any(np.isnan(x) for x in [w_ma5, w_ma10, w_ma20] if isinstance(x, float)):
                if w_ma5 > w_ma10 > w_ma20:
                    weekly_trend = '↑ 多头排列'
                    weekly_score = 2
                elif w_ma5 > w_ma10:
                    weekly_trend = '↗ 偏多'
                    weekly_score = 1
                elif w_ma5 < w_ma10 < w_ma20:
                    weekly_trend = '↓ 空头排列'
                    weekly_score = -2
                elif w_ma5 < w_ma10:
                    weekly_trend = '↘ 偏空'
                    weekly_score = -1
                else:
                    weekly_trend = '→ 震荡'
                    weekly_score = 0

        # === 趋势强度综合评分（0-10）===
        strength = 0
        # 均线排列（0-3）
        strength += max(0, alignment_score)
        # 趋势定义（0-2）
        strength += max(0, trend_def_score)
        # MA20斜率（0-2）
        if ma20_slope > 3:
            strength += 2
        elif ma20_slope > 0.5:
            strength += 1
        # 周线趋势（0-2）
        strength += max(0, weekly_score)
        # 价格在MA120以上（0-1）
        if ma120 and price > ma120:
            strength += 1

        strength = min(10, strength)

        # === 日线趋势方向综合判断 ===
        if alignment_score >= 2 and trend_def_score >= 1:
            daily_direction = '↑ 上升'
        elif alignment_score >= 1 and ma20_slope > 0:
            daily_direction = '↗ 偏多'
        elif alignment_score <= -2:
            daily_direction = '↓ 下降'
        elif alignment_score <= -1:
            daily_direction = '↘ 偏空'
        else:
            daily_direction = '→ 震荡'

        result['daily'] = {
            'direction': daily_direction,
            'alignment': ma_alignment,
            'alignment_score': alignment_score,
            'trend_def': trend_def,
            'trend_def_score': trend_def_score,
            'highs_rising': highs_rising,
            'lows_rising': lows_rising,
            'ma20_dir': ma20_dir,
            'ma20_slope': ma20_slope,
            'ma60_slope': ma60_slope,
        }
        result['weekly'] = {
            'trend': weekly_trend,
            'score': weekly_score,
        }
        result['strength'] = strength
        result['ma_values'] = {
            'MA5': ma5, 'MA10': ma10, 'MA20': ma20,
            'MA60': ma60, 'MA120': ma120, 'MA250': ma250,
        }

        return result

    def analyze_pendulum_position(self):
        """钟摆位置分析（使用公共模块）"""
        price = self.data['current_price']
        latest = self.df_daily.iloc[-1]
        ma_values = {
            'MA20': _safe_ma(latest, 'MA20'),
            'MA60': _safe_ma(latest, 'MA60'),
            'MA120': _safe_ma(latest, 'MA120'),
            'MA250': _safe_ma(latest, 'MA250'),
        }
        return calculate_pendulum(price, ma_values)

    def analyze_intraday_t0(self):
        """日内T+0策略分析 — 以「顺大势逆小势」为核心"""
        # 多级别趋势
        trend = self.analyze_multi_timeframe_trend()
        # 钟摆位置
        pendulum = self.analyze_pendulum_position()

        current_price = self.data['current_price']
        latest_daily = self.df_daily.iloc[-1]

        result = {
            'trend': trend,
            'pendulum': pendulum,
            'has_intraday': False,
            'current_time': datetime.now().strftime('%H:%M'),
            'trading_opportunities': [],
            'key_levels': {},
            'strategy': {},
            't0_direction': {},
        }

        # === 顺大势逆小势：确定做T方向 ===
        strength = trend['strength']
        daily_dir = trend['daily']['direction']
        weekly_score = trend['weekly']['score']
        dev_ma20 = pendulum['MA20']['deviation']

        # 大势判断
        if strength >= 6 and '上' in daily_dir or '多' in daily_dir:
            major_trend = '看多'
            t0_bias = '偏多做T（低买为主，逢日内回踩VWAP/均线买入）'
        elif strength <= 3 and ('下' in daily_dir or '空' in daily_dir):
            major_trend = '看空'
            t0_bias = '⚠️ 不建议做T买入（趋势向下，做T风险极高）'
        elif '震荡' in daily_dir:
            major_trend = '震荡'
            t0_bias = '双向做T（区间操作，高卖低买）'
        else:
            major_trend = '不明确'
            t0_bias = '谨慎做T（趋势不明确，轻仓操作）'

        result['t0_direction'] = {
            'major_trend': major_trend,
            'bias': t0_bias,
            'strength': strength,
        }

        # === 关键价位（基于均线）===
        ma20 = latest_daily['MA20']
        ma60 = latest_daily.get('MA60', np.nan)
        ma5 = latest_daily['MA5']
        ma10 = latest_daily['MA10']

        # 支撑位：以均线为核心
        supports = []
        if not np.isnan(ma20):
            supports.append(('MA20', ma20))
        if not np.isnan(ma60) if isinstance(ma60, float) else ma60 is not None:
            supports.append(('MA60', ma60))
        supports.append(('昨日低点', self.df_daily.iloc[-2]['最低']))

        # 压力位
        resistances = []
        resistances.append(('昨日高点', self.df_daily.iloc[-2]['最高']))
        if self.data['high'] > self.df_daily.iloc[-2]['最高']:
            resistances.append(('今日高点', self.data['high']))

        # 找到最近的支撑和压力
        nearest_support = min(supports, key=lambda x: abs(current_price - x[1]) if x[1] < current_price else float('inf'))
        nearest_resistance = min(resistances, key=lambda x: abs(x[1] - current_price) if x[1] > current_price else float('inf'))

        result['key_levels'] = {
            'supports': supports,
            'resistances': resistances,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'current': current_price,
            'ma20': ma20 if not np.isnan(ma20) else None,
        }

        # === 日内分时分析 ===
        if self.df_minute is not None and not self.df_minute.empty and len(self.df_minute) >= 5:
            result['has_intraday'] = True
            latest_minute = self.df_minute.iloc[-1]

            # VWAP — 日内价值中枢
            vwap = latest_minute['VWAP']

            # 日内波动幅度
            intraday_range = ((self.data['high'] - self.data['low']) / self.data['open']) * 100

            # 当前价格相对VWAP位置
            dev_vwap = (current_price - vwap) / vwap * 100

            # 量能分析
            current_vol = latest_minute['成交量']
            avg_vol = latest_minute['VOL_MA'] if not np.isnan(latest_minute['VOL_MA']) else current_vol
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1

            result['intraday'] = {
                'vwap': vwap,
                'dev_vwap': dev_vwap,
                'intraday_range': intraday_range,
                'vol_ratio': vol_ratio,
            }

            # === 生成交易机会（基于趋势+均线+钟摆）===
            opportunities = []

            # ============================================================
            # 核心策略1: 顺大势+回踩VWAP做T
            # ============================================================
            if major_trend == '看多':
                if dev_vwap < -0.3:
                    # 大势看多 + 日内价格在VWAP下方 = 买入做T
                    conf = '高' if strength >= 7 else '中'
                    opportunities.append({
                        'type': '买入',
                        'strategy': '📈 顺大势+回踩VWAP买入',
                        'price': f"{current_price:.2f}",
                        'target': f"{vwap * 1.003:.2f}",
                        'reason': f'大势看多(强度{strength}/10) + 日内价格低于VWAP({dev_vwap:+.1f}%)，钟摆回摆买入',
                        'confidence': conf,
                        'profit_potential': '+1.0-2.0%'
                    })
                elif dev_vwap > 0.5:
                    # 大势看多 + 日内价格在VWAP上方偏高 = 卖出做T（逆小势）
                    opportunities.append({
                        'type': '卖出',
                        'strategy': '📉 逆小势+偏离VWAP卖出',
                        'price': f"{current_price:.2f}",
                        'target': f"{vwap:.2f}",
                        'reason': f'大势看多但日内偏高(VWAP+{dev_vwap:.1f}%)，逆小势做T卖出，等回踩再买',
                        'confidence': '中',
                        'profit_potential': '+0.5-1.5%'
                    })

            elif major_trend == '看空':
                if dev_vwap > 0.3:
                    opportunities.append({
                        'type': '卖出',
                        'strategy': '⚠️ 趋势偏弱+反弹卖出',
                        'price': f"{current_price:.2f}",
                        'target': f"{vwap * 0.997:.2f}",
                        'reason': f'大势偏空(强度{strength}/10) + 日内反弹至VWAP上方，卖出避险',
                        'confidence': '高',
                        'profit_potential': '避免损失'
                    })

            elif major_trend == '震荡':
                if dev_vwap < -0.5:
                    opportunities.append({
                        'type': '买入',
                        'strategy': '🔄 震荡区间低买',
                        'price': f"{current_price:.2f}",
                        'target': f"{vwap:.2f}",
                        'reason': f'震荡市 + 日内价格低于VWAP({dev_vwap:+.1f}%)，区间低买',
                        'confidence': '中',
                        'profit_potential': '+0.5-1.0%'
                    })
                elif dev_vwap > 0.5:
                    opportunities.append({
                        'type': '卖出',
                        'strategy': '🔄 震荡区间高卖',
                        'price': f"{current_price:.2f}",
                        'target': f"{vwap:.2f}",
                        'reason': f'震荡市 + 日内价格高于VWAP(+{dev_vwap:.1f}%)，区间高卖',
                        'confidence': '中',
                        'profit_potential': '+0.5-1.0%'
                    })

            # ============================================================
            # 核心策略2: 均线偏离度均值回归
            # ============================================================
            if dev_ma20 is not None:
                if dev_ma20 > 8 and major_trend != '看空':
                    opportunities.append({
                        'type': '卖出',
                        'strategy': '🔔 偏离MA20过大，均值回归卖出',
                        'price': f"{current_price:.2f}",
                        'target': f"{ma20 * 1.03:.2f}" if not np.isnan(ma20) else '均线附近',
                        'reason': f'价格偏离MA20达{dev_ma20:+.1f}%，绳子偏紧，有回归MA20压力',
                        'confidence': '高' if dev_ma20 > 10 else '中',
                        'profit_potential': f'+{abs(dev_ma20)*0.3:.1f}-{abs(dev_ma20)*0.5:.1f}%'
                    })
                elif dev_ma20 < -5 and major_trend == '看多':
                    opportunities.append({
                        'type': '买入',
                        'strategy': '🔔 回踩MA20附近，均值回归买入',
                        'price': f"{current_price:.2f}",
                        'target': f"{ma20:.2f}" if not np.isnan(ma20) else '均线附近',
                        'reason': f'大势看多但价格回踩至MA20附近({dev_ma20:+.1f}%)，钟摆回摆，买入时机',
                        'confidence': '高',
                        'profit_potential': f'+{abs(dev_ma20)*0.3:.1f}-{abs(dev_ma20)*0.5:.1f}%'
                    })

            # ============================================================
            # 辅助策略: 支撑/压力位
            # ============================================================
            for name, level in supports:
                if current_price <= level * 1.005 and current_price >= level * 0.995:
                    opportunities.append({
                        'type': '买入',
                        'strategy': f'🛡️ 触及{name}支撑买入',
                        'price': f"{level:.2f}",
                        'target': f"{(level + nearest_resistance[1]) / 2:.2f}",
                        'reason': f'价格触及{name}(¥{level:.2f})支撑位',
                        'confidence': '中' if major_trend == '看多' else '低',
                        'profit_potential': '+1.0-2.0%'
                    })
                    break

            for name, level in resistances:
                if current_price >= level * 0.995:
                    opportunities.append({
                        'type': '卖出',
                        'strategy': f'⚡ 触及{name}压力卖出',
                        'price': f"{level:.2f}",
                        'target': f"{(nearest_support[1] + level) / 2:.2f}",
                        'reason': f'价格触及{name}(¥{level:.2f})压力位',
                        'confidence': '中',
                        'profit_potential': '+1.0-2.0%'
                    })
                    break

            # ============================================================
            # 辅助策略: 放量突破（需趋势配合）
            # ============================================================
            if vol_ratio > 2.0 and current_price > vwap and major_trend == '看多':
                opportunities.append({
                    'type': '买入',
                    'strategy': '🚀 趋势看多+放量突破',
                    'price': f"{current_price:.2f}",
                    'target': f"{current_price * 1.025:.2f}",
                    'reason': f'大势看多+放量{vol_ratio:.1f}倍+价格在VWAP上方，强势做T',
                    'confidence': '高',
                    'profit_potential': '+2.0-3.0%'
                })
            elif vol_ratio > 2.0 and current_price < vwap and major_trend != '看多':
                opportunities.append({
                    'type': '卖出',
                    'strategy': '⚠️ 放量下跌避险',
                    'price': f"{current_price:.2f}",
                    'target': '观望',
                    'reason': f'放量{vol_ratio:.1f}倍+价格在VWAP下方，风险信号',
                    'confidence': '高',
                    'profit_potential': '避免损失'
                })

            # ============================================================
            # 辅助策略: 时间窗口
            # ============================================================
            current_hour = int(datetime.now().strftime('%H'))
            if 9 <= current_hour < 11 and self.data['change_pct'] < -2 and major_trend == '看多':
                opportunities.append({
                    'type': '买入',
                    'strategy': '🌅 大势看多+早盘急跌抄底',
                    'price': f"{self.data['low']:.2f}",
                    'target': f"{vwap:.2f}",
                    'reason': '大势看多但早盘恐慌杀跌，钟摆过度向下，回归机会',
                    'confidence': '中',
                    'profit_potential': '+2.0-4.0%'
                })
            elif 14 <= current_hour < 15 and self.data['change_pct'] > 3:
                opportunities.append({
                    'type': '卖出',
                    'strategy': '🌆 午后大涨锁利',
                    'price': f"{current_price:.2f}",
                    'target': f"{vwap:.2f}",
                    'reason': '午后大涨，钟摆过度向上，锁定利润',
                    'confidence': '高',
                    'profit_potential': '锁定当日利润'
                })

            # 趋势弱势时降低买入置信度
            if major_trend == '看空':
                for opp in opportunities:
                    if opp['type'] == '买入':
                        opp['confidence'] = '很低'
                        opp['reason'] += ' ⚠️大势偏空，做T买入风险极高'

            result['trading_opportunities'] = opportunities

            # T+0策略类型
            if major_trend == '看空':
                result['strategy'] = {
                    'type': '⚠️ 趋势偏弱',
                    'desc': '大势向下，做T买入风险高',
                    'method': '建议仅做卖出操作，或暂停做T等待趋势企稳'
                }
            elif intraday_range > 4:
                result['strategy'] = {
                    'type': '🎯 高波动T+0',
                    'desc': f'日内波动{intraday_range:.1f}%，适合多次T+0',
                    'method': '建议分批操作：1/3仓位做T，2-3次交易'
                }
            elif intraday_range > 2:
                result['strategy'] = {
                    'type': '📊 常规T+0',
                    'desc': f'日内波动{intraday_range:.1f}%，适合1-2次T',
                    'method': '建议1/4仓位做T，1-2次交易'
                }
            else:
                result['strategy'] = {
                    'type': '💤 低波动观望',
                    'desc': f'日内波动{intraday_range:.1f}%，不适合做T',
                    'method': '建议观望，等待更好机会'
                }
        else:
            # 无分时数据，基于日线给简单建议
            if major_trend == '看多' and dev_ma20 is not None and dev_ma20 < 3:
                result['trading_opportunities'].append({
                    'type': '买入',
                    'strategy': '趋势看多+接近MA20',
                    'price': f"{ma20:.2f}" if not np.isnan(ma20) else f"{current_price:.2f}",
                    'target': f"{current_price * 1.02:.2f}",
                    'reason': f'日线趋势向上，价格接近MA20(偏离{dev_ma20:+.1f}%)',
                    'confidence': '中',
                    'profit_potential': '+2.0-3.0%'
                })

        return result

    def print_t0_report(self):
        """打印T+0分析报告"""
        result = self.analyze_intraday_t0()
        trend = result['trend']
        pendulum = result['pendulum']

        print("\n" + "=" * 70)
        print(f"🔥 {self.data['name']}({self.stock_code}) T+0做T分析")
        print("=" * 70)

        # 实时状态
        print("\n━━━ 实时状态 ━━━")
        emoji = "📈" if self.data['change_pct'] > 0 else "📉"
        print(f"当前价: ¥{self.data['current_price']:.2f} ({emoji} {self.data['change_pct']:+.2f}%)")
        print(f"今日区间: ¥{self.data['low']:.2f} - ¥{self.data['high']:.2f}")
        print(f"分析时间: {result['current_time']}")

        if '上证指数' in self.market_data:
            sz = self.market_data['上证指数']
            emoji = "📈" if sz['change_pct'] > 0 else "📉"
            print(f"大盘: 上证指数 {sz['price']:.2f} ({emoji} {sz['change_pct']:+.2f}%)")

        # ━━━ 核心：多级别趋势分析 ━━━
        print("\n━━━ 多级别趋势分析（核心）━━━")
        d = trend['daily']
        print(f"周线趋势: {trend['weekly']['trend']}")
        print(f"日线趋势: {d['direction']} | {d['alignment']}")
        print(f"MA20方向: {d['ma20_dir']} (斜率{d['ma20_slope']:+.2f}%)")
        h_mark = '✅' if d['highs_rising'] else '❌'
        l_mark = '✅' if d['lows_rising'] else '❌'
        print(f"趋势定义: 近高递增{h_mark} 近低递增{l_mark} → {d['trend_def']}")
        print(f"趋势强度: {'█' * trend['strength']}{'░' * (10 - trend['strength'])} {trend['strength']}/10")

        # ━━━ 核心：钟摆位置 ━━━
        print("\n━━━ 钟摆位置（均线偏离度）━━━")
        for ma_name in ['MA20', 'MA60', 'MA120', 'MA250']:
            p = pendulum.get(ma_name, {})
            if p.get('value') is not None and p.get('deviation') is not None:
                print(f"偏离{ma_name}: {p['deviation']:+.1f}% (¥{p['value']:.2f}) → {p['phase']}")
        print(f"综合判断: {pendulum['overall']}")

        # ━━━ 顺大势逆小势 ━━━
        print("\n━━━ 顺大势逆小势 ━━━")
        t0d = result['t0_direction']
        print(f"大势方向: {t0d['major_trend']}（趋势强度 {t0d['strength']}/10）")

        if result.get('intraday'):
            intra = result['intraday']
            vwap_emoji = '上方' if intra['dev_vwap'] > 0 else '下方'
            print(f"小势状态: 日内价格在VWAP{vwap_emoji}({intra['dev_vwap']:+.1f}%)")
        print(f"做T建议: {t0d['bias']}")

        # ━━━ 关键价位 ━━━
        print("\n━━━ 关键价位（基于均线）━━━")
        levels = result['key_levels']
        if result.get('intraday'):
            print(f"📍 日内VWAP: ¥{result['intraday']['vwap']:.2f}（日内价值中枢）")
        print(f"📍 当前价: ¥{levels['current']:.2f}")
        for name, val in levels['supports']:
            print(f"📍 支撑-{name}: ¥{val:.2f}")
        for name, val in levels['resistances']:
            print(f"📍 压力-{name}: ¥{val:.2f}")

        # T+0策略
        if result['strategy']:
            print(f"\n━━━ T+0策略 ━━━")
            s = result['strategy']
            print(f"策略类型: {s.get('type', '分析中')}")
            print(f"策略说明: {s.get('desc', '')}")
            print(f"操作建议: {s.get('method', '')}")

        # 交易机会
        print(f"\n━━━ 交易机会 ━━━")
        if result['trading_opportunities']:
            for i, opp in enumerate(result['trading_opportunities'], 1):
                print(f"\n💡 机会 {i}: {opp['strategy']}")
                print(f"   类型: {'🟢 ' + opp['type'] if opp['type'] == '买入' else '🔴 ' + opp['type']}")
                print(f"   价格: ¥{opp['price']}")
                print(f"   目标: ¥{opp['target']}")
                print(f"   理由: {opp['reason']}")
                print(f"   置信度: {opp['confidence']}")
                print(f"   收益预期: {opp['profit_potential']}")
        else:
            print("⚪️ 当前无明确交易机会，建议观望")

        # ━━━ 可选参考：传统指标 ━━━
        print(f"\n━━━ 可选参考：传统指标（仅供参考）━━━")
        latest = self.df_daily.iloc[-1]
        prev = self.df_daily.iloc[-2]
        macd_bull = latest['DIF'] > latest['DEA']
        macd_status = '多头' if macd_bull else '空头'
        if macd_bull and prev['DIF'] <= prev['DEA']:
            macd_status = '🔥金叉'
        elif not macd_bull and prev['DIF'] >= prev['DEA']:
            macd_status = '💀死叉'
        print(f"MACD(8,17,9): {macd_status} (DIF:{latest['DIF']:.3f} DEA:{latest['DEA']:.3f})")
        print(f"KDJ(6,3,3): K={latest['K']:.1f} D={latest['D']:.1f} J={latest['J']:.1f}")
        print(f"（注：MACD本质是均线偏离度的衍生，KDJ本质是偏离度的另一种计算，均线分析已覆盖）")

        # 风险提示
        print(f"\n━━━ 风险提示 ━━━")
        print("⚠️ T+0交易风险提示:")
        print("   1. 严格止损：单次亏损不超过-1%")
        print("   2. 分批操作：建议用1/4-1/3仓位做T")
        print("   3. 顺大势：只在趋势向上的股票做T买入")
        print("   4. 逆小势：利用日内回调买入，日内冲高卖出")
        print("   5. 控制频率：建议每日1-3次，避免过度交易")

        if self.market_data.get('上证指数', {}).get('change_pct', 0) < -1.5:
            print("\n⛔️ 大盘弱势，不建议T+0交易！")

        # 内功提醒
        print(f"\n━━━ 内功提醒 ━━━")
        print("⚠️ 技术分析只能看到「狗」（价格），看不到「人」（价值）")
        print("   请确认您了解该股票趋势向上的基本面原因：")
        print("   - 业绩是否在增长？行业景气度如何？")
        print("   - 是真实的价值提升，还是资金炒作？")
        print("   - 有没有潜在风险（财务造假、政策打压等）？")
        print("   记住：内功为本（基本面），招式为辅（技术面）")

        print("\n" + "=" * 70)
        print(f"⏰ 报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")


def main():
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python3 analyze_intraday_t0.py <股票代码>")
        print("示例: python3 analyze_intraday_t0.py 600519")
        sys.exit(1)

    stock_code = sys.argv[1]
    analyzer = IntradayT0Analyzer(stock_code)

    if analyzer.fetch_data():
        analyzer.print_t0_report()
    else:
        print("\n❌ 分析失败: 无法获取股票数据")
        print("请检查: 1) 股票代码是否正确 2) 网络连接是否正常")
        sys.exit(1)


if __name__ == "__main__":
    main()
