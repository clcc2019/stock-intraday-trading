#!/usr/bin/env python3
"""
股票日线综合分析工具
基于「内功为本，招式为辅」投资哲学

评分体系（满分100分）：
- 基本面（内功）50分:
  - 盈利能力(15分) + 成长能力(10分) + 财务健康(10分) + 估值水平(10分) + 资金面(5分)
- 技术面（招式）50分:
  - 趋势方向(15分) + 钟摆位置(12.5分) + 趋势强度(10分) + 量价关系(7.5分) + 传统指标(5分)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import os
import sys

warnings.filterwarnings('ignore')

# 导入基本面分析模块、数据源适配层和公共技术指标
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fundamental_analyzer import FundamentalAnalyzer
from data_source import DataSource
from technical import (
    calculate_all_indicators, detect_highs_lows,
    analyze_ma_alignment, calculate_pendulum, calculate_trend_strength,
)


class SimpleStockAnalyzer:
    """股票综合分析器 — 基于趋势+均线+钟摆模型"""

    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.df = None
        self.df_weekly = None
        self.data = {}
        self.market_data = {}

    def fetch_data(self):
        """获取股票数据（使用 baostock，扩展至400天，支持MA120/MA250）"""
        try:
            print(f"📊 正在获取 {self.stock_code} 的数据...")

            end_date = datetime.now()
            start_date = end_date - timedelta(days=400)

            # 使用 baostock 获取日K线
            self.df = DataSource.get_stock_hist(
                stock_code=self.stock_code,
                start_date=start_date,
                end_date=end_date,
                adjust='qfq',
                period='daily'
            )

            if self.df is None or self.df.empty:
                print(f"❌ 无法获取股票 {self.stock_code} 的历史数据")
                return False

            # 获取周K线
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

            latest = self.df.iloc[-1]

            self.data = {
                'name': f'股票{self.stock_code}',
                'current_price': latest['收盘'],
                'change_pct': ((latest['收盘'] - self.df.iloc[-2]['收盘']) / self.df.iloc[-2]['收盘']) * 100,
                'high': latest['最高'],
                'low': latest['最低'],
                'open': latest['开盘'],
                'volume': latest['成交量'],
                'turnover': latest['换手率'] if '换手率' in latest else 0
            }

            # baostock 数据中已包含股票代码，可从中提取名称
            # 但为了兼容性，仍保留从 code 列提取（如果有）
            if 'code' in self.df.columns and not self.df.empty:
                # baostock 返回的 code 格式如 'sh.600519'
                # 名称需要单独查询，暂时保持默认
                pass

            self.calculate_indicators()
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
            # 获取上证指数
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

            # 行业数据暂时无法从 baostock 获取，跳过
            # 可以考虑从其他数据源补充，或者不显示行业数据
        except:
            pass

    def calculate_indicators(self):
        """计算技术指标 — 使用公共模块"""
        calculate_all_indicators(self.df)

        # 周线均线
        if self.df_weekly is not None and not self.df_weekly.empty:
            from technical import calculate_ma
            calculate_ma(self.df_weekly, windows=[5, 10, 20])
            # 重命名为 W_ 前缀以区分
            for w in [5, 10, 20]:
                if f'MA{w}' in self.df_weekly.columns:
                    self.df_weekly[f'W_MA{w}'] = self.df_weekly[f'MA{w}']

    def analyze(self):
        """综合分析 — 以趋势+均线+钟摆为核心"""
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2]
        prev2 = self.df.iloc[-3] if len(self.df) >= 3 else prev
        current_price = self.data['current_price']

        signals = {
            'buy': 0, 'sell': 0,
            'indicators': {},
            'key_signals': [],
        }

        # ============================================================
        # 核心维度1: 趋势方向（满分6分，权重30%）
        # ============================================================
        trend_buy = 0
        trend_sell = 0
        trend_details = []

        ma5 = latest['MA5']
        ma10 = latest['MA10']
        ma20 = latest['MA20']
        ma60 = latest.get('MA60', np.nan)
        ma120 = latest.get('MA120', np.nan)

        # 均线排列
        has_ma60 = not (isinstance(ma60, float) and np.isnan(ma60))
        if has_ma60 and ma5 > ma10 > ma20 > ma60:
            trend_buy += 3
            trend_details.append('完美多头排列')
            signals['key_signals'].append('⭐ 均线完美多头排列（MA5>10>20>60）')
        elif ma5 > ma10 > ma20:
            trend_buy += 2
            trend_details.append('多头排列')
            signals['key_signals'].append('⭐ 均线多头排列（MA5>10>20）')
        elif ma5 > ma10:
            trend_buy += 1
            trend_details.append('短期偏多')
        elif has_ma60 and ma5 < ma10 < ma20 < ma60:
            trend_sell += 3
            trend_details.append('空头排列')
            signals['key_signals'].append('⛔ 均线空头排列（MA5<10<20<60）')
        elif ma5 < ma10 < ma20:
            trend_sell += 2
            trend_details.append('偏空排列')
        elif ma5 < ma10:
            trend_sell += 1
            trend_details.append('短期偏空')
        else:
            trend_details.append('震荡缠绕')

        # 高低点递增/递减（使用公共模块）
        hl = detect_highs_lows(self.df)

        if hl['highs_rising']:
            trend_buy += 1
            trend_details.append('高点递增')
        if hl['lows_rising']:
            trend_buy += 1
            trend_details.append('低点递增')
        if hl['highs_falling']:
            trend_sell += 1
            trend_details.append('高点递减')
        if hl['lows_falling']:
            trend_sell += 1
            trend_details.append('低点递减')

        # 价格与MA120的关系
        has_ma120 = not (isinstance(ma120, float) and np.isnan(ma120))
        if has_ma120 and current_price > ma120:
            trend_buy += 1
            trend_details.append('价格>MA120')

        signals['buy'] += min(6, trend_buy)
        signals['sell'] += min(6, trend_sell)

        trend_status = '✅' if trend_buy > trend_sell else ('❌' if trend_sell > trend_buy else '⚠️')
        signals['indicators']['趋势方向'] = f'{trend_status} {"/".join(trend_details)}'

        # ============================================================
        # 核心维度2: 多级别钟摆位置/均线偏离度（满分5分，权重25%）
        # MA5=超短期情绪, MA10=短期情绪, MA20=中期中枢, MA60=季度趋势
        # ============================================================
        pendulum_buy = 0
        pendulum_sell = 0
        pendulum_details = []

        dev_ma5 = (current_price - ma5) / ma5 * 100 if not np.isnan(ma5) else 0
        dev_ma10 = (current_price - ma10) / ma10 * 100 if not np.isnan(ma10) else 0
        dev_ma20 = (current_price - ma20) / ma20 * 100 if not np.isnan(ma20) else 0
        dev_ma60 = (current_price - ma60) / ma60 * 100 if has_ma60 else 0
        dev_ma120 = (current_price - ma120) / ma120 * 100 if has_ma120 else 0

        # --- 短期钟摆（MA5/MA10联合判断）---
        if dev_ma5 <= 1 and dev_ma10 <= 2:
            pendulum_buy += 1  # 短期均线收敛，安全
            pendulum_details.append(f'短期均线收敛(MA5:{dev_ma5:+.1f}%/MA10:{dev_ma10:+.1f}%)')
        elif dev_ma5 > 5 and dev_ma10 > 4:
            pendulum_sell += 1  # 短期过热
            pendulum_details.append(f'短期过热(MA5:{dev_ma5:+.1f}%/MA10:{dev_ma10:+.1f}%)')
        elif dev_ma5 < -3 and dev_ma10 < -2:
            pendulum_buy += 1  # 短期超跌
            pendulum_details.append(f'短期超跌(MA5:{dev_ma5:+.1f}%/MA10:{dev_ma10:+.1f}%)')

        # --- 中期钟摆（MA20）---
        if -3 <= dev_ma20 <= 3:
            pendulum_buy += 2  # 中枢附近，适合买入
            pendulum_details.append(f'MA20中枢附近({dev_ma20:+.1f}%)')
        elif dev_ma20 > 8:
            pendulum_sell += 2  # 过度偏高
            pendulum_details.append(f'MA20偏离过大({dev_ma20:+.1f}%)')
            signals['key_signals'].append(f'⚠️ 价格偏离MA20达{dev_ma20:+.1f}%，回归压力增大')
        elif dev_ma20 > 5:
            pendulum_sell += 1
            pendulum_details.append(f'MA20偏高({dev_ma20:+.1f}%)')
        elif dev_ma20 < -8:
            pendulum_buy += 2  # 过度偏低
            pendulum_details.append(f'MA20过度偏低({dev_ma20:+.1f}%)')
            signals['key_signals'].append(f'⭐ 价格偏离MA20达{dev_ma20:+.1f}%，反弹动力增大')
        elif dev_ma20 < -5:
            pendulum_buy += 1
            pendulum_details.append(f'MA20偏低({dev_ma20:+.1f}%)')

        # --- 季度钟摆（MA60）---
        if has_ma60:
            if dev_ma60 > 15:
                pendulum_sell += 2
                pendulum_details.append(f'MA60偏离大({dev_ma60:+.1f}%)')
                signals['key_signals'].append(f'⛔ 价格偏离MA60达{dev_ma60:+.1f}%，绳子很紧')
            elif dev_ma60 > 8:
                pendulum_sell += 1
                pendulum_details.append(f'MA60偏高({dev_ma60:+.1f}%)')
            elif -3 <= dev_ma60 <= 5:
                pendulum_buy += 1
                pendulum_details.append(f'MA60附近({dev_ma60:+.1f}%)')
            elif dev_ma60 < -10:
                pendulum_buy += 2
                pendulum_details.append(f'MA60偏离大({dev_ma60:+.1f}%)')
            elif dev_ma60 < -5:
                pendulum_buy += 1
                pendulum_details.append(f'MA60偏低({dev_ma60:+.1f}%)')

        signals['buy'] += min(5, pendulum_buy)
        signals['sell'] += min(5, pendulum_sell)

        pend_status = '✅' if pendulum_buy > pendulum_sell else ('❌' if pendulum_sell > pendulum_buy else '⚠️')
        signals['indicators']['钟摆位置'] = f'{pend_status} {"/".join(pendulum_details)}'

        # ============================================================
        # 核心维度3: 趋势强度（满分4分，权重20%）
        # ============================================================
        strength_buy = 0
        strength_sell = 0
        strength_details = []

        # MA20斜率
        ma20_slope = latest.get('MA20_slope', 0)
        if isinstance(ma20_slope, float) and np.isnan(ma20_slope):
            ma20_slope = 0

        if ma20_slope > 2:
            strength_buy += 2
            strength_details.append(f'MA20加速上行({ma20_slope:+.1f}%)')
        elif ma20_slope > 0:
            strength_buy += 1
            strength_details.append(f'MA20上行({ma20_slope:+.1f}%)')
        elif ma20_slope < -2:
            strength_sell += 2
            strength_details.append(f'MA20加速下行({ma20_slope:+.1f}%)')
        elif ma20_slope < 0:
            strength_sell += 1
            strength_details.append(f'MA20下行({ma20_slope:+.1f}%)')

        # 近20日涨幅（相对强度）
        price_20d_ago = self.df.iloc[-20]['收盘'] if len(self.df) >= 20 else current_price
        change_20d = (current_price - price_20d_ago) / price_20d_ago * 100
        if change_20d > 10:
            strength_buy += 2
            strength_details.append(f'20日强势(+{change_20d:.1f}%)')
        elif change_20d > 3:
            strength_buy += 1
            strength_details.append(f'20日偏强(+{change_20d:.1f}%)')
        elif change_20d < -10:
            strength_sell += 2
            strength_details.append(f'20日弱势({change_20d:+.1f}%)')
        elif change_20d < -3:
            strength_sell += 1
            strength_details.append(f'20日偏弱({change_20d:+.1f}%)')

        signals['buy'] += min(4, strength_buy)
        signals['sell'] += min(4, strength_sell)

        str_status = '✅' if strength_buy > strength_sell else ('❌' if strength_sell > strength_buy else '⚠️')
        signals['indicators']['趋势强度'] = f'{str_status} {"/".join(strength_details) if strength_details else "中性"}'

        # ============================================================
        # 辅助维度: 量价关系（满分3分，权重15%）
        # ============================================================
        vol_buy = 0
        vol_sell = 0
        vol_details = []

        vol_ratio = latest['成交量'] / latest['VOL_MA5'] if latest['VOL_MA5'] > 0 else 1
        vol_ratio_20 = latest['成交量'] / latest['VOL_MA20'] if 'VOL_MA20' in latest and latest['VOL_MA20'] > 0 else 1

        if vol_ratio > 1.5 and self.data['change_pct'] > 0:
            vol_buy += 2
            vol_details.append(f'放量上涨(量比{vol_ratio:.1f})')
        elif vol_ratio > 1.5 and self.data['change_pct'] < 0:
            vol_sell += 2
            vol_details.append(f'放量下跌(量比{vol_ratio:.1f})')
        elif vol_ratio < 0.5 and self.data['change_pct'] < 0:
            vol_buy += 1
            vol_details.append(f'缩量下跌(量比{vol_ratio:.1f})，止跌信号')
        elif vol_ratio < 0.5:
            vol_details.append(f'极度缩量(量比{vol_ratio:.1f})')
        else:
            vol_details.append(f'量比正常({vol_ratio:.1f})')

        # 量价配合
        if self.data['change_pct'] > 1 and vol_ratio > 1.2:
            vol_buy += 1
            vol_details.append('量价配合良好')
        elif self.data['change_pct'] < -1 and vol_ratio > 1.5:
            vol_sell += 1
            vol_details.append('放量杀跌')

        signals['buy'] += min(3, vol_buy)
        signals['sell'] += min(3, vol_sell)

        vol_status = '✅' if vol_buy > vol_sell else ('❌' if vol_sell > vol_buy else '⚠️')
        signals['indicators']['量价关系'] = f'{vol_status} {"/".join(vol_details)}'

        # ============================================================
        # 可选参考: 传统指标 MACD/KDJ（满分2分，权重10%）
        # ============================================================
        legacy_buy = 0
        legacy_sell = 0
        legacy_details = []

        # MACD
        macd_bull = latest['DIF'] > latest['DEA']
        macd_golden = macd_bull and prev['DIF'] <= prev['DEA']
        macd_death = not macd_bull and prev['DIF'] >= prev['DEA']

        if macd_golden:
            legacy_buy += 1
            legacy_details.append('MACD金叉')
        elif macd_death:
            legacy_sell += 1
            legacy_details.append('MACD死叉')
        elif macd_bull:
            legacy_details.append('MACD多头')
        else:
            legacy_details.append('MACD空头')

        # KDJ
        j_value = latest['J']
        k_value = latest['K']
        d_value = latest['D']
        kdj_golden = k_value > d_value and prev['K'] <= prev['D']
        kdj_death = k_value < d_value and prev['K'] >= prev['D']

        if kdj_golden and j_value < 30:
            legacy_buy += 1
            legacy_details.append('KDJ低位金叉')
        elif kdj_death and j_value > 70:
            legacy_sell += 1
            legacy_details.append('KDJ高位死叉')
        elif j_value < 20:
            legacy_details.append(f'KDJ超卖J={j_value:.0f}')
        elif j_value > 80:
            legacy_details.append(f'KDJ超买J={j_value:.0f}')

        signals['buy'] += min(2, legacy_buy)
        signals['sell'] += min(2, legacy_sell)

        legacy_status = '✅' if legacy_buy > legacy_sell else ('❌' if legacy_sell > legacy_buy else '⚠️')
        signals['indicators']['传统指标(参考)'] = f'{legacy_status} {"/".join(legacy_details)} (DIF:{latest["DIF"]:.3f} K:{k_value:.0f} J:{j_value:.0f})'

        # ============================================================
        # 市场环境调整
        # ============================================================
        market_adj = 0
        market_desc = []

        if '上证指数' in self.market_data:
            sz_change = self.market_data['上证指数']['change_pct']
            if sz_change > 1:
                market_adj += 2
                market_desc.append(f"✅ 大盘强势 ({sz_change:+.2f}%)")
            elif sz_change < -1:
                market_adj -= 2
                market_desc.append(f"❌ 大盘弱势 ({sz_change:+.2f}%)")
            else:
                market_desc.append(f"⚠️ 大盘震荡 ({sz_change:+.2f}%)")

        if '行业' in self.market_data:
            industry_change = self.market_data['行业']['change_pct']
            if industry_change > 2:
                market_adj += 2
                market_desc.append(f"✅ 板块强势 ({industry_change:+.2f}%)")
            elif industry_change < -2:
                market_adj -= 2
                market_desc.append(f"❌ 板块弱势 ({industry_change:+.2f}%)")
            else:
                market_desc.append(f"⚠️ 板块正常 ({industry_change:+.2f}%)")

        signals['market_desc'] = market_desc

        # 技术面得分（原始满分20分，等比放大至50分）
        raw_buy = signals['buy']
        raw_sell = signals['sell']
        tech_buy = min(50, int(raw_buy * 2.5 + max(0, market_adj) * 2.5))
        tech_sell = min(50, int(raw_sell * 2.5 + max(0, -market_adj) * 2.5))

        # 基本面得分（满分50分）
        fundamental_result = None
        fundamental_score = 0
        try:
            fa = FundamentalAnalyzer(self.stock_code, self.data.get('name'))
            fa.fetch_all_data()
            fundamental_result = fa.get_fundamental_score()
            fundamental_score = fundamental_result['total']
            self._fundamental_report = fa.get_report_text()
            self._fundamental_result = fundamental_result
        except Exception:
            self._fundamental_report = "\n━━━ 基本面分析（内功）━━━\n⚠️ 基本面数据获取失败，仅展示技术面分析"
            self._fundamental_result = None

        # 综合得分（满分100分 = 技术面50 + 基本面50）
        buy_score = tech_buy + fundamental_score
        sell_score = tech_sell + max(0, 50 - fundamental_score)  # 基本面差时增加卖出分

        # 价格建议
        support = ma20 if not np.isnan(ma20) else latest['MA10']
        if has_ma60 and ma60 < support:
            support = ma60
        resistance = max(latest['MA5'], self.data['high'])

        buy_price_low = support * 0.99
        buy_price_high = current_price * 0.995
        sell_price = current_price * 1.02
        stop_loss = current_price * 0.97

        # 生成建议（基于100分制综合评分）
        is_uptrend = trend_buy >= 3
        is_downtrend = trend_sell >= 3
        is_near_ma = abs(dev_ma20) <= 5
        has_good_fundamental = fundamental_score >= 28  # 基本面良好

        if buy_score >= 75 and is_uptrend and has_good_fundamental:
            action = '🟢 强烈买入'
            confidence = '很高'
            position = '30-50%'
            advice = '技术面+基本面共振向好，积极买入'
        elif buy_score >= 60 and is_uptrend:
            action = '🟢 买入'
            confidence = '高'
            position = '20-30%'
            advice = '趋势偏多，基本面支撑，可适量买入'
        elif buy_score >= 55 and is_near_ma:
            action = '🟡 可考虑买入'
            confidence = '中'
            position = '10-20%'
            advice = '价格接近均线，基本面尚可，等待趋势确认后买入'
        elif sell_score >= 75 and is_downtrend:
            action = '🔴 强烈卖出'
            confidence = '很高'
            position = '70-100%'
            advice = '趋势向下+基本面走弱，建议清仓或大幅减仓'
        elif sell_score >= 60:
            action = '🔴 卖出'
            confidence = '高'
            position = '50-70%'
            advice = '趋势转弱，建议减仓'
        elif sell_score >= 45:
            action = '🟠 可考虑卖出'
            confidence = '中'
            position = '30-50%'
            advice = '适度减仓，等待趋势企稳'
        elif buy_score >= 45:
            action = '🟡 可考虑买入'
            confidence = '中'
            position = '10-20%'
            advice = '信号偏多但不强烈，小仓位试探'
        else:
            action = '⚪️ 观望'
            confidence = '低'
            position = '0%'
            advice = '趋势不明确，等待方向明确后再操作'

        # 趋势下行降级
        if is_downtrend and '买入' in action:
            action = '⚪️ 观望'
            confidence = '低'
            position = '0%'
            advice = '⚠️ 趋势向下（均线空头排列），不建议买入'

        # 基本面极差降级
        if fundamental_score < 15 and '强烈买入' in action:
            action = '🟡 可考虑买入'
            confidence = '中'
            position = '10-20%'
            advice = '⚠️ 技术面向好但基本面偏弱，控制仓位'

        return {
            'action': action,
            'confidence': confidence,
            'position': position,
            'advice': advice,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'tech_buy': tech_buy,
            'tech_sell': tech_sell,
            'fundamental_score': fundamental_score,
            'max_score': 100,
            'market_adj': market_adj,
            'signals': signals,
            'trend_info': {
                'trend_buy': trend_buy,
                'trend_sell': trend_sell,
                'is_uptrend': is_uptrend,
                'is_downtrend': is_downtrend,
                'dev_ma5': dev_ma5,
                'dev_ma10': dev_ma10,
                'dev_ma20': dev_ma20,
                'dev_ma60': dev_ma60,
                'dev_ma120': dev_ma120,
                'ma20_slope': ma20_slope,
                'change_20d': change_20d,
            },
            'prices': {
                'current': current_price,
                'buy_low': buy_price_low,
                'buy_high': buy_price_high,
                'sell': sell_price,
                'stop_loss': stop_loss,
                'support': support,
                'resistance': resistance,
                'ma20': ma20,
                'ma60': ma60 if has_ma60 else None,
                'ma120': ma120 if has_ma120 else None,
            }
        }

    def print_report(self):
        """打印分析报告"""
        result = self.analyze()
        trend = result['trend_info']

        print("\n" + "=" * 70)
        print(f"📊 {self.data['name']}({self.stock_code}) 分析报告")
        print("=" * 70)

        # 市场环境
        print("\n━━━ 市场环境 ━━━")
        if '上证指数' in self.market_data:
            sz = self.market_data['上证指数']
            emoji = "📈" if sz['change_pct'] > 0 else "📉"
            print(f"大盘: 上证指数 {sz['price']:.2f} ({emoji} {sz['change_pct']:+.2f}%)")
        else:
            print("大盘: 数据获取中...")

        if '行业' in self.market_data:
            industry = self.market_data['行业']
            emoji = "📈" if industry['change_pct'] > 0 else "📉"
            print(f"板块: {industry['name']} ({emoji} {industry['change_pct']:+.2f}%)")

        # 当前价格
        print("\n━━━ 当前状态 ━━━")
        emoji = "📈" if self.data['change_pct'] > 0 else "📉"
        print(f"当前价: ¥{self.data['current_price']:.2f} ({emoji} {self.data['change_pct']:+.2f}%)")
        print(f"今日区间: ¥{self.data['low']:.2f} - ¥{self.data['high']:.2f}")

        # 多级别趋势
        print("\n━━━ 多级别趋势（核心）━━━")
        print(f"趋势方向: {result['signals']['indicators']['趋势方向']}")
        print(f"趋势强度: {result['signals']['indicators']['趋势强度']}")

        # 均线值
        prices = result['prices']
        ma_str = f"MA20:¥{prices['ma20']:.2f}" if prices['ma20'] and not np.isnan(prices['ma20']) else ""
        if prices.get('ma60'):
            ma_str += f" MA60:¥{prices['ma60']:.2f}"
        if prices.get('ma120'):
            ma_str += f" MA120:¥{prices['ma120']:.2f}"
        if ma_str:
            print(f"均线值: {ma_str}")

        # 钟摆位置
        print("\n━━━ 多级别钟摆位置（均线偏离度）━━━")
        print(f"{result['signals']['indicators']['钟摆位置']}")
        dev_short = f"短期: MA5:{trend['dev_ma5']:+.1f}% MA10:{trend['dev_ma10']:+.1f}%"
        dev_mid = f"中期: MA20:{trend['dev_ma20']:+.1f}%"
        if trend['dev_ma60'] != 0:
            dev_mid += f" MA60:{trend['dev_ma60']:+.1f}%"
        if trend['dev_ma120'] != 0:
            dev_mid += f" MA120:{trend['dev_ma120']:+.1f}%"
        print(f"偏离度 {dev_short}")
        print(f"偏离度 {dev_mid}")

        # 量价关系
        print("\n━━━ 量价关系 ━━━")
        print(f"{result['signals']['indicators']['量价关系']}")

        # 传统指标（可选参考）
        print("\n━━━ 可选参考：传统指标 ━━━")
        print(f"{result['signals']['indicators']['传统指标(参考)']}")
        print(f"（MACD本质是均线偏离度衍生，KDJ是偏离度的另一种计算）")

        # 关键信号
        if result['signals'].get('key_signals'):
            print("\n━━━ 关键信号 ━━━")
            for sig in result['signals']['key_signals']:
                print(sig)

        # 基本面分析（内功）
        if hasattr(self, '_fundamental_report'):
            print(self._fundamental_report)

        # 综合评分
        max_score = result.get('max_score', 100)
        print("\n━━━ 综合评分 ━━━")
        tech_buy = result.get('tech_buy', 0)
        tech_sell = result.get('tech_sell', 0)
        fund_score = result.get('fundamental_score', 0)
        print(f"技术面(招式): 买入 {tech_buy}/50 | 卖出 {tech_sell}/50")
        print(f"基本面(内功): {fund_score}/50")
        print(f"综合买入评分: {result['buy_score']}/{max_score}")
        print(f"综合卖出评分: {result['sell_score']}/{max_score}")
        print(f"评分构成: 基本面50%(盈利15+成长10+健康10+估值10+资金5) + 技术面50%(趋势15+钟摆12.5+强度10+量价7.5+指标5)")
        if result['market_adj'] != 0:
            print(f"市场调整: {result['market_adj']:+d} 分")

        if result['signals']['market_desc']:
            for desc in result['signals']['market_desc']:
                print(f"  {desc}")

        # 操作建议
        print("\n━━━ 操作建议 ━━━")
        print(f"{result['action']}")
        print(f"置信度: {result['confidence']}")
        print(f"建议: {result['advice']}")

        print()
        if '买入' in result['action']:
            print(f"💰 买入价: ¥{prices['buy_low']:.2f} - ¥{prices['buy_high']:.2f}")
            print(f"🎯 目标价: ¥{prices['sell']:.2f} (预期收益 +{((prices['sell']/prices['current'])-1)*100:.1f}%)")
            print(f"⛔️ 止损价: ¥{prices['stop_loss']:.2f} (最大亏损 -3%)")
            print(f"📊 建议仓位: {result['position']}")
            print(f"📍 关键支撑: ¥{prices['support']:.2f}")
        elif '卖出' in result['action']:
            print(f"💰 卖出价: ¥{prices['current']:.2f} 以上")
            print(f"⛔️ 止损价: ¥{prices['stop_loss']:.2f}")
            print(f"📊 建议减仓: {result['position']}")
            print(f"📍 关键压力: ¥{prices['resistance']:.2f}")
        else:
            print(f"💰 观望价位:")
            print(f"   买入参考: ¥{prices['buy_low']:.2f} 附近（接近均线支撑）")
            print(f"   卖出参考: ¥{prices['sell']:.2f} 以上")
            print(f"📍 支撑位: ¥{prices['support']:.2f}")
            print(f"📍 压力位: ¥{prices['resistance']:.2f}")

        # 内功提醒
        print("\n━━━ 投资流程提醒 ━━━")
        print("📋 本报告已融合基本面（内功）+ 技术面（招式）综合分析")
        print("   投资流程：1.量化筛选 → 2.定性验证（管理层/文化/行业前景）→ 3.交易决策")
        print("   定性因素（管理层诚信、公司文化、行业竞争格局）仍需您自行判断")

        if result['market_adj'] < -2:
            print("\n⚠️ 风险提示: 市场环境不佳，建议降低仓位或观望")

        print("\n" + "=" * 70)
        print(f"⏰ 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📅 数据日期: {self.df.iloc[-1]['日期']}")
        print("=" * 70 + "\n")


def main():
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python3 analyze_stock_simple.py <股票代码>")
        print("示例: python3 analyze_stock_simple.py 600519")
        sys.exit(1)

    stock_code = sys.argv[1]
    analyzer = SimpleStockAnalyzer(stock_code)

    if analyzer.fetch_data():
        analyzer.print_report()
    else:
        print("\n❌ 分析失败: 无法获取股票数据")
        print("请检查: 1) 股票代码是否正确 2) 网络连接是否正常")
        sys.exit(1)


if __name__ == "__main__":
    main()
