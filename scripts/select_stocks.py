#!/usr/bin/env python3
"""
趋势选股工具
基于「顺大势」投资哲学，筛选趋势向上的股票

核心理念：
- 公设一：价格围绕价值波动 → 均线 = 价值中枢
- 公设二：钟摆式过度波动 → 均线偏离度 = 钟摆位置
- 顺大势：只选均线多头排列、趋势方向向上的股票
- 逆小势：标注钟摆回摆至均线附近的最佳做T候选
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import argparse
import os
import sys

warnings.filterwarnings('ignore')

# 导入基本面分析模块、数据源适配层和公共技术指标
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fundamental_analyzer import FundamentalAnalyzer
from data_source import DataSource
from technical import calculate_ma, calculate_macd, calculate_kdj, detect_highs_lows, analyze_ma_alignment, _safe_ma


class TrendStockSelector:
    """趋势选股器 — 基于均线+趋势+钟摆模型"""

    def __init__(self, index=None, sector=None, top_n=30, no_fundamental=False):
        self.index = index
        self.sector = sector
        self.top_n = top_n
        self.no_fundamental = no_fundamental
        self.results = []
        self.stock_names = {}  # code -> name 映射，避免逐个查询

    def get_stock_pool(self):
        """获取股票池"""
        try:
            if self.index:
                return self._get_index_stocks()
            elif self.sector:
                return self._get_sector_stocks()
            else:
                return self._get_all_a_stocks()
        except Exception as e:
            print(f"❌ 获取股票池失败: {e}")
            return []

    def _get_index_stocks(self):
        """获取指数成分股（使用 baostock）"""
        index_map = {
            'hs300': ('沪深300', 'sh.000300'),
            'zz500': ('中证500', 'sh.000905'),
            'sz50': ('上证50', 'sh.000016'),
        }

        key = self.index.lower()
        if key == 'zz1000':
            print(f"⚠️ baostock 不支持中证1000成分股查询，将从全A股中选股...")
            return self._get_all_a_stocks()
        if key not in index_map:
            print(f"⚠️ 不支持的指数: {self.index}，支持: hs300, zz500, sz50")
            print("将使用沪深300")
            key = 'hs300'

        name, index_code = index_map[key]
        print(f"📊 从{name}成分股中选股...")

        try:
            df = DataSource.get_index_stocks(index_code)
            if df is not None and not df.empty:
                codes = df['代码'].astype(str).tolist()
                # 构建名称映射
                for _, row in df.iterrows():
                    self.stock_names[row['代码']] = row['名称']
                print(f"✅ 获取到 {len(codes)} 只成分股")
                return codes
        except Exception as e:
            print(f"⚠️ 获取指数成分股失败: {e}")

        # 备用方案
        print("⚠️ 使用备用方案获取股票列表...")
        return self._get_all_a_stocks()[:300]

    def _get_sector_stocks(self):
        """获取板块成分股（baostock 不支持板块，使用全市场）"""
        print(f"⚠️ baostock 不支持板块筛选，将从全市场选股...")
        return self._get_all_a_stocks()

    def _get_all_a_stocks(self):
        """获取全A股列表（使用 baostock）"""
        print("📊 获取全A股列表（较慢，建议使用 --index hs300）...")
        try:
            df = DataSource.get_stock_list()
            if df is not None and not df.empty:
                codes = df['代码'].tolist()
                # 构建名称映射
                for _, row in df.iterrows():
                    self.stock_names[row['代码']] = row['名称']
                print(f"✅ 获取到 {len(codes)} 只A股")
                return codes
        except Exception as e:
            print(f"❌ 获取A股列表失败: {e}")
            return []

    def _fetch_stock_data(self, stock_code, days=400):
        """获取股票数据（使用 baostock）"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            df = DataSource.get_stock_hist(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                adjust='qfq',
                period='daily'
            )
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

        return None

    def analyze_single_stock(self, stock_code):
        """分析单只股票的趋势状态"""
        try:
            df = self._fetch_stock_data(stock_code)

            if df is None or df.empty or len(df) < 120:
                return None

            # 计算均线（使用公共模块）
            calculate_ma(df, windows=[5, 10, 20, 60, 120, 250])
            # 计算 MACD 和 KDJ（用于高位接盘过滤）
            calculate_macd(df)
            calculate_kdj(df)

            latest = df.iloc[-1]
            price = latest['收盘']
            name = self.stock_names.get(stock_code, stock_code)

            # === 均线排列分析（使用公共模块）===
            ma5 = _safe_ma(latest, 'MA5')
            ma10 = _safe_ma(latest, 'MA10')
            ma20 = _safe_ma(latest, 'MA20')
            ma60 = _safe_ma(latest, 'MA60')
            ma120 = _safe_ma(latest, 'MA120')

            if any(v is None for v in [ma5, ma10, ma20, ma60, ma120]):
                return None

            # 均线多头排列检查
            perfect_bull = (ma5 > ma10 > ma20 > ma60)  # 完美多头
            strong_bull = (ma5 > ma10 > ma20) and (ma20 > ma60 * 0.99)  # 强势多头
            basic_bull = (ma5 > ma10) and (ma10 > ma20 * 0.99)  # 基本多头

            if not basic_bull:
                return None  # 不符合基本多头排列，跳过

            # === 均线方向（斜率）===
            if len(df) >= 26:
                ma20_slope = (ma20 - df.iloc[-6]['MA20']) / df.iloc[-6]['MA20'] * 100 if df.iloc[-6]['MA20'] > 0 else 0
                ma60_slope = (ma60 - df.iloc[-21]['MA60']) / df.iloc[-21]['MA60'] * 100 if len(df) >= 81 and df.iloc[-21]['MA60'] > 0 else 0
            else:
                ma20_slope = 0
                ma60_slope = 0

            # MA20必须向上
            if ma20_slope <= 0:
                return None

            # === 均线偏离度（钟摆位置）===
            dev_ma20 = (price - ma20) / ma20 * 100
            dev_ma60 = (price - ma60) / ma60 * 100
            dev_ma120 = (price - ma120) / ma120 * 100

            # 过滤过度偏离（绳子太紧，追高风险）
            if dev_ma60 > 20:
                return None

            # === MACD/KDJ 高位接盘过滤 ===
            overbought_flags = []
            overbought_penalty = 0  # 扣分（0-3分）

            # KDJ 超买检测
            j_val = latest.get('J', 50)
            k_val = latest.get('K', 50)
            d_val = latest.get('D', 50)
            if not (isinstance(j_val, float) and np.isnan(j_val)):
                if j_val > 90 and k_val > d_val:
                    # J > 90 但 K 仍在 D 上方 = 强势超买还没死叉，轻度扣分
                    overbought_flags.append(f'KDJ超买(J={j_val:.0f})')
                    overbought_penalty += 1
                if j_val > 80 and len(df) >= 2:
                    prev_k = df.iloc[-2].get('K', 0)
                    prev_d = df.iloc[-2].get('D', 0)
                    # K/D 死叉：前一天 K > D，今天 K < D → 超买死叉，重度扣分
                    if prev_k > prev_d and k_val < d_val:
                        overbought_flags.append(f'KDJ高位死叉(J={j_val:.0f})')
                        overbought_penalty += 2
                    # J 极值钝化 > 100
                    if j_val > 100:
                        overbought_flags.append(f'J值极端({j_val:.0f})')
                        overbought_penalty += 1

            # MACD 顶背离检测（价格新高但 MACD 柱缩短）
            dif_val = latest.get('DIF', 0)
            macd_val = latest.get('MACD', 0)
            if not (isinstance(dif_val, float) and np.isnan(dif_val)):
                if len(df) >= 20:
                    recent_20 = df.tail(20)
                    price_high_idx = recent_20['收盘'].idxmax()
                    # 价格在近20日高位（前3名）
                    price_rank = (recent_20['收盘'] >= price).sum()
                    if price_rank <= 3:
                        # 检查 MACD 柱是否在缩短（近5日 MACD 柱连续缩短）
                        recent_macd = df['MACD'].tail(5).tolist()
                        if len(recent_macd) >= 5:
                            # MACD 柱从正值开始缩短 = 上涨动能衰竭
                            if recent_macd[-1] > 0 and recent_macd[-1] < recent_macd[-3]:
                                overbought_flags.append('MACD柱缩短(动能衰竭)')
                                overbought_penalty += 1
                            # DIF/DEA 死叉（DIF 下穿 DEA）
                            if len(df) >= 2:
                                prev_dif = df.iloc[-2].get('DIF', 0)
                                prev_dea = df.iloc[-2].get('DEA', 0)
                                dea_val = latest.get('DEA', 0)
                                if not (isinstance(prev_dif, float) and np.isnan(prev_dif)):
                                    if prev_dif > prev_dea and dif_val < dea_val:
                                        overbought_flags.append('MACD死叉')
                                        overbought_penalty += 2

            # 极端超买直接过滤（扣分 >= 4 表示多个超买信号同时出现）
            if overbought_penalty >= 4:
                return None

            # === 趋势定义验证（使用公共模块）===
            hl = detect_highs_lows(df)
            highs_rising = hl['highs_rising']
            lows_rising = hl['lows_rising']

            # === 趋势强度评分（0-10）===
            strength = 0

            # 均线排列（0-3分）
            if perfect_bull:
                strength += 3
            elif strong_bull:
                strength += 2
            elif basic_bull:
                strength += 1

            # MA120也在下方（0-1分）
            if price > ma120:
                strength += 1

            # 均线斜率（0-2分）
            if ma20_slope > 1:
                strength += 1
            if ma20_slope > 3:
                strength += 1

            # 高低点递增（0-2分）
            if highs_rising:
                strength += 1
            if lows_rising:
                strength += 1

            # 相对强度（近20日涨幅，0-2分）
            price_20d_ago = df.iloc[-20]['收盘'] if len(df) >= 20 else price
            change_20d = (price - price_20d_ago) / price_20d_ago * 100
            if change_20d > 5:
                strength += 2
            elif change_20d > 0:
                strength += 1

            # === MACD/KDJ 动态加减分 ===
            momentum_bonus = 0
            momentum_flags = []
            if not (isinstance(macd_val, float) and np.isnan(macd_val)):
                # MACD 金叉加分（DIF 上穿 DEA）
                if len(df) >= 2:
                    prev_dif = df.iloc[-2].get('DIF', 0)
                    prev_dea = df.iloc[-2].get('DEA', 0)
                    dea_val = latest.get('DEA', 0)
                    if not (isinstance(prev_dif, float) and np.isnan(prev_dif)):
                        if prev_dif < prev_dea and dif_val > dea_val:
                            momentum_bonus += 1
                            momentum_flags.append('MACD金叉')
                # MACD 零轴上方，柱递增 = 强势
                recent_macd = df['MACD'].tail(3).tolist()
                if len(recent_macd) >= 3 and all(v > 0 for v in recent_macd if not (isinstance(v, float) and np.isnan(v))):
                    if recent_macd[-1] > recent_macd[-2]:
                        momentum_bonus += 1
                        momentum_flags.append('MACD红柱增长')

            if not (isinstance(j_val, float) and np.isnan(j_val)):
                # KDJ 金叉加分（K 上穿 D，且不在超买区）
                if j_val < 80 and len(df) >= 2:
                    prev_k = df.iloc[-2].get('K', 50)
                    prev_d = df.iloc[-2].get('D', 50)
                    if prev_k < prev_d and k_val > d_val:
                        momentum_bonus += 1
                        momentum_flags.append('KDJ金叉')

            # 应用超买扣分和动能加分
            strength = max(0, strength - overbought_penalty + min(2, momentum_bonus))
            strength = min(10, strength)

            # === 钟摆位置评估（更严格的分级）===
            if dev_ma20 <= 2:
                pendulum = '回踩MA20附近'
                pendulum_score = 4  # 最佳做T位置
            elif dev_ma20 <= 4:
                pendulum = '略高于MA20'
                pendulum_score = 3
            elif dev_ma20 <= 6:
                pendulum = '偏高'
                pendulum_score = 2
            elif dev_ma20 <= 10:
                pendulum = '明显偏高'
                pendulum_score = 1
            else:
                pendulum = '过度偏高'
                pendulum_score = 0

            # 超买信号降低钟摆评分
            if overbought_penalty >= 2:
                pendulum_score = max(0, pendulum_score - 1)

            # === 做T适合度（钟摆位置是关键因素）===
            if strength >= 6 and pendulum_score >= 3 and overbought_penalty == 0:
                t0_label = '⭐⭐⭐'  # 趋势强+回踩到位+无超买
            elif strength >= 5 and pendulum_score >= 2 and overbought_penalty <= 1:
                t0_label = '⭐⭐'    # 趋势好+位置尚可
            elif strength >= 4 and pendulum_score >= 2:
                t0_label = '⭐'      # 基本可做
            else:
                t0_label = '-'       # 不适合（偏高或趋势弱）

            # === 均线排列描述 ===
            if perfect_bull and price > ma120:
                ma_desc = '完美多头(MA5>10>20>60>120)'
            elif perfect_bull:
                ma_desc = '强势多头(MA5>10>20>60)'
            elif strong_bull:
                ma_desc = '多头(MA5>10>20≈60)'
            else:
                ma_desc = '基本多头(MA5>10>20)'

            result = {
                'code': stock_code,
                'name': name,
                'price': price,
                '_ma20': ma20,  # 保存MA20值，用于实时偏离度计算
                'strength': strength,
                'ma_desc': ma_desc,
                'dev_ma20': dev_ma20,
                'dev_ma60': dev_ma60,
                'dev_ma120': dev_ma120,
                'ma20_slope': ma20_slope,
                'pendulum': pendulum,
                'pendulum_score': pendulum_score,
                't0_label': t0_label,
                'highs_rising': highs_rising,
                'lows_rising': lows_rising,
                'change_20d': change_20d,
                'overbought_flags': overbought_flags,
                'overbought_penalty': overbought_penalty,
                'momentum_flags': momentum_flags,
                'fund_score': 0,
                'fund_max': 10,
                'combined_score': strength,  # 默认等于技术面强度
            }

            # 轻量基本面评分（如果未禁用）
            if not self.no_fundamental:
                try:
                    fa = FundamentalAnalyzer(stock_code, name)
                    fa.fetch_financial_data()
                    fa.fetch_valuation_data()
                    light = fa.get_light_score()
                    result['fund_score'] = light['score']
                    result['fund_max'] = light['max_score']
                    # 综合得分 = 技术面强度(0-10)*0.4 + 基本面(0-10)*0.4 + 钟摆位置(0-4→0-10)*0.2
                    pendulum_norm = min(10, pendulum_score * 2.5)  # 归一化到 0-10
                    result['combined_score'] = round(strength * 0.4 + light['score'] * 0.4 + pendulum_norm * 0.2, 1)
                except Exception:
                    pendulum_norm = min(10, pendulum_score * 2.5)
                    result['combined_score'] = round(strength * 0.4 + pendulum_norm * 0.2, 1)  # 基本面失败按0分

            return result

        except Exception:
            return None

    def _enrich_with_realtime(self, results):
        """用 adata 实时行情补充最新价格（交易日盘中有效）"""
        codes = [r['code'] for r in results]
        try:
            rt = DataSource.get_realtime_quote(codes)
            if rt is None or rt.empty:
                return False
            # 构建 code -> row 映射
            rt_map = {}
            code_col = 'stock_code' if 'stock_code' in rt.columns else ('code' if 'code' in rt.columns else None)
            if code_col is None:
                return False
            for _, row in rt.iterrows():
                rt_map[str(row[code_col])] = row

            updated = 0
            for r in results:
                row = rt_map.get(r['code'])
                if row is None:
                    continue
                # 获取实时价格
                price_col = 'price' if 'price' in row.index else ('trade_price' if 'trade_price' in row.index else None)
                if price_col and pd.notna(row[price_col]) and float(row[price_col]) > 0:
                    rt_price = float(row[price_col])
                    r['rt_price'] = rt_price
                    # 用实时价格重新计算偏离度
                    r['rt_dev_ma20'] = (rt_price - r.get('_ma20', r['price'])) / r.get('_ma20', r['price']) * 100 if r.get('_ma20', 0) > 0 else r['dev_ma20']
                    # 涨跌幅
                    chg_col = 'change_pct' if 'change_pct' in row.index else ('pct_chg' if 'pct_chg' in row.index else None)
                    if chg_col and pd.notna(row[chg_col]):
                        r['rt_change'] = float(row[chg_col])
                    updated += 1
            return updated > 0
        except Exception:
            return False

    def run(self):
        """执行选股"""
        print("\n" + "=" * 70)
        print("📊 趋势选股报告")
        print("=" * 70)
        print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📋 投资哲学: 顺大势（均线多头排列+趋势向上），逆小势（钟摆回摆至均线附近）")

        # 获取股票池
        stock_pool = self.get_stock_pool()
        if not stock_pool:
            print("❌ 无法获取股票池")
            return

        total = len(stock_pool)
        print(f"\n🔍 开始分析 {total} 只股票...")
        print(f"   筛选条件: 均线多头排列 + MA20向上 + 偏离MA60<20%")

        # 逐个分析
        results = []
        for i, code in enumerate(stock_pool):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"   进度: {i + 1}/{total}...")

            result = self.analyze_single_stock(code)
            if result:
                results.append(result)

            # 控制请求频率（baostock 不限流，降低 sleep 频率）
            if (i + 1) % 20 == 0:
                time.sleep(0.2)

        if not results:
            print("\n⚠️ 未找到符合条件的股票")
            print("   建议：放宽筛选范围或更换股票池")
            return

        # 尝试用实时行情补充最新价格
        has_realtime = self._enrich_with_realtime(results)

        # 按综合得分排序（技术面*0.5 + 基本面*0.5），优先钟摆回踩到位
        if self.no_fundamental:
            results.sort(key=lambda x: (x['strength'], x['pendulum_score'], -abs(x.get('rt_dev_ma20', x['dev_ma20']))), reverse=True)
        else:
            results.sort(key=lambda x: (x['combined_score'], x['pendulum_score'], x['strength'], -abs(x.get('rt_dev_ma20', x['dev_ma20']))), reverse=True)

        # 输出结果
        top_results = results[:self.top_n]
        self.results = top_results

        # 数据日期说明
        print(f"\n━━━ 数据说明 ━━━")
        if has_realtime:
            print(f"   历史K线: baostock（可能延迟1个交易日）")
            print(f"   实时价格: adata（标记 [实时]，盘中自动更新）")
        else:
            print(f"   数据源: baostock（历史K线，可能延迟1个交易日）")
            print(f"   ⚠️ 实时行情不可用，价格为最近收盘价")

        print(f"\n━━━ 筛选结果：{len(results)} 只股票符合趋势向上条件 ━━━")
        sort_label = "综合得分" if not self.no_fundamental else "趋势强度"
        print(f"   显示前 {len(top_results)} 只（按{sort_label}+钟摆位置排序）")
        print(f"   优先展示回踩MA20附近的股票（偏离度低=买点好）\n")

        # 表头
        if self.no_fundamental:
            print(f"{'排名':<4} {'代码':<8} {'名称':<8} {'价格':>10} {'强度':>4} {'均线排列':<24} {'偏离MA20':>8} {'偏离MA60':>8} {'钟摆位置':<14} {'做T':>4}")
            print("-" * 115)
            for i, r in enumerate(top_results, 1):
                price_str, dev_str = self._format_price_dev(r, has_realtime)
                print(f"{i:<4} {r['code']:<8} {r['name']:<8} {price_str:>10} {r['strength']:>3}/10 {r['ma_desc']:<24} {dev_str:>8} {r['dev_ma60']:>+7.1f}% {r['pendulum']:<14} {r['t0_label']:>4}")
        else:
            print(f"{'排名':<4} {'代码':<8} {'名称':<8} {'价格':>10} {'技术':>4} {'基本面':>5} {'综合':>4} {'均线排列':<24} {'偏离MA20':>8} {'钟摆位置':<14} {'做T':>4}")
            print("-" * 125)
            for i, r in enumerate(top_results, 1):
                price_str, dev_str = self._format_price_dev(r, has_realtime)
                print(f"{i:<4} {r['code']:<8} {r['name']:<8} {price_str:>10} {r['strength']:>3}/10 {r['fund_score']:>3}/10 {r['combined_score']:>4.1f} {r['ma_desc']:<24} {dev_str:>8} {r['pendulum']:<14} {r['t0_label']:>4}")

        # 最佳做T候选（严格：偏离MA20 < 5% + 无超买信号）
        t0_candidates = [r for r in top_results
                         if r['pendulum_score'] >= 2 and r['strength'] >= 5
                         and abs(r.get('rt_dev_ma20', r['dev_ma20'])) <= 5
                         and r.get('overbought_penalty', 0) == 0]
        if t0_candidates:
            print(f"\n━━━ 最佳做T候选（趋势强 + 回踩均线 + MACD/KDJ健康）━━━")
            print(f"   这些股票趋势向上、钟摆回摆至MA20附近、无超买信号\n")
            for r in t0_candidates[:10]:
                trend_def = ''
                if r['highs_rising'] and r['lows_rising']:
                    trend_def = '标准上升趋势(高低点递增)'
                elif r['highs_rising']:
                    trend_def = '高点递增'
                elif r['lows_rising']:
                    trend_def = '低点递增'
                dev_val = r.get('rt_dev_ma20', r['dev_ma20'])
                price_val = r.get('rt_price', r['price'])
                mom_str = ' '.join(r.get('momentum_flags', []))
                if mom_str:
                    mom_str = f' | {mom_str}'
                print(f"   ⭐ {r['code']} {r['name']} ¥{price_val:.2f} | 强度{r['strength']}/10 | 偏离MA20:{dev_val:+.1f}% | {trend_def}{mom_str}")
        else:
            print(f"\n━━━ 做T候选 ━━━")
            print("   当前无理想做T候选（趋势向上但钟摆偏高或指标超买，建议等待回踩）")

        # MACD/KDJ 超买预警
        overbought_stocks = [r for r in top_results if r.get('overbought_penalty', 0) >= 1]
        if overbought_stocks:
            print(f"\n━━━ ⚠️ MACD/KDJ超买预警（趋势向上但短期接盘风险高）━━━")
            for r in overbought_stocks[:8]:
                flags_str = ', '.join(r.get('overbought_flags', []))
                dev_val = r.get('rt_dev_ma20', r['dev_ma20'])
                print(f"   ⚠️ {r['code']} {r['name']} 偏离MA20:{dev_val:+.1f}% | {flags_str} — 建议等MACD/KDJ修复后再买入")

        # 高位提醒（偏离度高但无超买信号的）
        high_stocks = [r for r in top_results
                       if r.get('rt_dev_ma20', r['dev_ma20']) > 5
                       and r.get('overbought_penalty', 0) == 0]
        if high_stocks:
            print(f"\n━━━ ⚠️ 高位提醒（偏离MA20 > 5%，追高风险大）━━━")
            for r in high_stocks[:5]:
                dev_val = r.get('rt_dev_ma20', r['dev_ma20'])
                print(f"   ⚠️ {r['code']} {r['name']} 偏离MA20:{dev_val:+.1f}% — 建议等回调至MA20附近再买入")

        # 内功提醒
        print(f"\n━━━ 内功提醒 ━━━")
        if self.no_fundamental:
            print("⚠️ 技术筛选只是「望远镜」，帮你缩小范围")
            print("   选出的股票还需要：")
            print("   1. 基本面验证（显微镜）— 理解趋势向上的原因")
            print("   2. 前瞻判断 — 评估趋势能否持续")
            print("   3. 交易决策 — 在钟摆回摆至均线附近时出手")
            print("   记住：内功为本（基本面），招式为辅（技术面）")
        else:
            print("📋 已融合基本面（内功）+ 技术面（招式）综合排序")
            print("   综合得分 = 技术面强度×50% + 基本面评分×50%")
            print("   基本面评分包含：ROE、营收增长率、PE估值")
            print("   定性因素仍需您自行判断：管理层诚信、公司文化、行业前景")
            print("   建议对排名靠前的股票使用 analyze_stock_simple.py 做详细分析")

        print(f"\n{'=' * 70}")
        print(f"⏰ 报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 共分析 {total} 只股票，筛选出 {len(results)} 只趋势向上")
        print(f"{'=' * 70}\n")

    @staticmethod
    def _format_price_dev(r, has_realtime):
        """格式化价格和偏离度显示（带实时标记）"""
        if has_realtime and 'rt_price' in r:
            price_str = f"{r['rt_price']:.2f}*"
            dev_str = f"{r['rt_dev_ma20']:+.1f}%*"
        else:
            price_str = f"{r['price']:.2f}"
            dev_str = f"{r['dev_ma20']:+.1f}%"
        return price_str, dev_str


def main():
    parser = argparse.ArgumentParser(description='趋势选股 — 基于「内功+招式」投资哲学')
    parser.add_argument('--index', type=str, help='指数代码: hs300, zz500, sz50, zz1000')
    parser.add_argument('--sector', type=str, help='板块名称，如: 白酒, 新能源, 半导体')
    parser.add_argument('--top', type=int, default=30, help='显示前N只股票（默认30）')
    parser.add_argument('--no-fundamental', action='store_true', help='跳过基本面分析（加速选股）')

    args = parser.parse_args()

    selector = TrendStockSelector(
        index=args.index,
        sector=args.sector,
        top_n=args.top,
        no_fundamental=args.no_fundamental,
    )
    selector.run()


if __name__ == "__main__":
    main()
