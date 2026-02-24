#!/usr/bin/env python3
"""
趋势选股工具（高性能版）
基于「顺大势」投资哲学，筛选趋势向上的股票

核心理念：
- 公设一：价格围绕价值波动 → 均线 = 价值中枢
- 公设二：钟摆式过度波动 → 均线偏离度 = 钟摆位置
- 顺大势：只选均线多头排列、趋势方向向上的股票
- 逆小势：标注钟摆回摆至均线附近的最佳做T候选

性能优化：
- 磁盘缓存：日K线数据当日缓存，重复运行秒出结果
- 两阶段筛选：先快速技术面过滤，通过的才做基本面（减少80%网络请求）
- 多指数合并：支持 --index core（沪深300+上证50去重）
- 并发获取：使用线程池并发拉取K线数据
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
from fundamental_analyzer import FundamentalAnalyzer, cleanup_fundamental_cache
from data_source import DataSource
from technical import (
    calculate_ma, calculate_macd, calculate_volume_ma, calculate_kdj, calculate_rsi,
    calculate_bollinger, detect_highs_lows, analyze_ma_alignment, _safe_ma,
    detect_topping_signals, detect_bottoming_signals,
)


class TrendStockSelector:
    """趋势选股器 — 基于均线+趋势+钟摆模型（高性能版）"""

    # 预定义指数映射
    INDEX_MAP = {
        'hs300': ('沪深300', ['sh.000300']),
        'zz500': ('中证500', ['sh.000905']),
        'sz50':  ('上证50',  ['sh.000016']),
        'core':  ('核心指数(沪深300+上证50)', ['sh.000300', 'sh.000016']),
        'wide':  ('宽基指数(沪深300+中证500)', ['sh.000300', 'sh.000905']),
    }

    def __init__(self, index=None, sector=None, top_n=30, no_fundamental=False):
        self.index = index
        self.sector = sector
        self.top_n = top_n
        self.no_fundamental = no_fundamental
        self.results = []
        self.stock_names = {}  # code -> name 映射

    def get_stock_pool(self):
        """获取股票池"""
        try:
            if self.index:
                return self._get_index_stocks()
            elif self.sector:
                return self._get_sector_stocks()
            else:
                # 默认使用核心指数（而非全A股），大幅提速
                print("💡 未指定指数，默认使用核心指数(沪深300+上证50)，可用 --index wide 扩大范围")
                self.index = 'core'
                return self._get_index_stocks()
        except Exception as e:
            print(f"❌ 获取股票池失败: {e}")
            return []

    def _get_index_stocks(self):
        """获取指数成分股，支持合并多指数去重"""
        key = self.index.lower()
        
        if key == 'all':
            return self._get_all_a_stocks()
        
        if key == 'zz1000':
            print(f"⚠️ baostock 不支持中证1000成分股查询，将从全A股中选股...")
            return self._get_all_a_stocks()
        
        if key not in self.INDEX_MAP:
            print(f"⚠️ 不支持的指数: {self.index}")
            print(f"   支持: {', '.join(self.INDEX_MAP.keys())}, all(全A股)")
            print("   将使用 core（沪深300+上证50）")
            key = 'core'

        name, index_codes = self.INDEX_MAP[key]
        print(f"📊 从{name}中选股...")

        all_codes = {}  # code -> name，用于去重
        for idx_code in index_codes:
            try:
                df = DataSource.get_index_stocks(idx_code)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        code = row['代码']
                        if code not in all_codes:
                            all_codes[code] = row['名称']
            except Exception as e:
                print(f"⚠️ 获取 {idx_code} 成分股失败: {e}")

        if all_codes:
            self.stock_names.update(all_codes)
            codes = list(all_codes.keys())
            print(f"✅ 获取到 {len(codes)} 只成分股（已去重）")
            return codes

        # 备用方案
        print("⚠️ 获取指数成分股失败，使用备用方案...")
        return self._get_all_a_stocks()[:300]

    def _get_sector_stocks(self):
        """获取板块成分股（baostock 不支持板块，使用全市场）"""
        print(f"⚠️ baostock 不支持板块筛选，将从全市场选股...")
        return self._get_all_a_stocks()

    def _get_all_a_stocks(self):
        """获取全A股列表（使用 baostock）"""
        print("📊 获取全A股列表（较慢，建议使用 --index core）...")
        try:
            df = DataSource.get_stock_list()
            if df is not None and not df.empty:
                codes = df['代码'].tolist()
                for _, row in df.iterrows():
                    self.stock_names[row['代码']] = row['名称']
                print(f"✅ 获取到 {len(codes)} 只A股")
                return codes
        except Exception as e:
            print(f"❌ 获取A股列表失败: {e}")
            return []

    def _fetch_stock_data(self, stock_code, days=400):
        """获取股票数据（自动利用磁盘+内存缓存）"""
        # 使用日期字符串（不含时分秒），确保同一天的缓存 key 一致
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
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
            calculate_volume_ma(df)
            calculate_macd(df)

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

            # === 见顶/出货检测 ===
            # 核心场景：MA20向上但短期已开始连续下跌
            topping = detect_topping_signals(df, price)
            topping_score = topping['score']
            topping_level = topping['level']

            # 见顶信号强烈的直接排除
            if topping_score >= 70:
                return None

            # === 多级别均线偏离度（钟摆位置）===
            # MA5=超短期情绪, MA10=短期情绪, MA20=中期中枢, MA60=季度趋势
            dev_ma5 = (price - ma5) / ma5 * 100
            dev_ma10 = (price - ma10) / ma10 * 100
            dev_ma20 = (price - ma20) / ma20 * 100
            dev_ma60 = (price - ma60) / ma60 * 100
            dev_ma120 = (price - ma120) / ma120 * 100

            # 多级别过度偏离过滤（避免追高）
            if dev_ma60 > 20:        # MA60绳子太紧
                return None
            if dev_ma20 > 12:        # MA20偏离过大，追高风险极大
                return None
            if dev_ma5 > 7 and dev_ma20 > 8:  # 短期+中期同时过热
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

            # === 多级别钟摆位置评估（MA5/MA10/MA20联合判断）===
            # 最佳买点：价格回踩至均线簇附近（短中期均线收敛）
            # 高风险：价格远离所有均线（追高陷阱）
            if dev_ma5 <= 1 and dev_ma10 <= 2 and dev_ma20 <= 3:
                pendulum = '均线簇收敛★'
                pendulum_score = 4  # 短中期均线收敛，最佳安全买点
            elif dev_ma5 <= 2 and dev_ma10 <= 3 and dev_ma20 <= 4:
                pendulum = '回踩均线附近'
                pendulum_score = 3  # 接近均线，安全性高
            elif dev_ma5 <= 3 and dev_ma20 <= 5:
                pendulum = '略高于均线'
                pendulum_score = 2  # 偏高但可接受
            elif dev_ma20 <= 8 and dev_ma5 <= 5:
                pendulum = '偏高⚠'
                pendulum_score = 1  # 有一定追高风险
            elif dev_ma20 <= 8:
                pendulum = '短期过热⚠'
                pendulum_score = 0  # 短期情绪过热
            else:
                pendulum = '高位风险🔴'
                pendulum_score = -1  # 追高风险极大

            # === 做T适合度（趋势+钟摆双重确认）===
            # 核心：趋势向上是必要条件，钟摆回摆至均线附近才是最佳时机
            if strength >= 7 and pendulum_score >= 3:
                t0_label = '⭐⭐⭐'   # 趋势强+位置安全
            elif strength >= 5 and pendulum_score >= 2:
                t0_label = '⭐⭐'     # 趋势好+位置可接受
            elif strength >= 4 and pendulum_score >= 1:
                t0_label = '⭐'       # 趋势尚可+位置偏高
            else:
                t0_label = '-'        # 不适合做T（位置不佳或趋势不强）

            # === 均线排列描述 ===
            if perfect_bull and price > ma120:
                ma_desc = '完美多头(MA5>10>20>60>120)'
            elif perfect_bull:
                ma_desc = '强势多头(MA5>10>20>60)'
            elif strong_bull:
                ma_desc = '多头(MA5>10>20≈60)'
            else:
                ma_desc = '基本多头(MA5>10>20)'

            # === 见顶信号降级 ===
            if topping_score >= 50:
                t0_label = '⚠️'
                pendulum = f'{pendulum}(见顶⚠)'
                strength = max(0, strength - 3)
            elif topping_score >= 30:
                t0_label = '⚠️' if t0_label != '-' else '-'
                pendulum = f'{pendulum}(见顶⚠)'
                strength = max(0, strength - 2)

            # 基本面评分在两阶段筛选的第二阶段统一处理，此处先返回技术面结果
            result = {
                'code': stock_code,
                'name': name,
                'price': price,
                'strength': strength,
                'ma_desc': ma_desc,
                'dev_ma5': dev_ma5,
                'dev_ma10': dev_ma10,
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
                'topping_score': topping_score,
                'topping_level': topping_level,
                'topping_signals': topping['signals'],
                'fund_score': 0,
                'fund_max': 10,
                'combined_score': strength,  # 默认等于技术面强度，基本面在第二阶段补充
            }

            return result

        except Exception:
            return None

    def _batch_fetch_and_analyze(self, stock_pool):
        """
        两阶段筛选（串行获取 + 磁盘缓存加速）
        第一阶段：获取K线 + 纯技术面快速过滤（baostock串行，磁盘缓存秒回）
        第二阶段：仅对通过的股票做基本面评分（大幅减少akshare请求）
        """
        total = len(stock_pool)
        results = []
        start_time = time.time()

        print(f"   ⚡ 磁盘缓存加速（首次需要网络获取，第二次运行秒出）")

        # 第一阶段：技术面快速筛选
        for i, code in enumerate(stock_pool):
            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"   进度: {i + 1}/{total} ({speed:.0f}只/秒, 已筛出{len(results)}只)")
            try:
                result = self.analyze_single_stock(code)
                if result:
                    results.append(result)
            except Exception:
                pass

        elapsed = time.time() - start_time
        print(f"   ✅ 技术面筛选完成：{len(results)}/{total} 通过，耗时 {elapsed:.1f}s")

        # 第二阶段：基本面评分（仅对技术面通过的股票，大幅减少请求量）
        if not self.no_fundamental and results:
            print(f"   📊 基本面评分：{len(results)} 只股票（仅技术面通过的）...")
            fund_start = time.time()
            for i, r in enumerate(results):
                try:
                    fa = FundamentalAnalyzer(r['code'], r['name'])
                    fa.fetch_financial_data()
                    fa.fetch_valuation_data()
                    light = fa.get_light_score()
                    r['fund_score'] = light['score']
                    r['fund_max'] = light['max_score']
                    r['combined_score'] = round(r['strength'] * 0.5 + light['score'] * 0.5, 1)
                except Exception:
                    r['combined_score'] = r['strength'] * 0.5
                if (i + 1) % 10 == 0:
                    time.sleep(0.3)  # akshare 限流保护
            print(f"   ✅ 基本面评分完成，耗时 {time.time() - fund_start:.1f}s")

        return results

    def run(self):
        """执行选股"""
        run_start = time.time()
        print("\n" + "=" * 70)
        print("📊 趋势选股报告（高性能版）")
        print("=" * 70)
        print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📋 投资哲学: 顺大势（均线多头排列+趋势向上），逆小势（钟摆回摆至均线附近）")

        DataSource.cleanup_old_disk_cache(keep_days=7)
        cleanup_fundamental_cache(keep_days=3)
        DataSource.reset_stats()

        stock_pool = self.get_stock_pool()
        if not stock_pool:
            print("❌ 无法获取股票池")
            return

        total = len(stock_pool)
        print(f"\n🔍 开始分析 {total} 只股票...")
        print(f"   筛选条件: 均线多头排列 + MA20向上 + 多级别偏离度控制 + 见顶检测")
        print(f"   过滤规则: MA60偏离>20% | MA20偏离>12% | MA5>7%且MA20>8% | 见顶评分>=70 → 排除")

        # 批量预加载当日实时行情（交易时段自动补充当日数据）
        DataSource.preload_realtime_prices(stock_pool)

        # 两阶段筛选 + 并发获取
        results = self._batch_fetch_and_analyze(stock_pool)

        if not results:
            print("\n⚠️ 未找到符合条件的股票")
            print("   建议：放宽筛选范围或更换股票池")
            return

        # 按综合得分排序（技术面*0.5 + 基本面*0.5），同分优先钟摆位置好的
        if self.no_fundamental:
            results.sort(key=lambda x: (x['strength'], x['pendulum_score'], -x['dev_ma20']), reverse=True)
        else:
            results.sort(key=lambda x: (x['combined_score'], x['pendulum_score'], -x['dev_ma20']), reverse=True)

        # 输出结果
        top_results = results[:self.top_n]
        self.results = top_results

        print(f"\n━━━ 筛选结果：{len(results)} 只股票符合趋势向上条件 ━━━")
        sort_label = "综合得分" if not self.no_fundamental else "趋势强度"
        print(f"   显示前 {len(top_results)} 只（按{sort_label}排序）\n")

        # 表头
        if self.no_fundamental:
            print(f"{'排名':<4} {'代码':<8} {'名称':<12} {'价格':>8} {'强度':>4} {'均线排列':<26} {'MA5':>5} {'MA10':>5} {'MA20':>5} {'钟摆位置':<16} {'做T':>6}")
            print("-" * 120)
            for i, r in enumerate(top_results, 1):
                print(f"{i:<4} {r['code']:<8} {r['name']:<12} {r['price']:>8.2f} {r['strength']:>3}/10 {r['ma_desc']:<26} {r['dev_ma5']:>+4.0f}% {r['dev_ma10']:>+4.0f}% {r['dev_ma20']:>+4.0f}% {r['pendulum']:<16} {r['t0_label']:>6}")
        else:
            print(f"{'排名':<4} {'代码':<8} {'名称':<12} {'价格':>8} {'技术':>4} {'基本面':>5} {'综合':>4} {'均线排列':<26} {'MA5':>5} {'MA10':>5} {'MA20':>5} {'钟摆位置':<16} {'做T':>6}")
            print("-" * 140)
            for i, r in enumerate(top_results, 1):
                print(f"{i:<4} {r['code']:<8} {r['name']:<12} {r['price']:>8.2f} {r['strength']:>3}/10 {r['fund_score']:>3}/10 {r['combined_score']:>4.1f} {r['ma_desc']:<26} {r['dev_ma5']:>+4.0f}% {r['dev_ma10']:>+4.0f}% {r['dev_ma20']:>+4.0f}% {r['pendulum']:<16} {r['t0_label']:>6}")

        # 最佳做T候选（钟摆位置>=3 表示回踩均线附近）
        t0_candidates = [r for r in top_results if r['pendulum_score'] >= 3 and r['strength'] >= 5]
        # 次优候选（钟摆位置>=2 略高于均线但可接受）
        t0_secondary = [r for r in top_results if r['pendulum_score'] == 2 and r['strength'] >= 5 and r not in t0_candidates]

        if t0_candidates:
            print(f"\n━━━ 最佳做T候选（趋势强 + 回踩均线簇附近）━━━")
            print(f"   这些股票趋势向上且钟摆回摆至均线附近，MA5/MA10/MA20收敛，安全边际高\n")
            for r in t0_candidates[:10]:
                trend_def = ''
                if r['highs_rising'] and r['lows_rising']:
                    trend_def = '标准上升趋势(高低点递增)'
                elif r['highs_rising']:
                    trend_def = '高点递增'
                elif r['lows_rising']:
                    trend_def = '低点递增'
                dev_str = f"MA5:{r['dev_ma5']:+.1f}% MA10:{r['dev_ma10']:+.1f}% MA20:{r['dev_ma20']:+.1f}%"
                print(f"   ⭐ {r['code']} {r['name']} ¥{r['price']:.2f} | 强度{r['strength']}/10 | {r['pendulum']} | {dev_str} | {trend_def}")
        else:
            print(f"\n━━━ 最佳做T候选 ━━━")
            print("   当前无理想做T候选（趋势向上但钟摆偏高，建议等待回踩）")

        if t0_secondary:
            print(f"\n━━━ 次优做T候选（趋势好但略高于均线，可小仓位参与）━━━")
            for r in t0_secondary[:5]:
                trend_def = ''
                if r['highs_rising'] and r['lows_rising']:
                    trend_def = '高低点递增'
                elif r['lows_rising']:
                    trend_def = '低点递增'
                dev_str = f"MA5:{r['dev_ma5']:+.1f}% MA10:{r['dev_ma10']:+.1f}% MA20:{r['dev_ma20']:+.1f}%"
                print(f"   ○ {r['code']} {r['name']} ¥{r['price']:.2f} | 强度{r['strength']}/10 | {r['pendulum']} | {dev_str} | {trend_def}")

        # 高位风险提示
        high_risk = [r for r in top_results if r['pendulum_score'] <= 0]
        if high_risk:
            print(f"\n━━━ ⚠️ 高位风险提示（以下股票趋势好但偏离均线过大，追高有风险）━━━")
            for r in high_risk[:5]:
                dev_str = f"MA5:{r['dev_ma5']:+.1f}% MA10:{r['dev_ma10']:+.1f}% MA20:{r['dev_ma20']:+.1f}%"
                print(f"   ⚠️ {r['code']} {r['name']} ¥{r['price']:.2f} | {r['pendulum']} | {dev_str} | 建议等待回踩MA20后再介入")

        # 见顶风险提示（MA20向上但短期转弱）
        topping_risk = [r for r in top_results if r.get('topping_score', 0) >= 30]
        if topping_risk:
            print(f"\n━━━ 🔴 见顶/出货风险提示（MA20向上但短期出现转弱信号）━━━")
            print(f"   ⚠️ 以下股票虽然MA20仍向上，但短期出现见顶/主力出货迹象\n")
            for r in topping_risk[:10]:
                level_emoji = '🔴' if r['topping_score'] >= 50 else '🟡'
                print(f"   {level_emoji} {r['code']} {r['name']} ¥{r['price']:.2f} | 见顶评分:{r['topping_score']} ({r['topping_level']})")
                for sig in r.get('topping_signals', [])[:3]:
                    print(f"      → {sig}")

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

        total_elapsed = time.time() - run_start
        print(f"\n{'=' * 70}")
        print(f"⏰ 报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 共分析 {total} 只股票，筛选出 {len(results)} 只趋势向上")
        DataSource.print_cache_stats()
        print(f"⚡ 总耗时: {total_elapsed:.1f}s")
        print(f"{'=' * 70}\n")


class BottomReversalSelector:
    """底部反弹选股器 — 寻找基本面优秀但被低估、即将触底反弹的股票"""

    INDEX_MAP = TrendStockSelector.INDEX_MAP

    def __init__(self, index=None, sector=None, top_n=30, no_fundamental=False):
        self.index = index
        self.sector = sector
        self.top_n = top_n
        self.no_fundamental = no_fundamental
        self.results = []
        self.stock_names = {}
        self._pool_helper = TrendStockSelector(index=index, sector=sector, top_n=top_n)

    def get_stock_pool(self):
        pool = self._pool_helper.get_stock_pool()
        self.stock_names = self._pool_helper.stock_names
        return pool

    def _fetch_stock_data(self, stock_code, days=400):
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            df = DataSource.get_stock_hist(
                stock_code=stock_code, start_date=start_date, end_date=end_date,
                adjust='qfq', period='daily'
            )
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        return None

    def analyze_single_stock(self, stock_code):
        """分析单只股票的底部反弹潜力"""
        try:
            df = self._fetch_stock_data(stock_code)
            if df is None or df.empty or len(df) < 60:
                return None

            name = self.stock_names.get(stock_code, stock_code)

            # ST 股票排除
            if 'ST' in name or 'st' in name:
                return None

            calculate_ma(df, windows=[5, 10, 20, 60, 120, 250])
            calculate_macd(df)
            calculate_kdj(df)
            calculate_rsi(df)
            calculate_volume_ma(df)
            calculate_bollinger(df)

            latest = df.iloc[-1]
            price = latest['收盘']

            ma5 = _safe_ma(latest, 'MA5')
            ma10 = _safe_ma(latest, 'MA10')
            ma20 = _safe_ma(latest, 'MA20')
            ma60 = _safe_ma(latest, 'MA60')
            ma120 = _safe_ma(latest, 'MA120')

            if any(v is None for v in [ma5, ma10, ma20, ma60]):
                return None

            # === 第二层：跌幅 / 低估过滤 ===
            dev_ma20 = (price - ma20) / ma20 * 100
            dev_ma60 = (price - ma60) / ma60 * 100

            # 距近60日高点的跌幅
            high_60d = df.tail(60)['最高'].max() if len(df) >= 60 else df['最高'].max()
            drawdown_60d = (high_60d - price) / high_60d * 100

            # 必须满足至少一个"跌够了"条件
            is_oversold = (
                price < ma60 or           # 价格在 MA60 以下
                dev_ma20 < -3 or          # 偏离 MA20 超过 -3%
                drawdown_60d >= 15         # 从60日高点跌幅 >= 15%
            )
            if not is_oversold:
                return None

            # 排除还在暴跌中的（近5日跌幅 > 15%，避免接飞刀）
            if len(df) >= 5:
                price_5d_ago = df.iloc[-5]['收盘']
                change_5d = (price - price_5d_ago) / price_5d_ago * 100
                if change_5d < -15:
                    return None

            # === 第三层：底部信号检测 ===
            bottoming = detect_bottoming_signals(df, price)
            bottom_score = bottoming['score']
            bottom_level = bottoming['level']

            if bottom_score < 25:
                return None

            # === 均线排列描述（底部特征）===
            alignment = analyze_ma_alignment(latest, price)
            ma_desc = alignment['desc']

            # 跌幅深度评分（0-10）：跌得越多，反弹空间越大
            depth_score = 0
            if drawdown_60d >= 30:
                depth_score = 10
            elif drawdown_60d >= 25:
                depth_score = 8
            elif drawdown_60d >= 20:
                depth_score = 6
            elif drawdown_60d >= 15:
                depth_score = 4
            elif drawdown_60d >= 10:
                depth_score = 2

            dev_ma120 = (price - ma120) / ma120 * 100 if ma120 and ma120 > 0 else 0

            result = {
                'code': stock_code,
                'name': name,
                'price': price,
                'bottom_score': bottom_score,
                'bottom_level': bottom_level,
                'bottom_signals': bottoming['signals'],
                'depth_score': depth_score,
                'drawdown_60d': drawdown_60d,
                'dev_ma20': dev_ma20,
                'dev_ma60': dev_ma60,
                'dev_ma120': dev_ma120,
                'ma_desc': ma_desc,
                'fund_score': 0,
                'fund_max': 10,
                'value_details': [],
                'is_value_trap': False,
                'combined_score': 0,
            }
            return result

        except Exception:
            return None

    def _batch_fetch_and_analyze(self, stock_pool):
        """两阶段筛选：先技术面快筛，再基本面评估"""
        total = len(stock_pool)
        results = []
        start_time = time.time()

        print(f"   ⚡ 磁盘缓存加速（首次需要网络获取，第二次运行秒出）")

        # 第一阶段：技术面筛选（底部信号）
        for i, code in enumerate(stock_pool):
            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"   进度: {i + 1}/{total} ({speed:.0f}只/秒, 已筛出{len(results)}只)")
            try:
                result = self.analyze_single_stock(code)
                if result:
                    results.append(result)
            except Exception:
                pass

        elapsed = time.time() - start_time
        print(f"   ✅ 技术面筛选完成：{len(results)}/{total} 有底部信号，耗时 {elapsed:.1f}s")

        # 第二阶段：基本面价值评估
        if not self.no_fundamental and results:
            print(f"   📊 基本面价值评估：{len(results)} 只候选...")
            fund_start = time.time()
            filtered = []
            for i, r in enumerate(results):
                try:
                    fa = FundamentalAnalyzer(r['code'], r['name'])
                    fa.fetch_financial_data()
                    fa.fetch_valuation_data()
                    value = fa.get_value_score()
                    r['fund_score'] = value['score']
                    r['fund_max'] = value['max_score']
                    r['value_details'] = value['details']
                    r['is_value_trap'] = value['is_value_trap']

                    # 基本面门槛：>= 5/10 且非价值陷阱
                    if value['score'] >= 5 and not value['is_value_trap']:
                        # 综合评分 = 基本面*0.4 + 底部信号*0.4 + 跌幅深度*0.2
                        r['combined_score'] = round(
                            value['score'] * 0.4 +
                            (r['bottom_score'] / 10) * 0.4 +
                            r['depth_score'] * 0.2,
                            1
                        )
                        filtered.append(r)
                    elif self.no_fundamental:
                        filtered.append(r)
                except Exception:
                    pass
                if (i + 1) % 10 == 0:
                    time.sleep(0.3)
            print(f"   ✅ 价值评估完成（{len(filtered)}/{len(results)}通过），耗时 {time.time() - fund_start:.1f}s")
            return filtered
        else:
            # 无基本面时只用技术面排序
            for r in results:
                r['combined_score'] = round(
                    (r['bottom_score'] / 10) * 0.6 + r['depth_score'] * 0.4,
                    1
                )
            return results

    def run(self):
        """执行底部反弹选股"""
        run_start = time.time()
        print("\n" + "=" * 70)
        print("📊 底部反弹选股报告")
        print("=" * 70)
        print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📋 策略: 基本面优秀(内功好) + 估值低估(被错杀) + 技术面见底(即将反弹)")

        DataSource.cleanup_old_disk_cache(keep_days=7)
        cleanup_fundamental_cache(keep_days=3)
        DataSource.reset_stats()

        stock_pool = self.get_stock_pool()
        if not stock_pool:
            print("❌ 无法获取股票池")
            return

        total = len(stock_pool)
        print(f"\n🔍 开始分析 {total} 只股票...")
        print(f"   筛选条件: 价格低于MA60 | MA20偏离<-3% | 60日跌幅>=15%")
        print(f"   底部信号: RSI超卖 + KDJ超卖金叉 + MACD底背离 + 缩量企稳 + 布林带下轨 + 均线金叉 + 反转K线")
        if not self.no_fundamental:
            print(f"   基本面门槛: 价值评分>=5/10，排除价值陷阱（低ROE+低PE）")

        DataSource.preload_realtime_prices(stock_pool)

        results = self._batch_fetch_and_analyze(stock_pool)

        if not results:
            print("\n⚠️ 未找到符合条件的底部反弹候选")
            print("   可能原因：市场整体偏强（没有大幅回调的好股票），或放宽范围 --index wide")
            return

        results.sort(key=lambda x: (x['combined_score'], x['bottom_score'], x['drawdown_60d']), reverse=True)

        top_results = results[:self.top_n]
        self.results = top_results

        print(f"\n━━━ 筛选结果：{len(results)} 只股票出现底部反弹信号 ━━━")
        sort_label = "综合得分" if not self.no_fundamental else "底部信号强度"
        print(f"   显示前 {len(top_results)} 只（按{sort_label}排序）\n")

        if self.no_fundamental:
            print(f"{'排名':<4} {'代码':<8} {'名称':<12} {'价格':>8} {'底部':>5} {'60日跌':>6} {'MA20':>6} {'MA60':>6} {'均线排列':<20} {'底部信号':<30}")
            print("-" * 130)
            for i, r in enumerate(top_results, 1):
                sig_str = '; '.join(r['bottom_signals'][:2]) if r['bottom_signals'] else '-'
                if len(sig_str) > 28:
                    sig_str = sig_str[:28] + '…'
                print(f"{i:<4} {r['code']:<8} {r['name']:<12} {r['price']:>8.2f} {r['bottom_score']:>4}/100 {r['drawdown_60d']:>5.0f}% {r['dev_ma20']:>+5.0f}% {r['dev_ma60']:>+5.0f}% {r['ma_desc']:<20} {sig_str}")
        else:
            print(f"{'排名':<4} {'代码':<8} {'名称':<12} {'价格':>8} {'底部':>5} {'价值':>4} {'综合':>4} {'60日跌':>6} {'MA20':>6} {'MA60':>6} {'均线排列':<20}")
            print("-" * 130)
            for i, r in enumerate(top_results, 1):
                print(f"{i:<4} {r['code']:<8} {r['name']:<12} {r['price']:>8.2f} {r['bottom_score']:>4}/100 {r['fund_score']:>3}/10 {r['combined_score']:>4.1f} {r['drawdown_60d']:>5.0f}% {r['dev_ma20']:>+5.0f}% {r['dev_ma60']:>+5.0f}% {r['ma_desc']:<20}")

        # 强底部信号候选（详细展示）
        strong = [r for r in top_results if r['bottom_score'] >= 60]
        medium = [r for r in top_results if 40 <= r['bottom_score'] < 60 and r not in strong]

        if strong:
            print(f"\n━━━ 强底部信号候选（底部评分>=60，反弹概率高）━━━\n")
            for r in strong[:10]:
                value_str = ' | '.join(r['value_details'][:3]) if r['value_details'] else '(纯技术面)'
                print(f"   🟢 {r['code']} {r['name']} ¥{r['price']:.2f} | 底部:{r['bottom_score']}/100 ({r['bottom_level']}) | 60日跌幅:-{r['drawdown_60d']:.1f}%")
                print(f"      价值: {value_str}")
                for sig in r['bottom_signals'][:3]:
                    print(f"      → {sig}")
                print()

        if medium:
            print(f"\n━━━ 中等底部信号候选（底部评分40-59，需关注确认信号）━━━\n")
            for r in medium[:8]:
                value_str = ' | '.join(r['value_details'][:3]) if r['value_details'] else '(纯技术面)'
                print(f"   🟡 {r['code']} {r['name']} ¥{r['price']:.2f} | 底部:{r['bottom_score']}/100 | 60日跌幅:-{r['drawdown_60d']:.1f}%")
                print(f"      价值: {value_str}")
                for sig in r['bottom_signals'][:2]:
                    print(f"      → {sig}")
                print()

        # 价值陷阱提示
        traps = [r for r in results if r.get('is_value_trap')]
        if traps:
            print(f"\n━━━ ⚠️ 疑似价值陷阱（已排除，仅供参考）━━━")
            for r in traps[:5]:
                print(f"   ⚠️ {r['code']} {r['name']} | {' | '.join(r['value_details'][:2])}")

        # 策略提醒
        print(f"\n━━━ 策略提醒 ━━━")
        print("📋 底部反弹选股 ≠ 无风险抄底，请注意：")
        print("   1. 底部信号是概率性的，不保证一定反弹")
        print("   2. 强底部信号 + 好基本面 = 胜率更高")
        print("   3. 分批建仓（如1/3仓位），设止损（如跌破前低-3%）")
        print("   4. 等待确认信号（放量阳线、突破MA5/MA10）再加仓")
        print("   5. 避免「接飞刀」— 近5日暴跌>15%的已自动排除")

        total_elapsed = time.time() - run_start
        print(f"\n{'=' * 70}")
        print(f"⏰ 报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 共分析 {total} 只股票，筛选出 {len(results)} 只底部反弹候选")
        DataSource.print_cache_stats()
        print(f"⚡ 总耗时: {total_elapsed:.1f}s")
        print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='选股工具 — 支持趋势选股和底部反弹选股',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
策略选项:
  trend    趋势选股（默认）— 均线多头排列+MA20向上，追涨不追高
  bottom   底部反弹选股 — 基本面好+跌够了+底部信号，抄底不抄死

指数选项:
  core   沪深300+上证50（默认，约320只，推荐日常使用）
  hs300  沪深300（300只）
  zz500  中证500（500只）
  sz50   上证50（50只，最快）
  wide   沪深300+中证500（约800只）
  all    全A股（5000+只，较慢）

示例:
  python3 select_stocks.py                          # 默认趋势选股
  python3 select_stocks.py --strategy bottom         # 底部反弹选股
  python3 select_stocks.py --strategy bottom --index wide  # 宽基底部反弹
  python3 select_stocks.py --index sz50              # 上证50趋势选股
  python3 select_stocks.py --no-fundamental          # 跳过基本面（纯技术面）
"""
    )
    parser.add_argument('--strategy', type=str, default='trend', choices=['trend', 'bottom'],
                        help='选股策略: trend(趋势,默认), bottom(底部反弹)')
    parser.add_argument('--index', type=str, help='指数: core(默认), hs300, zz500, sz50, wide, all')
    parser.add_argument('--sector', type=str, help='板块名称，如: 白酒, 新能源, 半导体')
    parser.add_argument('--top', type=int, default=30, help='显示前N只股票（默认30）')
    parser.add_argument('--no-fundamental', action='store_true', help='跳过基本面分析（纯技术面筛选更快）')
    args = parser.parse_args()

    if args.strategy == 'bottom':
        selector = BottomReversalSelector(
            index=args.index,
            sector=args.sector,
            top_n=args.top,
            no_fundamental=args.no_fundamental,
        )
    else:
        selector = TrendStockSelector(
            index=args.index,
            sector=args.sector,
            top_n=args.top,
            no_fundamental=args.no_fundamental,
        )
    selector.run()


if __name__ == "__main__":
    main()
