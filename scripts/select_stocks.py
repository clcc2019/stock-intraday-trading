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
from fundamental_analyzer import FundamentalAnalyzer
from data_source import DataSource
from technical import calculate_ma, detect_highs_lows, analyze_ma_alignment, _safe_ma


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

        # 清理旧缓存
        DataSource.cleanup_old_disk_cache(keep_days=3)

        # 获取股票池
        stock_pool = self.get_stock_pool()
        if not stock_pool:
            print("❌ 无法获取股票池")
            return

        total = len(stock_pool)
        print(f"\n🔍 开始分析 {total} 只股票...")
        print(f"   筛选条件: 均线多头排列 + MA20向上 + 多级别偏离度控制")
        print(f"   过滤规则: MA60偏离>20% | MA20偏离>12% | MA5>7%且MA20>8% → 排除")

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
        print(f"⚡ 总耗时: {total_elapsed:.1f}s（第二次运行有磁盘缓存更快）")
        print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='趋势选股 — 基于「内功+招式」投资哲学（高性能版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
指数选项:
  core   沪深300+上证50（默认，约320只，推荐日常使用）
  hs300  沪深300（300只）
  zz500  中证500（500只）
  sz50   上证50（50只，最快）
  wide   沪深300+中证500（约800只）
  all    全A股（5000+只，较慢）

示例:
  python3 select_stocks.py                      # 默认核心指数
  python3 select_stocks.py --index sz50         # 上证50（最快）
  python3 select_stocks.py --index wide         # 宽基指数
  python3 select_stocks.py --no-fundamental     # 跳过基本面（纯技术面）
"""
    )
    parser.add_argument('--index', type=str, help='指数: core(默认), hs300, zz500, sz50, wide, all')
    parser.add_argument('--sector', type=str, help='板块名称，如: 白酒, 新能源, 半导体')
    parser.add_argument('--top', type=int, default=30, help='显示前N只股票（默认30）')
    parser.add_argument('--no-fundamental', action='store_true', help='跳过基本面分析（纯技术面筛选更快）')
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
