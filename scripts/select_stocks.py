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

# 导入基本面分析模块和数据源适配层
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fundamental_analyzer import FundamentalAnalyzer
from data_source import DataSource


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
            'zz1000': ('中证1000', 'sh.000852'),
        }

        key = self.index.lower()
        if key not in index_map:
            print(f"⚠️ 不支持的指数: {self.index}，支持: {', '.join(index_map.keys())}")
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

        return []

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

            # 计算均线
            df['MA5'] = df['收盘'].rolling(window=5).mean()
            df['MA10'] = df['收盘'].rolling(window=10).mean()
            df['MA20'] = df['收盘'].rolling(window=20).mean()
            df['MA60'] = df['收盘'].rolling(window=60).mean()
            df['MA120'] = df['收盘'].rolling(window=120).mean()
            if len(df) >= 250:
                df['MA250'] = df['收盘'].rolling(window=250).mean()

            latest = df.iloc[-1]
            price = latest['收盘']
            name = self.stock_names.get(stock_code, stock_code)

            # === 均线排列分析 ===
            ma5 = latest['MA5']
            ma10 = latest['MA10']
            ma20 = latest['MA20']
            ma60 = latest['MA60']
            ma120 = latest['MA120']

            if any(np.isnan(x) for x in [ma5, ma10, ma20, ma60, ma120]):
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

            # === 趋势定义验证（高低点递增）===
            recent_20 = df.tail(20)
            recent_highs = []
            recent_lows = []
            for i in range(2, len(recent_20) - 2):
                row = recent_20.iloc[i]
                prev1 = recent_20.iloc[i - 1]
                prev2 = recent_20.iloc[i - 2]
                next1 = recent_20.iloc[i + 1]
                next2 = recent_20.iloc[i + 2]
                if row['最高'] >= prev1['最高'] and row['最高'] >= prev2['最高'] and row['最高'] >= next1['最高'] and row['最高'] >= next2['最高']:
                    recent_highs.append(row['最高'])
                if row['最低'] <= prev1['最低'] and row['最低'] <= prev2['最低'] and row['最低'] <= next1['最低'] and row['最低'] <= next2['最低']:
                    recent_lows.append(row['最低'])

            highs_rising = len(recent_highs) >= 2 and recent_highs[-1] > recent_highs[0]
            lows_rising = len(recent_lows) >= 2 and recent_lows[-1] > recent_lows[0]

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

            # === 钟摆位置评估 ===
            if dev_ma20 <= 2:
                pendulum = '回踩MA20附近'
                pendulum_score = 3  # 最佳做T位置
            elif dev_ma20 <= 5:
                pendulum = '略高于MA20'
                pendulum_score = 2
            elif dev_ma20 <= 10:
                pendulum = '偏高'
                pendulum_score = 1
            else:
                pendulum = '过度偏高'
                pendulum_score = 0

            # === 做T适合度 ===
            # 趋势向上 + 钟摆回摆至均线附近 = 最佳做T候选
            t0_score = min(3, pendulum_score)
            if strength >= 7:
                t0_label = '⭐⭐⭐'
            elif strength >= 5 and pendulum_score >= 2:
                t0_label = '⭐⭐'
            elif strength >= 4:
                t0_label = '⭐'
            else:
                t0_label = '-'

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
                    # 综合得分 = 技术面强度(0-10) * 0.5 + 基本面(0-10) * 0.5
                    result['combined_score'] = round(strength * 0.5 + light['score'] * 0.5, 1)
                except Exception:
                    result['combined_score'] = strength * 0.5  # 基本面失败按0分

            return result

        except Exception:
            return None

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

            # 控制请求频率
            if (i + 1) % 5 == 0:
                time.sleep(0.3)

        if not results:
            print("\n⚠️ 未找到符合条件的股票")
            print("   建议：放宽筛选范围或更换股票池")
            return

        # 按综合得分排序（技术面*0.5 + 基本面*0.5）
        if self.no_fundamental:
            results.sort(key=lambda x: (x['strength'], -x['dev_ma20']), reverse=True)
        else:
            results.sort(key=lambda x: (x['combined_score'], x['strength'], -x['dev_ma20']), reverse=True)

        # 输出结果
        top_results = results[:self.top_n]
        self.results = top_results

        print(f"\n━━━ 筛选结果：{len(results)} 只股票符合趋势向上条件 ━━━")
        sort_label = "综合得分" if not self.no_fundamental else "趋势强度"
        print(f"   显示前 {len(top_results)} 只（按{sort_label}排序）\n")

        # 表头
        if self.no_fundamental:
            print(f"{'排名':<4} {'代码':<8} {'名称':<8} {'价格':>8} {'强度':>4} {'均线排列':<24} {'偏离MA20':>8} {'偏离MA60':>8} {'钟摆位置':<14} {'做T':>4}")
            print("-" * 110)
            for i, r in enumerate(top_results, 1):
                print(f"{i:<4} {r['code']:<8} {r['name']:<8} {r['price']:>8.2f} {r['strength']:>3}/10 {r['ma_desc']:<24} {r['dev_ma20']:>+7.1f}% {r['dev_ma60']:>+7.1f}% {r['pendulum']:<14} {r['t0_label']:>4}")
        else:
            print(f"{'排名':<4} {'代码':<8} {'名称':<8} {'价格':>8} {'技术':>4} {'基本面':>5} {'综合':>4} {'均线排列':<24} {'偏离MA20':>8} {'钟摆位置':<14} {'做T':>4}")
            print("-" * 120)
            for i, r in enumerate(top_results, 1):
                print(f"{i:<4} {r['code']:<8} {r['name']:<8} {r['price']:>8.2f} {r['strength']:>3}/10 {r['fund_score']:>3}/10 {r['combined_score']:>4.1f} {r['ma_desc']:<24} {r['dev_ma20']:>+7.1f}% {r['pendulum']:<14} {r['t0_label']:>4}")

        # 最佳做T候选
        t0_candidates = [r for r in top_results if r['pendulum_score'] >= 2 and r['strength'] >= 5]
        if t0_candidates:
            print(f"\n━━━ 最佳做T候选（趋势强 + 回踩均线附近）━━━")
            print(f"   这些股票趋势向上且钟摆回摆至MA20附近，适合「顺大势逆小势」做T\n")
            for r in t0_candidates[:10]:
                trend_def = ''
                if r['highs_rising'] and r['lows_rising']:
                    trend_def = '标准上升趋势(高低点递增)'
                elif r['highs_rising']:
                    trend_def = '高点递增'
                elif r['lows_rising']:
                    trend_def = '低点递增'
                print(f"   ⭐ {r['code']} {r['name']} ¥{r['price']:.2f} | 强度{r['strength']}/10 | {r['pendulum']} | {trend_def}")
        else:
            print(f"\n━━━ 做T候选 ━━━")
            print("   当前无理想做T候选（趋势向上但钟摆偏高，建议等待回踩）")

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
