#!/usr/bin/env python3
"""
股票基本面分析模块
基于「内功为本」投资哲学 — 分析公司盈利能力、成长性、财务健康、估值水平、资金面

评分体系（满分50分）：
- 盈利能力: 15分（ROE、净利率、毛利率）
- 成长能力: 10分（营收增长、利润增长趋势）
- 财务健康: 10分（资产负债率、流动比率、现金流）
- 估值水平: 10分（PE、机构参与度、排名）
- 资金面:   5分（主力资金流向、股东户数变化）

数据源说明：
- 基本面数据仍使用 akshare（baostock 不提供基本面数据）
- 如遇频繁限流，建议降低调用频率或等待1-2小时后重试

优化策略：
- 缓存机制减少重复查询
- 容错处理，部分数据缺失不影响整体分析
- 轻量级评分模式（仅查询核心指标）
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import hashlib
import os
import pickle

warnings.filterwarnings('ignore')

_FUND_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'fundamental')
os.makedirs(_FUND_CACHE_DIR, exist_ok=True)

# 全局内存缓存
_FUNDAMENTAL_CACHE = {}
_CACHE_TTL = 600

# 全市场估值数据缓存（单次运行内共享，避免重复获取 ak.stock_comment_em()）
_VALUATION_FULL_DF = None
_VALUATION_FULL_TS = 0
_VALUATION_FULL_TTL = 1800  # 全市场数据30分钟有效


def _get_cache_key(*args, **kwargs):
    key_str = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache(key):
    if key in _FUNDAMENTAL_CACHE:
        data, timestamp = _FUNDAMENTAL_CACHE[key]
        if time.time() - timestamp < _CACHE_TTL:
            return data
        else:
            del _FUNDAMENTAL_CACHE[key]
    return None


def _set_cache(key, data):
    _FUNDAMENTAL_CACHE[key] = (data, time.time())


def _disk_cache_path(category, key):
    """基本面磁盘缓存路径（当日有效）"""
    today = datetime.now().strftime('%Y%m%d')
    day_dir = os.path.join(_FUND_CACHE_DIR, today)
    os.makedirs(day_dir, exist_ok=True)
    safe_key = key.replace('/', '_').replace('.', '_')
    return os.path.join(day_dir, f'{category}_{safe_key}.pkl')


def _get_disk_cache(category, key):
    path = _disk_cache_path(category, key)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


def _set_disk_cache(category, key, data):
    path = _disk_cache_path(category, key)
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    except Exception:
        pass


def _get_valuation_full_df():
    """获取全市场估值数据（内存 + 磁盘双层缓存）"""
    global _VALUATION_FULL_DF, _VALUATION_FULL_TS

    # 内存缓存
    if _VALUATION_FULL_DF is not None and (time.time() - _VALUATION_FULL_TS) < _VALUATION_FULL_TTL:
        return _VALUATION_FULL_DF

    # 磁盘缓存（当日有效）
    disk_data = _get_disk_cache('valuation_full', 'all')
    if disk_data is not None:
        _VALUATION_FULL_DF = disk_data
        _VALUATION_FULL_TS = time.time()
        return _VALUATION_FULL_DF

    # 网络获取（全市场一次性获取）
    try:
        df = ak.stock_comment_em()
        if df is not None and not df.empty:
            _VALUATION_FULL_DF = df
            _VALUATION_FULL_TS = time.time()
            _set_disk_cache('valuation_full', 'all', df)
            return df
    except Exception:
        pass

    return None


def cleanup_fundamental_cache(keep_days=3):
    """清理过期的基本面磁盘缓存"""
    if not os.path.exists(_FUND_CACHE_DIR):
        return
    cutoff = datetime.now() - timedelta(days=keep_days)
    cutoff_str = cutoff.strftime('%Y%m%d')
    for d in os.listdir(_FUND_CACHE_DIR):
        full = os.path.join(_FUND_CACHE_DIR, d)
        if d < cutoff_str and os.path.isdir(full):
            import shutil
            shutil.rmtree(full, ignore_errors=True)


class FundamentalAnalyzer:
    """基本面分析器 — 内功评估（带缓存优化）"""

    def __init__(self, stock_code, stock_name=None):
        self.stock_code = stock_code
        self.stock_name = stock_name or stock_code
        self.financial_data = None      # 财务分析指标
        self.financial_abstract = None  # 财务摘要（多期）
        self.valuation_data = None      # 估值/机构评分
        self.fund_flow_data = None      # 资金流向
        self.shareholder_data = None    # 股东户数
        self.scores = {}                # 各维度得分
        self.details = {}               # 各维度详情
        self._fetch_errors = []         # 数据获取错误记录
        self._use_cache = True          # 是否使用缓存

    def fetch_all_data(self):
        """获取所有基本面数据"""
        self.fetch_financial_data()
        self.fetch_valuation_data()
        self.fetch_fund_flow()
        self.fetch_shareholder_data()

    def fetch_financial_data(self):
        """获取财务分析指标（最近3年，内存+磁盘双层缓存）"""
        cache_key = _get_cache_key('financial', self.stock_code)
        if self._use_cache:
            cached = _get_cache(cache_key)
            if cached is not None:
                self.financial_data = cached
                return

            disk_data = _get_disk_cache('financial', self.stock_code)
            if disk_data is not None:
                self.financial_data = disk_data
                _set_cache(cache_key, disk_data)
                return
        
        try:
            start_year = str(datetime.now().year - 3)
            df = ak.stock_financial_analysis_indicator(
                symbol=self.stock_code, start_year=start_year
            )
            if df is not None and not df.empty:
                self.financial_data = df
                if self._use_cache:
                    _set_cache(cache_key, df)
                    _set_disk_cache('financial', self.stock_code, df)
        except Exception as e:
            self._fetch_errors.append(f"财务指标: {e}")

    def fetch_valuation_data(self):
        """获取估值和机构评分（使用全市场级缓存，避免重复获取）"""
        try:
            full_df = _get_valuation_full_df()
            if full_df is not None and not full_df.empty:
                row = full_df[full_df['代码'] == self.stock_code]
                if not row.empty:
                    self.valuation_data = row.iloc[0]
        except Exception as e:
            self._fetch_errors.append(f"估值数据: {e}")

    def fetch_fund_flow(self):
        """获取近期主力资金流向（带磁盘缓存）"""
        disk_data = _get_disk_cache('fund_flow', self.stock_code)
        if disk_data is not None:
            self.fund_flow_data = disk_data
            return

        try:
            if self.stock_code.startswith('6'):
                market = 'sh'
            else:
                market = 'sz'
            df = ak.stock_individual_fund_flow(stock=self.stock_code, market=market)
            if df is not None and not df.empty:
                self.fund_flow_data = df.tail(20)
                _set_disk_cache('fund_flow', self.stock_code, self.fund_flow_data)
        except Exception as e:
            self._fetch_errors.append(f"资金流向: {e}")

    def fetch_shareholder_data(self):
        """获取股东户数变化趋势（带磁盘缓存）"""
        disk_data = _get_disk_cache('shareholder', self.stock_code)
        if disk_data is not None:
            self.shareholder_data = disk_data
            return

        try:
            df = ak.stock_zh_a_gdhs_detail_em(symbol=self.stock_code)
            if df is not None and not df.empty:
                self.shareholder_data = df.tail(10)
                _set_disk_cache('shareholder', self.stock_code, self.shareholder_data)
        except Exception as e:
            self._fetch_errors.append(f"股东户数: {e}")

    # =================================================================
    # 评分函数
    # =================================================================

    def _safe_float(self, val, default=None):
        """安全转换为float"""
        if val is None or (isinstance(val, str) and val.strip() in ('', '-', '--', 'False', 'None')):
            return default
        try:
            result = float(val)
            if np.isnan(result) or np.isinf(result):
                return default
            return result
        except (ValueError, TypeError):
            return default

    def _get_latest_financial(self, column, n=1):
        """从财务数据中获取最近n期的值列表"""
        if self.financial_data is None or self.financial_data.empty:
            return []
        values = []
        for i in range(min(n, len(self.financial_data))):
            row = self.financial_data.iloc[-(i + 1)]
            val = self._safe_float(row.get(column))
            if val is not None:
                values.append(val)
        return values

    def score_profitability(self):
        """盈利能力评分（满分15分）"""
        score = 0
        details = []

        if self.financial_data is None or self.financial_data.empty:
            self.scores['profitability'] = 0
            self.details['profitability'] = ['数据获取失败']
            return 0

        latest = self.financial_data.iloc[-1]

        # --- ROE 净资产收益率（0-6分）---
        roe = self._safe_float(latest.get('净资产收益率(%)'))
        if roe is not None:
            if roe >= 20:
                score += 6
                details.append(f'ROE {roe:.1f}% 优秀')
            elif roe >= 15:
                score += 5
                details.append(f'ROE {roe:.1f}% 良好')
            elif roe >= 10:
                score += 4
                details.append(f'ROE {roe:.1f}% 中等')
            elif roe >= 5:
                score += 2
                details.append(f'ROE {roe:.1f}% 偏低')
            elif roe >= 0:
                score += 1
                details.append(f'ROE {roe:.1f}% 较差')
            else:
                details.append(f'ROE {roe:.1f}% 亏损')
        else:
            details.append('ROE 数据缺失')

        # --- 销售净利率（0-5分）---
        net_margin = self._safe_float(latest.get('销售净利率(%)'))
        if net_margin is not None:
            if net_margin >= 30:
                score += 5
                details.append(f'净利率 {net_margin:.1f}% 极强')
            elif net_margin >= 20:
                score += 4
                details.append(f'净利率 {net_margin:.1f}% 优秀')
            elif net_margin >= 10:
                score += 3
                details.append(f'净利率 {net_margin:.1f}% 良好')
            elif net_margin >= 5:
                score += 2
                details.append(f'净利率 {net_margin:.1f}% 中等')
            elif net_margin >= 0:
                score += 1
                details.append(f'净利率 {net_margin:.1f}% 偏低')
            else:
                details.append(f'净利率 {net_margin:.1f}% 亏损')
        else:
            details.append('净利率 数据缺失')

        # --- 毛利率（0-4分）---
        gross_margin = self._safe_float(latest.get('销售毛利率(%)'))
        # 备选：用主营业务利润率，或从成本率反算
        if gross_margin is None:
            gross_margin = self._safe_float(latest.get('主营业务利润率(%)'))
        if gross_margin is None:
            cost_rate = self._safe_float(latest.get('主营业务成本率(%)'))
            if cost_rate is not None:
                gross_margin = 100 - cost_rate
        if gross_margin is not None:
            if gross_margin >= 60:
                score += 4
                details.append(f'毛利率 {gross_margin:.1f}%')
            elif gross_margin >= 40:
                score += 3
                details.append(f'毛利率 {gross_margin:.1f}%')
            elif gross_margin >= 25:
                score += 2
                details.append(f'毛利率 {gross_margin:.1f}%')
            elif gross_margin >= 15:
                score += 1
                details.append(f'毛利率 {gross_margin:.1f}%')
            else:
                details.append(f'毛利率 {gross_margin:.1f}% 偏低')
        else:
            details.append('毛利率 数据缺失')

        self.scores['profitability'] = min(15, score)
        self.details['profitability'] = details
        return self.scores['profitability']

    def score_growth(self):
        """成长能力评分（满分10分）"""
        score = 0
        details = []

        if self.financial_data is None or self.financial_data.empty:
            self.scores['growth'] = 0
            self.details['growth'] = ['数据获取失败']
            return 0

        latest = self.financial_data.iloc[-1]

        # --- 营收增长率（0-4分）---
        rev_growth = self._safe_float(latest.get('主营业务收入增长率(%)'))
        if rev_growth is not None:
            if rev_growth >= 30:
                score += 4
                details.append(f'营收增长 +{rev_growth:.1f}% 高速')
            elif rev_growth >= 15:
                score += 3
                details.append(f'营收增长 +{rev_growth:.1f}% 快速')
            elif rev_growth >= 5:
                score += 2
                details.append(f'营收增长 +{rev_growth:.1f}% 稳健')
            elif rev_growth >= 0:
                score += 1
                details.append(f'营收增长 +{rev_growth:.1f}% 平稳')
            else:
                details.append(f'营收增长 {rev_growth:+.1f}% 下滑')
        else:
            details.append('营收增长 数据缺失')

        # --- 净利润增长率（0-4分）---
        profit_growth = self._safe_float(latest.get('净利润增长率(%)'))
        if profit_growth is not None:
            if profit_growth >= 30:
                score += 4
                details.append(f'利润增长 +{profit_growth:.1f}% 高速')
            elif profit_growth >= 15:
                score += 3
                details.append(f'利润增长 +{profit_growth:.1f}% 快速')
            elif profit_growth >= 5:
                score += 2
                details.append(f'利润增长 +{profit_growth:.1f}% 稳健')
            elif profit_growth >= 0:
                score += 1
                details.append(f'利润增长 +{profit_growth:.1f}% 平稳')
            else:
                details.append(f'利润增长 {profit_growth:+.1f}% 下滑')
        else:
            details.append('利润增长 数据缺失')

        # --- 增长持续性（0-2分）---
        # 检查最近几期是否持续正增长
        rev_growths = self._get_latest_financial('主营业务收入增长率(%)', 4)
        profit_growths = self._get_latest_financial('净利润增长率(%)', 4)

        positive_rev = sum(1 for g in rev_growths if g and g > 0)
        positive_profit = sum(1 for g in profit_growths if g and g > 0)

        if positive_rev >= 3 and positive_profit >= 3:
            score += 2
            details.append(f'连续{min(positive_rev, positive_profit)}期正增长')
        elif positive_rev >= 2 or positive_profit >= 2:
            score += 1
            details.append('增长有波动')
        else:
            details.append('增长不稳定')

        self.scores['growth'] = min(10, score)
        self.details['growth'] = details
        return self.scores['growth']

    def score_financial_health(self):
        """财务健康度评分（满分10分）"""
        score = 0
        details = []

        if self.financial_data is None or self.financial_data.empty:
            self.scores['health'] = 0
            self.details['health'] = ['数据获取失败']
            return 0

        latest = self.financial_data.iloc[-1]

        # --- 资产负债率（0-4分）---
        debt_ratio = self._safe_float(latest.get('资产负债率(%)'))
        if debt_ratio is not None:
            if debt_ratio <= 30:
                score += 4
                details.append(f'资产负债率 {debt_ratio:.1f}% 极低')
            elif debt_ratio <= 50:
                score += 3
                details.append(f'资产负债率 {debt_ratio:.1f}% 健康')
            elif debt_ratio <= 65:
                score += 2
                details.append(f'资产负债率 {debt_ratio:.1f}% 中等')
            elif debt_ratio <= 80:
                score += 1
                details.append(f'资产负债率 {debt_ratio:.1f}% 偏高')
            else:
                details.append(f'资产负债率 {debt_ratio:.1f}% 风险')
        else:
            details.append('资产负债率 数据缺失')

        # --- 流动比率（0-3分）---
        current_ratio = self._safe_float(latest.get('流动比率'))
        if current_ratio is not None:
            if current_ratio >= 2.0:
                score += 3
                details.append(f'流动比率 {current_ratio:.1f} 充裕')
            elif current_ratio >= 1.5:
                score += 2
                details.append(f'流动比率 {current_ratio:.1f} 良好')
            elif current_ratio >= 1.0:
                score += 1
                details.append(f'流动比率 {current_ratio:.1f} 一般')
            else:
                details.append(f'流动比率 {current_ratio:.1f} 紧张')
        else:
            details.append('流动比率 数据缺失')

        # --- 经营现金流（0-3分）---
        # 现金流/净利润比率 > 1 表明利润含金量高
        cashflow_ratio = self._safe_float(latest.get('经营现金净流量与净利润的比率(%)'))
        if cashflow_ratio is not None:
            if cashflow_ratio >= 100:
                score += 3
                details.append('现金流充沛')
            elif cashflow_ratio >= 70:
                score += 2
                details.append('现金流良好')
            elif cashflow_ratio >= 30:
                score += 1
                details.append('现金流一般')
            else:
                details.append('现金流不足')

            # 财务一致性检查：现金流大幅低于利润可能存在应收账款风险
            if cashflow_ratio < 30:
                details.append('⚠️ 利润含金量低（现金流远低于利润）')
        else:
            details.append('现金流 数据缺失')

        self.scores['health'] = min(10, score)
        self.details['health'] = details
        return self.scores['health']

    def score_valuation(self):
        """估值水平评分（满分10分）"""
        score = 0
        details = []

        # --- PE 市盈率（0-5分）---
        pe = None
        if self.valuation_data is not None:
            pe = self._safe_float(self.valuation_data.get('市盈率'))

        if pe is not None and pe > 0:
            if pe <= 15:
                score += 5
                details.append(f'PE {pe:.1f} 低估')
            elif pe <= 25:
                score += 4
                details.append(f'PE {pe:.1f} 合理')
            elif pe <= 40:
                score += 3
                details.append(f'PE {pe:.1f} 偏高')
            elif pe <= 60:
                score += 2
                details.append(f'PE {pe:.1f} 较高')
            elif pe <= 100:
                score += 1
                details.append(f'PE {pe:.1f} 高估')
            else:
                details.append(f'PE {pe:.1f} 极高')
        elif pe is not None and pe < 0:
            details.append(f'PE {pe:.1f} 亏损')
        else:
            details.append('PE 数据缺失')

        # --- 机构参与度（0-3分）---
        inst_ratio = None
        if self.valuation_data is not None:
            inst_ratio = self._safe_float(self.valuation_data.get('机构参与度'))

        if inst_ratio is not None:
            inst_pct = inst_ratio * 100 if inst_ratio <= 1 else inst_ratio
            if inst_pct >= 60:
                score += 3
                details.append(f'机构参与度 {inst_pct:.1f}% 高')
            elif inst_pct >= 40:
                score += 2
                details.append(f'机构参与度 {inst_pct:.1f}% 中等')
            elif inst_pct >= 20:
                score += 1
                details.append(f'机构参与度 {inst_pct:.1f}% 偏低')
            else:
                details.append(f'机构参与度 {inst_pct:.1f}% 低')
        else:
            details.append('机构参与度 数据缺失')

        # --- 综合排名（0-2分）---
        ranking = None
        if self.valuation_data is not None:
            ranking = self._safe_float(self.valuation_data.get('目前排名'))

        if ranking is not None:
            if ranking <= 500:
                score += 2
                details.append(f'排名 {int(ranking)}/5000+')
            elif ranking <= 1500:
                score += 1
                details.append(f'排名 {int(ranking)}/5000+')
            else:
                details.append(f'排名 {int(ranking)}/5000+')
        else:
            details.append('排名 数据缺失')

        self.scores['valuation'] = min(10, score)
        self.details['valuation'] = details
        return self.scores['valuation']

    def score_capital_flow(self):
        """资金面评分（满分5分）"""
        score = 0
        details = []

        # --- 主力资金流向（0-3分）---
        if self.fund_flow_data is not None and not self.fund_flow_data.empty:
            recent_5 = self.fund_flow_data.tail(5)
            total_net = recent_5['主力净流入-净额'].sum()
            total_net_billion = total_net / 1e8

            if total_net_billion > 5:
                score += 3
                details.append(f'近5日主力净流入 +{total_net_billion:.1f}亿 强')
            elif total_net_billion > 1:
                score += 2
                details.append(f'近5日主力净流入 +{total_net_billion:.1f}亿')
            elif total_net_billion > 0:
                score += 1
                details.append(f'近5日主力净流入 +{total_net_billion:.1f}亿 小幅')
            elif total_net_billion > -1:
                details.append(f'近5日主力净流出 {total_net_billion:.1f}亿 小幅')
            else:
                details.append(f'近5日主力净流出 {total_net_billion:.1f}亿')
        else:
            details.append('资金流向 数据缺失')

        # --- 股东户数变化（0-2分）---
        if self.shareholder_data is not None and not self.shareholder_data.empty:
            latest_sh = self.shareholder_data.iloc[-1]
            change_ratio = self._safe_float(latest_sh.get('股东户数-增减比例'))

            if change_ratio is not None:
                if change_ratio < -5:
                    score += 2
                    details.append(f'股东户数 {change_ratio:+.1f}% 集中')
                elif change_ratio < -1:
                    score += 1
                    details.append(f'股东户数 {change_ratio:+.1f}% 小幅集中')
                elif change_ratio > 5:
                    details.append(f'股东户数 {change_ratio:+.1f}% 分散')
                elif change_ratio > 1:
                    details.append(f'股东户数 {change_ratio:+.1f}% 小幅分散')
                else:
                    details.append(f'股东户数 {change_ratio:+.1f}% 稳定')
            else:
                details.append('股东户数变化 数据缺失')
        else:
            details.append('股东户数 数据缺失')

        self.scores['capital'] = min(5, score)
        self.details['capital'] = details
        return self.scores['capital']

    def get_fundamental_score(self):
        """计算基本面总分（满分50分）"""
        self.score_profitability()
        self.score_growth()
        self.score_financial_health()
        self.score_valuation()
        self.score_capital_flow()

        total = sum(self.scores.values())
        return {
            'total': total,
            'max_score': 50,
            'scores': dict(self.scores),
            'details': dict(self.details),
        }

    def get_report_text(self):
        """生成基本面报告文本"""
        result = self.get_fundamental_score()
        lines = []

        lines.append("\n━━━ 基本面分析（内功）━━━\n")

        # 盈利能力
        prof_score = self.scores.get('profitability', 0)
        prof_details = ' | '.join(self.details.get('profitability', []))
        lines.append(f"盈利能力: {prof_score:>2}/15 ({prof_details})")

        # 成长能力
        grow_score = self.scores.get('growth', 0)
        grow_details = ' | '.join(self.details.get('growth', []))
        lines.append(f"成长能力: {grow_score:>2}/10 ({grow_details})")

        # 财务健康
        health_score = self.scores.get('health', 0)
        health_details = ' | '.join(self.details.get('health', []))
        lines.append(f"财务健康: {health_score:>2}/10 ({health_details})")

        # 估值水平
        val_score = self.scores.get('valuation', 0)
        val_details = ' | '.join(self.details.get('valuation', []))
        lines.append(f"估值水平: {val_score:>2}/10 ({val_details})")

        # 资金面
        cap_score = self.scores.get('capital', 0)
        cap_details = ' | '.join(self.details.get('capital', []))
        lines.append(f"资金面:   {cap_score:>2}/5  ({cap_details})")

        lines.append(f"\n基本面总分: {result['total']}/{result['max_score']}")

        # 基本面等级
        total = result['total']
        if total >= 40:
            lines.append("基本面等级: ⭐⭐⭐⭐⭐ 极优")
        elif total >= 35:
            lines.append("基本面等级: ⭐⭐⭐⭐ 优秀")
        elif total >= 28:
            lines.append("基本面等级: ⭐⭐⭐ 良好")
        elif total >= 20:
            lines.append("基本面等级: ⭐⭐ 中等")
        elif total >= 10:
            lines.append("基本面等级: ⭐ 偏弱")
        else:
            lines.append("基本面等级: 较差")

        # 治理提醒
        lines.append("")
        lines.append("━━━ 治理与文化提醒 ━━━")
        lines.append("以上数据可量化公司经营结果，但以下定性因素同样重要：")
        lines.append("  1. 管理层诚信度 — 是否有信披违规、财务造假历史")
        lines.append("  2. 公司治理结构 — 股东大会/董事会运作是否规范")
        lines.append("  3. 企业文化 — 员工满意度、创新氛围、社会责任")
        lines.append("  4. 实控人背景 — 行业经验、经营口碑、减持记录")
        lines.append("  建议查阅年报中的《公司治理》章节和公开舆情信息")

        return '\n'.join(lines)

    def get_light_score(self):
        """轻量评分（仅核心指标：ROE、增长率、PE）用于选股"""
        score = 0
        max_score = 10

        # ROE（0-4分）
        if self.financial_data is not None and not self.financial_data.empty:
            latest = self.financial_data.iloc[-1]
            roe = self._safe_float(latest.get('净资产收益率(%)'))
            if roe is not None:
                if roe >= 20:
                    score += 4
                elif roe >= 15:
                    score += 3
                elif roe >= 10:
                    score += 2
                elif roe >= 5:
                    score += 1

            # 营收增长（0-3分）
            rev_growth = self._safe_float(latest.get('主营业务收入增长率(%)'))
            if rev_growth is not None:
                if rev_growth >= 20:
                    score += 3
                elif rev_growth >= 10:
                    score += 2
                elif rev_growth >= 0:
                    score += 1

        # PE（0-3分）
        if self.valuation_data is not None:
            pe = self._safe_float(self.valuation_data.get('市盈率'))
            if pe is not None and pe > 0:
                if pe <= 20:
                    score += 3
                elif pe <= 35:
                    score += 2
                elif pe <= 50:
                    score += 1

        return {'score': min(max_score, score), 'max_score': max_score}

    def get_value_score(self):
        """
        价值评估评分（侧重"是否被低估"）用于底部反弹选股

        评分维度（满分10分）：
        - ROE 质量（0-3分）：高ROE = 好公司
        - PE 低估（0-3分）：低PE = 便宜
        - 营收增长（0-2分）：正增长 = 非衰退
        - ROE+PE 联合加分（0-2分）：高ROE+低PE = 典型被低估

        返回:
            dict: {
                'score': int,         # 0-10
                'max_score': 10,
                'roe': float|None,    # ROE 值
                'pe': float|None,     # PE 值
                'rev_growth': float|None,  # 营收增长率
                'details': list[str], # 评分明细
                'is_value_trap': bool,# 是否疑似价值陷阱
            }
        """
        score = 0
        max_score = 10
        roe = None
        pe = None
        rev_growth = None
        details = []
        is_value_trap = False

        # ROE（0-3分）
        if self.financial_data is not None and not self.financial_data.empty:
            latest = self.financial_data.iloc[-1]
            roe = self._safe_float(latest.get('净资产收益率(%)'))
            if roe is not None:
                if roe >= 20:
                    score += 3
                    details.append(f'ROE {roe:.1f}% 优秀')
                elif roe >= 15:
                    score += 3
                    details.append(f'ROE {roe:.1f}% 良好')
                elif roe >= 10:
                    score += 2
                    details.append(f'ROE {roe:.1f}% 中等')
                elif roe >= 5:
                    score += 1
                    details.append(f'ROE {roe:.1f}% 偏低')
                else:
                    details.append(f'ROE {roe:.1f}% 较差')
                    if roe < 5:
                        is_value_trap = True
            else:
                details.append('ROE 数据缺失')

            # 营收增长（0-2分）
            rev_growth = self._safe_float(latest.get('主营业务收入增长率(%)'))
            if rev_growth is not None:
                if rev_growth >= 15:
                    score += 2
                    details.append(f'营收增长 +{rev_growth:.1f}%')
                elif rev_growth >= 0:
                    score += 1
                    details.append(f'营收增长 +{rev_growth:.1f}%')
                else:
                    details.append(f'营收下滑 {rev_growth:.1f}%')
                    if rev_growth < -10:
                        is_value_trap = True
            else:
                details.append('营收增长 数据缺失')

        # PE 低估（0-3分）
        if self.valuation_data is not None:
            pe = self._safe_float(self.valuation_data.get('市盈率'))
            if pe is not None and pe > 0:
                if pe <= 10:
                    score += 3
                    details.append(f'PE {pe:.1f} 极度低估')
                elif pe <= 15:
                    score += 3
                    details.append(f'PE {pe:.1f} 低估')
                elif pe <= 20:
                    score += 2
                    details.append(f'PE {pe:.1f} 偏低')
                elif pe <= 30:
                    score += 1
                    details.append(f'PE {pe:.1f} 合理')
                else:
                    details.append(f'PE {pe:.1f} 偏高')
            elif pe is not None and pe < 0:
                details.append(f'PE {pe:.1f} 亏损')
                is_value_trap = True

        # ROE+PE 联合加分（0-2分）— 高 ROE + 低 PE = 典型被低估
        if roe is not None and pe is not None and pe > 0:
            if roe >= 15 and pe <= 20:
                score += 2
                details.append('高ROE+低PE 典型被低估')
            elif roe >= 10 and pe <= 15:
                score += 2
                details.append('良好ROE+极低PE 被低估')
            elif roe >= 10 and pe <= 25:
                score += 1
                details.append('合理ROE+合理PE')

        return {
            'score': min(max_score, score),
            'max_score': max_score,
            'roe': roe,
            'pe': pe,
            'rev_growth': rev_growth,
            'details': details,
            'is_value_trap': is_value_trap,
        }


def main():
    """独立运行测试"""
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python3 fundamental_analyzer.py <股票代码>")
        print("示例: python3 fundamental_analyzer.py 600519")
        sys.exit(1)

    stock_code = sys.argv[1]
    print(f"📊 正在获取 {stock_code} 的基本面数据...")

    analyzer = FundamentalAnalyzer(stock_code)
    analyzer.fetch_all_data()

    if analyzer._fetch_errors:
        print(f"⚠️ 部分数据获取失败: {', '.join(analyzer._fetch_errors)}")

    print(analyzer.get_report_text())


if __name__ == "__main__":
    main()
