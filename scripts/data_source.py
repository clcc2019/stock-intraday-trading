#!/usr/bin/env python3
"""
数据源适配层 — 多数据源自动切换，提高可用性
避免单一数据源限流导致功能不可用

数据源优先级：
1. baostock（主）：稳定、免费、不限流，来自证券交易所（历史K线）
2. akshare（备）：历史K线备用
3. adata（补充）：实时行情、资金流向、分时行情、5档盘口

优化策略：
- 自动重试和降级切换
- 缓存机制减少重复查询（5分钟TTL）
- 批量查询优化
"""

import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import hashlib
import os

warnings.filterwarnings('ignore')

# 磁盘缓存目录
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
os.makedirs(_CACHE_DIR, exist_ok=True)

# 延迟导入 adata（可选依赖）
_adata = None
_adata_available = None  # None=未检测, True=可用, False=不可用


def _get_adata():
    """延迟导入 adata"""
    global _adata, _adata_available
    if _adata_available is False:
        return None
    if _adata is None:
        try:
            import adata
            _adata = adata
            _adata_available = True
        except ImportError:
            _adata_available = False
            return None
    return _adata


class DataSource:
    """统一数据源接口 — 多数据源自动切换"""
    
    _logged_in = False
    _cache = {}  # 简单内存缓存
    _cache_ttl = 300  # 缓存5分钟
    _cache_write_count = 0  # 写入计数，用于触发清理
    _akshare_available = None  # None=未检测, True=可用, False=不可用
    
    @classmethod
    def login(cls):
        """登录 baostock"""
        if not cls._logged_in:
            lg = bs.login()
            if lg.error_code == '0':
                cls._logged_in = True
            else:
                raise Exception(f"baostock 登录失败: {lg.error_msg}")
    
    @classmethod
    def logout(cls):
        """登出 baostock"""
        if cls._logged_in:
            bs.logout()
            cls._logged_in = False
    
    @classmethod
    def _get_cache_key(cls, *args, **kwargs):
        """生成缓存键"""
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @classmethod
    def _get_cache(cls, key):
        """获取缓存"""
        if key in cls._cache:
            data, timestamp = cls._cache[key]
            if time.time() - timestamp < cls._cache_ttl:
                return data
            else:
                del cls._cache[key]
        return None
    
    @classmethod
    def _set_cache(cls, key, data):
        """设置缓存，每100次写入清理一次过期条目"""
        cls._cache[key] = (data, time.time())
        cls._cache_write_count += 1
        if cls._cache_write_count >= 100:
            cls._cleanup_cache()
            cls._cache_write_count = 0

    @classmethod
    def _cleanup_cache(cls):
        """清理所有过期缓存条目"""
        now = time.time()
        expired_keys = [
            k for k, (_, ts) in cls._cache.items()
            if now - ts >= cls._cache_ttl
        ]
        for k in expired_keys:
            del cls._cache[k]

    @classmethod
    def clear_disk_cache(cls, stock_code=None):
        """清理磁盘缓存
        
        参数:
            stock_code: 指定股票代码，为 None 清理全部
        """
        if stock_code:
            import glob
            pattern = os.path.join(_CACHE_DIR, f'{stock_code}_*.csv')
            for f in glob.glob(pattern):
                os.remove(f)
        else:
            import shutil
            if os.path.exists(_CACHE_DIR):
                shutil.rmtree(_CACHE_DIR)
                os.makedirs(_CACHE_DIR, exist_ok=True)
    
    @classmethod
    def _convert_code(cls, stock_code):
        """转换股票代码为 baostock 格式"""
        if stock_code.startswith('6'):
            return f'sh.{stock_code}'
        elif stock_code.startswith(('0', '3')):
            return f'sz.{stock_code}'
        else:
            return f'sh.{stock_code}'
    
    @classmethod
    def get_stock_hist_minute(cls, stock_code, start_date=None, end_date=None, adjust='qfq', period='5'):
        """
        获取股票分钟K线数据（带缓存）
        
        参数:
            stock_code: 6位股票代码，如 '600519'
            start_date: 开始日期，格式 'YYYYMMDD' 或 datetime
            end_date: 结束日期，格式 'YYYYMMDD' 或 datetime
            adjust: 复权类型，'qfq'=前复权, 'hfq'=后复权, ''=不复权
            period: 周期，'5'=5分钟, '15'=15分钟, '30'=30分钟, '60'=60分钟
        
        返回:
            DataFrame，列名与 akshare 兼容：时间、开盘、最高、最低、收盘、成交量
        """
        # 检查缓存
        cache_key = cls._get_cache_key('minute', stock_code, start_date, end_date, adjust, period)
        cached = cls._get_cache(cache_key)
        if cached is not None:
            return cached.copy()
        
        cls.login()
        
        # 处理日期格式
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        elif start_date and len(start_date) == 8:
            start_date = f'{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}'
        
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        elif end_date and len(end_date) == 8:
            end_date = f'{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}'
        
        # 默认日期（今天）
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = end_date
        
        # 转换代码
        bs_code = cls._convert_code(stock_code)
        
        # 复权类型映射
        adjust_map = {'qfq': '2', 'hfq': '1', '': '3'}
        adjustflag = adjust_map.get(adjust, '2')
        
        # 查询数据
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,time,code,open,high,low,close,volume',
            start_date=start_date,
            end_date=end_date,
            frequency=period,
            adjustflag=adjustflag
        )
        
        if rs.error_code != '0':
            raise Exception(f"baostock 查询失败: {rs.error_msg}")
        
        # 转换为 DataFrame
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 时间格式转换（baostock 返回如 '20260206093500000'）
        df['时间'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M%S%f').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 列名映射（兼容 akshare）
        df = df.rename(columns={
            'open': '开盘',
            'high': '最高',
            'low': '最低',
            'close': '收盘',
            'volume': '成交量',
        })
        
        # 数据类型转换
        for col in ['开盘', '最高', '最低', '收盘']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce').fillna(0).astype(np.int64)
        
        result = df[['时间', '开盘', '最高', '最低', '收盘', '成交量']]
        cls._set_cache(cache_key, result)
        return result
    
    # ============================================================
    # 磁盘缓存：智能增量更新历史K线
    # ============================================================

    @classmethod
    def _disk_cache_path(cls, stock_code, adjust, period):
        """生成磁盘缓存文件路径"""
        return os.path.join(_CACHE_DIR, f'{stock_code}_{period}_{adjust}.csv')

    _disk_update_times = {}  # {cache_path: last_update_timestamp}
    _DISK_UPDATE_COOLDOWN = 900  # 15分钟增量冷却期

    @classmethod
    def _load_disk_cache(cls, stock_code, adjust, period):
        """从磁盘读取缓存"""
        path = cls._disk_cache_path(stock_code, adjust, period)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, dtype={'日期': str})
            if df.empty or '日期' not in df.columns:
                return None
            # 类型转换
            for col in ['开盘', '最高', '最低', '收盘', '换手率', '涨跌幅']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            if '成交量' in df.columns:
                df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce').fillna(0).astype(np.int64)
            if '成交额' in df.columns:
                df['成交额'] = pd.to_numeric(df['成交额'], errors='coerce').fillna(0).astype(np.float64)
            return df
        except Exception:
            return None

    @classmethod
    def _should_try_increment(cls, stock_code, adjust, period):
        """检查是否应该尝试增量更新（冷却期内跳过联网）"""
        path = cls._disk_cache_path(stock_code, adjust, period)
        last_update = cls._disk_update_times.get(path)
        if last_update is None:
            # 首次：用文件修改时间判断
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                if time.time() - mtime < cls._DISK_UPDATE_COOLDOWN:
                    return False
        else:
            if time.time() - last_update < cls._DISK_UPDATE_COOLDOWN:
                return False
        return True

    @classmethod
    def _mark_disk_updated(cls, stock_code, adjust, period):
        """标记磁盘缓存已更新"""
        path = cls._disk_cache_path(stock_code, adjust, period)
        cls._disk_update_times[path] = time.time()

    @classmethod
    def _save_disk_cache(cls, df, stock_code, adjust, period):
        """保存到磁盘缓存"""
        if df is None or df.empty:
            return
        path = cls._disk_cache_path(stock_code, adjust, period)
        try:
            df.to_csv(path, index=False)
        except Exception:
            pass

    @classmethod
    def get_stock_hist(cls, stock_code, start_date=None, end_date=None, adjust='qfq', period='daily'):
        """
        获取股票历史K线数据（磁盘缓存 + 智能增量更新）
        
        性能优化：
        1. 首先检查内存缓存（5分钟TTL）
        2. 再检查磁盘缓存，如有则只查询缺失的增量数据
        3. 无缓存才全量查询
        
        参数:
            stock_code: 6位股票代码，如 '600519'
            start_date: 开始日期，格式 'YYYYMMDD' 或 datetime
            end_date: 结束日期，格式 'YYYYMMDD' 或 datetime
            adjust: 复权类型，'qfq'=前复权, 'hfq'=后复权, ''=不复权
            period: 周期，'daily'=日线, 'weekly'=周线, 'monthly'=月线
        
        返回:
            DataFrame，列名与 akshare 兼容：日期、开盘、最高、最低、收盘、成交量、成交额、换手率
        """
        # 1. 检查内存缓存
        cache_key = cls._get_cache_key('hist', stock_code, start_date, end_date, adjust, period)
        cached = cls._get_cache(cache_key)
        if cached is not None:
            return cached.copy()
        
        # 标准化日期
        if isinstance(end_date, datetime):
            end_str = end_date.strftime('%Y-%m-%d')
        elif end_date and len(str(end_date)) == 8:
            end_str = f'{str(end_date)[:4]}-{str(end_date)[4:6]}-{str(end_date)[6:]}'
        elif end_date:
            end_str = str(end_date)
        else:
            end_str = datetime.now().strftime('%Y-%m-%d')

        if isinstance(start_date, datetime):
            start_str = start_date.strftime('%Y-%m-%d')
        elif start_date and len(str(start_date)) == 8:
            start_str = f'{str(start_date)[:4]}-{str(start_date)[4:6]}-{str(start_date)[6:]}'
        elif start_date:
            start_str = str(start_date)
        else:
            start_str = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

        # 2. 磁盘缓存 + 智能增量更新（仅日线/周线）
        if period in ('daily', 'weekly'):
            disk_df = cls._load_disk_cache(stock_code, adjust, period)
            if disk_df is not None and len(disk_df) > 0:
                last_cached_date = str(disk_df['日期'].iloc[-1])
                # 如果缓存已经覆盖到今天或昨天，直接用缓存
                today_str = datetime.now().strftime('%Y-%m-%d')
                yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                if last_cached_date >= yesterday_str:
                    # 缓存足够新，检查冷却期再决定是否增量
                    if cls._should_try_increment(stock_code, adjust, period):
                        incr_start = (datetime.strptime(last_cached_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                        if incr_start <= today_str:
                            try:
                                incr_df = cls._fetch_from_source(stock_code, incr_start, end_str, adjust, period)
                                if incr_df is not None and not incr_df.empty:
                                    disk_df = pd.concat([disk_df, incr_df], ignore_index=True)
                                    disk_df = disk_df.drop_duplicates(subset=['日期'], keep='last').sort_values('日期').reset_index(drop=True)
                                    cls._save_disk_cache(disk_df, stock_code, adjust, period)
                            except Exception:
                                pass  # 增量失败用现有缓存
                        cls._mark_disk_updated(stock_code, adjust, period)
                    # 按请求日期范围过滤
                    result = disk_df[(disk_df['日期'] >= start_str) & (disk_df['日期'] <= end_str)].reset_index(drop=True)
                    if not result.empty:
                        cls._set_cache(cache_key, result)
                        return result.copy()
                elif last_cached_date >= start_str:
                    # 缓存部分覆盖，追加后面缺失的部分
                    incr_start = (datetime.strptime(last_cached_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                    try:
                        incr_df = cls._fetch_from_source(stock_code, incr_start, end_str, adjust, period)
                        if incr_df is not None and not incr_df.empty:
                            disk_df = pd.concat([disk_df, incr_df], ignore_index=True)
                            disk_df = disk_df.drop_duplicates(subset=['日期'], keep='last').sort_values('日期').reset_index(drop=True)
                            cls._save_disk_cache(disk_df, stock_code, adjust, period)
                            cls._mark_disk_updated(stock_code, adjust, period)
                            result = disk_df[(disk_df['日期'] >= start_str) & (disk_df['日期'] <= end_str)].reset_index(drop=True)
                            if not result.empty:
                                cls._set_cache(cache_key, result)
                                return result.copy()
                    except Exception:
                        pass

        # 3. 全量获取（无缓存或缓存失效）
        df = cls._fetch_from_source(stock_code, start_str, end_str, adjust, period)
        if df is not None and not df.empty:
            cls._set_cache(cache_key, df)
            if period in ('daily', 'weekly'):
                # 合并保存（保留更多历史数据）
                disk_df = cls._load_disk_cache(stock_code, adjust, period)
                if disk_df is not None and not disk_df.empty:
                    merged = pd.concat([disk_df, df], ignore_index=True)
                    merged = merged.drop_duplicates(subset=['日期'], keep='last').sort_values('日期').reset_index(drop=True)
                    cls._save_disk_cache(merged, stock_code, adjust, period)
                else:
                    cls._save_disk_cache(df, stock_code, adjust, period)
            return df
        
        return pd.DataFrame()

    @classmethod
    def _fetch_from_source(cls, stock_code, start_date, end_date, adjust, period):
        """从数据源获取数据（baostock 主 → akshare 备）"""
        # 尝试 baostock（主数据源）
        try:
            df = cls._get_stock_hist_baostock(stock_code, start_date, end_date, adjust, period)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        
        # 降级到 akshare（备用数据源）
        if cls._akshare_available is not False:
            try:
                import akshare as ak
                df = cls._get_stock_hist_akshare(ak, stock_code, start_date, end_date, adjust, period)
                if df is not None and not df.empty:
                    cls._akshare_available = True
                    return df
            except Exception:
                cls._akshare_available = False
        
        return pd.DataFrame()
    
    @classmethod
    def _get_stock_hist_baostock(cls, stock_code, start_date, end_date, adjust, period):
        """从 baostock 获取历史数据"""
        cls.login()
        
        # 处理日期格式
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        elif start_date and len(start_date) == 8:
            start_date = f'{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}'
        
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        elif end_date and len(end_date) == 8:
            end_date = f'{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}'
        
        # 默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
        
        # 转换代码
        bs_code = cls._convert_code(stock_code)
        
        # 复权类型映射
        adjust_map = {'qfq': '2', 'hfq': '1', '': '3'}
        adjustflag = adjust_map.get(adjust, '2')
        
        # 周期映射
        freq_map = {'daily': 'd', 'weekly': 'w', 'monthly': 'm'}
        frequency = freq_map.get(period, 'd')
        
        # 查询数据
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,code,open,high,low,close,volume,amount,turn,pctChg',
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjustflag
        )
        
        if rs.error_code != '0':
            raise Exception(f"baostock 查询失败: {rs.error_msg}")
        
        # 转换为 DataFrame
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 列名映射（兼容 akshare）
        df = df.rename(columns={
            'date': '日期',
            'open': '开盘',
            'high': '最高',
            'low': '最低',
            'close': '收盘',
            'volume': '成交量',
            'amount': '成交额',
            'turn': '换手率',
            'pctChg': '涨跌幅',
        })
        
        # 数据类型转换
        for col in ['开盘', '最高', '最低', '收盘', '换手率', '涨跌幅']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 成交量和成交额（baostock 返回的是字符串，需要转换）
        df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce').fillna(0).astype(np.int64)
        df['成交额'] = pd.to_numeric(df['成交额'], errors='coerce').fillna(0).astype(np.float64)
        
        return df
    
    @classmethod
    def _get_stock_hist_akshare(cls, ak, stock_code, start_date, end_date, adjust, period):
        """从 akshare 获取历史数据（备用）"""
        # 处理日期格式
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y%m%d')
        elif start_date and '-' in str(start_date):
            start_date = str(start_date).replace('-', '')
        
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y%m%d')
        elif end_date and '-' in str(end_date):
            end_date = str(end_date).replace('-', '')
        
        # 默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=400)).strftime('%Y%m%d')
        
        # 调用 akshare
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        
        return df
    
    @classmethod
    def get_stock_list(cls):
        """
        获取全部A股列表（带缓存）
        
        返回:
            DataFrame，包含 code（股票代码）、code_name（股票名称）
        """
        cache_key = cls._get_cache_key('stock_list')
        cached = cls._get_cache(cache_key)
        if cached is not None:
            return cached.copy()
        
        cls.login()
        
        rs = bs.query_all_stock(day=datetime.now().strftime('%Y-%m-%d'))
        
        if rs.error_code != '0':
            raise Exception(f"获取股票列表失败: {rs.error_msg}")
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 过滤A股（sh/sz开头）
        df = df[df['code'].str.startswith(('sh.', 'sz.'))]
        
        # 提取6位代码
        df['stock_code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '')
        
        # 过滤ST股、退市股、北交所
        df = df[~df['code_name'].str.contains('ST|退市|\\*', na=False, regex=True)]
        df = df[~df['stock_code'].str.startswith(('8', '9', '4'))]
        
        result = df[['stock_code', 'code_name']].rename(columns={
            'stock_code': '代码',
            'code_name': '名称'
        })
        
        cls._set_cache(cache_key, result)
        return result
    
    @classmethod
    def get_index_stocks(cls, index_code):
        """
        获取指数成分股（带缓存）
        
        参数:
            index_code: 指数代码，如 'sh.000300'（沪深300）
            支持: sh.000300(沪深300), sh.000905(中证500), sh.000016(上证50)
        
        返回:
            DataFrame，包含 代码（股票代码）、名称（股票名称）
        """
        cache_key = cls._get_cache_key('index_stocks', index_code)
        cached = cls._get_cache(cache_key)
        if cached is not None:
            return cached.copy()
        
        cls.login()
        
        # 根据指数代码选择正确的 baostock API
        api_map = {
            'sh.000300': bs.query_hs300_stocks,
            'sh.000905': bs.query_zz500_stocks,
            'sh.000016': bs.query_sz50_stocks,
        }
        query_fn = api_map.get(index_code)
        if query_fn is None:
            raise Exception(f"不支持的指数: {index_code}，支持: sh.000300(沪深300), sh.000905(中证500), sh.000016(上证50)")
        
        date_str = datetime.now().strftime('%Y-%m-%d')
        rs = query_fn(date=date_str)
        
        if rs.error_code != '0':
            # 如果失败，尝试前一个交易日
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            rs = query_fn(date=yesterday)
        
        if rs.error_code != '0':
            raise Exception(f"获取指数成分股失败: {rs.error_msg}")
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 提取6位代码
        df['stock_code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '')
        
        result = df[['stock_code', 'code_name']].rename(columns={
            'stock_code': '代码',
            'code_name': '名称'
        })
        
        cls._set_cache(cache_key, result)
        return result
    
    @classmethod
    def batch_get_stock_hist(cls, stock_codes, start_date=None, end_date=None, adjust='qfq', period='daily'):
        """
        批量获取股票历史数据（优化版，减少查询次数）
        
        参数:
            stock_codes: 股票代码列表
            其他参数同 get_stock_hist
        
        返回:
            dict: {stock_code: DataFrame}
        """
        results = {}
        for code in stock_codes:
            try:
                df = cls.get_stock_hist(code, start_date, end_date, adjust, period)
                if df is not None and not df.empty:
                    results[code] = df
            except Exception:
                continue
        return results

    # ============================================================
    # adata 补充数据源：实时行情 / 资金流向 / 分时 / 5档盘口
    # ============================================================

    @classmethod
    def get_realtime_quote(cls, stock_codes):
        """
        获取实时行情（来源：adata，新浪/腾讯）
        
        参数:
            stock_codes: 股票代码列表，如 ['600519', '002594']
        
        返回:
            DataFrame: stock_code, short_name, price, change, change_pct, volume, amount
            失败返回 None
        """
        cache_key = cls._get_cache_key('realtime', tuple(sorted(stock_codes)))
        cached = cls._get_cache(cache_key)
        if cached is not None:
            return cached.copy()

        ad = _get_adata()
        if ad is None:
            return None
        try:
            df = ad.stock.market.list_market_current(code_list=stock_codes)
            if df is not None and not df.empty:
                # 短缓存（30秒）
                cls._cache[cache_key] = (df, time.time())
                return df
        except Exception:
            pass
        return None

    @classmethod
    def get_capital_flow(cls, stock_code, days=30):
        """
        获取资金流向（来源：adata，东方财富）
        
        参数:
            stock_code: 6位股票代码
            days: 返回最近 N 天的数据
        
        返回:
            DataFrame: stock_code, trade_date, main_net_inflow, sm_net_inflow,
                       mid_net_inflow, lg_net_inflow, max_net_inflow
            失败返回 None
        """
        cache_key = cls._get_cache_key('capital_flow', stock_code)
        cached = cls._get_cache(cache_key)
        if cached is not None:
            return cached.tail(days).copy()

        ad = _get_adata()
        if ad is None:
            return None
        try:
            df = ad.stock.market.get_capital_flow(stock_code=stock_code)
            if df is not None and not df.empty:
                cls._set_cache(cache_key, df)
                return df.tail(days).copy()
        except Exception:
            pass
        return None

    @classmethod
    def get_intraday_minute(cls, stock_code):
        """
        获取今日分时行情（来源：adata）
        
        参数:
            stock_code: 6位股票代码
        
        返回:
            DataFrame: stock_code, trade_time, price, change, change_pct,
                       volume, avg_price, amount
            失败返回 None
        """
        cache_key = cls._get_cache_key('intraday_min', stock_code, datetime.now().strftime('%Y%m%d'))
        cached = cls._get_cache(cache_key)
        if cached is not None:
            return cached.copy()

        ad = _get_adata()
        if ad is None:
            return None
        try:
            df = ad.stock.market.get_market_min(stock_code=stock_code)
            if df is not None and not df.empty:
                # 短缓存（60秒）
                cls._cache[cache_key] = (df, time.time())
                return df
        except Exception:
            pass
        return None

    @classmethod
    def get_market_five(cls, stock_code):
        """
        获取5档盘口行情（来源：adata）
        
        参数:
            stock_code: 6位股票代码
        
        返回:
            DataFrame: 包含 s1-s5(卖价), sv1-sv5(卖量), b1-b5(买价), bv1-bv5(买量)
            失败返回 None
        """
        ad = _get_adata()
        if ad is None:
            return None
        try:
            df = ad.stock.market.get_market_five(stock_code=stock_code)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        return None

    @classmethod
    def get_stock_concepts(cls, stock_code):
        """
        获取股票所属概念板块（来源：adata，东方财富）
        
        参数:
            stock_code: 6位股票代码
        
        返回:
            DataFrame: stock_code, concept_code, name, source, reason
            失败返回 None
        """
        cache_key = cls._get_cache_key('concepts', stock_code)
        cached = cls._get_cache(cache_key)
        if cached is not None:
            return cached.copy()

        ad = _get_adata()
        if ad is None:
            return None
        try:
            df = ad.stock.info.get_concept_east(stock_code=stock_code)
            if df is not None and not df.empty:
                cls._set_cache(cache_key, df)
                return df
        except Exception:
            pass
        return None

    @classmethod
    def get_index_realtime(cls):
        """
        获取指数实时行情（来源：adata）
        
        返回:
            DataFrame: index_code, trade_time, open, high, low, price, volume, amount, change, change_pct
            失败返回 None
        """
        cache_key = cls._get_cache_key('index_realtime')
        cached = cls._get_cache(cache_key)
        if cached is not None:
            return cached.copy()

        ad = _get_adata()
        if ad is None:
            return None
        try:
            df = ad.stock.market.get_market_index_current()
            if df is not None and not df.empty:
                cls._cache[cache_key] = (df, time.time())
                return df
        except Exception:
            pass
        return None


def main():
    """测试数据源"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python3 data_source.py <股票代码>")
        print("示例: python3 data_source.py 600519")
        sys.exit(1)
    
    stock_code = sys.argv[1]
    
    print(f"📊 测试获取 {stock_code} 的数据...")
    
    try:
        df = DataSource.get_stock_hist(stock_code)
        print(f"✅ 获取到 {len(df)} 条数据")
        print("\n最近3天数据:")
        print(df.tail(3))
        
        print(f"\n列名: {df.columns.tolist()}")
        
        # 测试缓存
        print("\n测试缓存...")
        import time
        start = time.time()
        df2 = DataSource.get_stock_hist(stock_code)
        print(f"✅ 缓存命中，耗时: {(time.time() - start)*1000:.0f}ms")
        
    except Exception as e:
        print(f"❌ 失败: {e}")
    finally:
        DataSource.logout()


if __name__ == "__main__":
    main()
