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
import pickle

warnings.filterwarnings('ignore')

# 磁盘缓存目录（当日有效）
_DISK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
os.makedirs(_DISK_CACHE_DIR, exist_ok=True)

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
    
    # ============================================================
    # 磁盘缓存：日K线当日只需获取一次，大幅减少网络请求
    # ============================================================
    
    @classmethod
    def _disk_cache_path(cls, category, key):
        """生成磁盘缓存文件路径（按日期分目录）"""
        today = datetime.now().strftime('%Y%m%d')
        day_dir = os.path.join(_DISK_CACHE_DIR, today)
        os.makedirs(day_dir, exist_ok=True)
        safe_key = key.replace('/', '_').replace('.', '_')
        return os.path.join(day_dir, f'{category}_{safe_key}.pkl')
    
    @classmethod
    def _get_disk_cache(cls, category, key):
        """读取磁盘缓存（当日有效）"""
        path = cls._disk_cache_path(category, key)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None
    
    @classmethod
    def _set_disk_cache(cls, category, key, data):
        """写入磁盘缓存"""
        path = cls._disk_cache_path(category, key)
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass
    
    @classmethod
    def cleanup_old_disk_cache(cls, keep_days=3):
        """清理过期的磁盘缓存（保留最近N天）"""
        if not os.path.exists(_DISK_CACHE_DIR):
            return
        cutoff = datetime.now() - timedelta(days=keep_days)
        cutoff_str = cutoff.strftime('%Y%m%d')
        for d in os.listdir(_DISK_CACHE_DIR):
            if d < cutoff_str and os.path.isdir(os.path.join(_DISK_CACHE_DIR, d)):
                import shutil
                shutil.rmtree(os.path.join(_DISK_CACHE_DIR, d), ignore_errors=True)
    
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
    
    @classmethod
    def _is_trading_hours(cls):
        """判断当前是否在交易时段（周一到周五 9:15-15:05）"""
        now = datetime.now()
        if now.weekday() >= 5:  # 周末
            return False
        t = now.hour * 100 + now.minute
        return 915 <= t <= 1505

    @classmethod
    def _append_today_realtime(cls, df, stock_code):
        """
        用 adata 实时行情补充当日数据行。
        如果 baostock 返回的最新日期不是今天，且当前在交易时段，
        则从 adata 分时数据中合成当日 OHLCV 并追加到 df 末尾。
        """
        if df is None or df.empty:
            return df

        today_str = datetime.now().strftime('%Y-%m-%d')
        last_date = str(df.iloc[-1]['日期'])

        # 如果已经包含今天数据，无需补充
        if last_date >= today_str:
            return df

        # 非交易时段也不补充（盘前盘后没有有效数据）
        if not cls._is_trading_hours():
            return df

        ad = _get_adata()
        if ad is None:
            return df

        try:
            # 优先用分时数据合成完整 OHLCV
            min_df = ad.stock.market.get_market_min(stock_code=stock_code)
            if min_df is not None and not min_df.empty:
                # 过滤掉集合竞价阶段成交量为0的数据
                trade_df = min_df[min_df['volume'] > 0]
                if trade_df.empty:
                    # 开盘前只有竞价数据，用最新价格作为所有OHLC
                    latest_price = float(min_df.iloc[-1]['price'])
                    today_row = pd.DataFrame([{
                        '日期': today_str,
                        '开盘': latest_price,
                        '最高': latest_price,
                        '最低': latest_price,
                        '收盘': latest_price,
                        '成交量': 0,
                        '成交额': 0.0,
                        '换手率': 0.0,
                        '涨跌幅': 0.0,
                    }])
                else:
                    open_price = float(trade_df.iloc[0]['price'])
                    close_price = float(trade_df.iloc[-1]['price'])
                    high_price = float(trade_df['price'].max())
                    low_price = float(trade_df['price'].min())
                    total_volume = int(trade_df['volume'].sum())
                    total_amount = float(trade_df['amount'].sum())

                    # 计算涨跌幅（基于前一日收盘价）
                    prev_close = float(df.iloc[-1]['收盘'])
                    change_pct = (close_price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0

                    today_row = pd.DataFrame([{
                        '日期': today_str,
                        '开盘': open_price,
                        '最高': high_price,
                        '最低': low_price,
                        '收盘': close_price,
                        '成交量': total_volume,
                        '成交额': total_amount,
                        '换手率': 0.0,
                        '涨跌幅': round(change_pct, 2),
                    }])

                df = pd.concat([df, today_row], ignore_index=True)
                return df
        except Exception:
            pass

        # 降级：用实时行情（仅有 price/volume/amount，OHLC 用 price 近似）
        try:
            rt_df = ad.stock.market.list_market_current(code_list=[stock_code])
            if rt_df is not None and not rt_df.empty:
                price = float(rt_df.iloc[0]['price'])
                volume = int(rt_df.iloc[0]['volume']) if rt_df.iloc[0]['volume'] else 0
                amount = float(rt_df.iloc[0]['amount']) if rt_df.iloc[0]['amount'] else 0.0
                change_pct = float(rt_df.iloc[0]['change_pct']) if rt_df.iloc[0]['change_pct'] else 0.0

                today_row = pd.DataFrame([{
                    '日期': today_str,
                    '开盘': price,
                    '最高': price,
                    '最低': price,
                    '收盘': price,
                    '成交量': volume,
                    '成交额': amount,
                    '换手率': 0.0,
                    '涨跌幅': change_pct,
                }])
                df = pd.concat([df, today_row], ignore_index=True)
        except Exception:
            pass

        return df

    @classmethod
    def get_stock_hist(cls, stock_code, start_date=None, end_date=None, adjust='qfq', period='daily'):
        """
        获取股票历史K线数据（带缓存和多数据源切换）
        交易时段自动用 adata 实时行情补充当日数据。
        
        参数:
            stock_code: 6位股票代码，如 '600519'
            start_date: 开始日期，格式 'YYYYMMDD' 或 datetime
            end_date: 结束日期，格式 'YYYYMMDD' 或 datetime
            adjust: 复权类型，'qfq'=前复权, 'hfq'=后复权, ''=不复权
            period: 周期，'daily'=日线, 'weekly'=周线, 'monthly'=月线
        
        返回:
            DataFrame，列名与 akshare 兼容：日期、开盘、最高、最低、收盘、成交量、成交额、换手率
        """
        # 1) 内存缓存（实时补充的数据也会缓存，TTL 5分钟自动刷新）
        cache_key = cls._get_cache_key('hist', stock_code, start_date, end_date, adjust, period)
        cached = cls._get_cache(cache_key)
        if cached is not None:
            return cached.copy()
        
        # 2) 磁盘缓存（日K线当日有效，大幅减少重复请求）
        #    注意：磁盘缓存不含实时补充的当日数据，需要在读取后补充
        disk_key = f'{stock_code}_{start_date}_{end_date}_{adjust}_{period}'
        disk_cached = cls._get_disk_cache('hist', disk_key)
        if disk_cached is not None:
            # 补充当日实时数据
            if period == 'daily':
                disk_cached = cls._append_today_realtime(disk_cached, stock_code)
            cls._set_cache(cache_key, disk_cached)  # 回填内存缓存（含当日数据）
            return disk_cached.copy()
        
        # 3) 网络请求 — baostock（主数据源）
        try:
            df = cls._get_stock_hist_baostock(stock_code, start_date, end_date, adjust, period)
            if df is not None and not df.empty:
                cls._set_disk_cache('hist', disk_key, df)  # 磁盘缓存仅存baostock原始数据
                # 补充当日实时数据
                if period == 'daily':
                    df = cls._append_today_realtime(df, stock_code)
                cls._set_cache(cache_key, df)
                return df
        except Exception as e:
            print(f"   ⚠ baostock 获取失败，尝试备用数据源...")
        
        # 4) 降级到 akshare（备用数据源）
        if cls._akshare_available is not False:
            try:
                import akshare as ak
                df = cls._get_stock_hist_akshare(ak, stock_code, start_date, end_date, adjust, period)
                if df is not None and not df.empty:
                    cls._akshare_available = True
                    cls._set_disk_cache('hist', disk_key, df)
                    if period == 'daily':
                        df = cls._append_today_realtime(df, stock_code)
                    cls._set_cache(cache_key, df)
                    return df
            except Exception as e:
                cls._akshare_available = False
                print(f"   ⚠ akshare 备用数据源也失败")
        
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
        
        # 磁盘缓存（指数成分股当日不变）
        disk_cached = cls._get_disk_cache('index', index_code)
        if disk_cached is not None:
            cls._set_cache(cache_key, disk_cached)
            return disk_cached.copy()
        
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
        cls._set_disk_cache('index', index_code, result)
        return result
    
    # 批量实时行情缓存（供选股等批量场景使用）
    _realtime_cache = {}   # code -> {price, volume, amount, change_pct}
    _realtime_cache_ts = 0  # 缓存时间戳

    @classmethod
    def preload_realtime_prices(cls, stock_codes):
        """
        批量预加载实时行情（选股等批量场景使用）。
        调用一次即可缓存所有股票的当日价格，后续 _append_today_realtime
        会优先使用此缓存，避免逐只调用 adata 分时接口。
        
        参数:
            stock_codes: 股票代码列表
        """
        if not cls._is_trading_hours():
            return

        ad = _get_adata()
        if ad is None:
            return

        try:
            # adata 批量查询非常快（50只 < 0.1秒）
            batch_size = 100
            for i in range(0, len(stock_codes), batch_size):
                batch = stock_codes[i:i + batch_size]
                rt_df = ad.stock.market.list_market_current(code_list=batch)
                if rt_df is not None and not rt_df.empty:
                    for _, row in rt_df.iterrows():
                        code = str(row['stock_code'])
                        cls._realtime_cache[code] = {
                            'price': float(row['price']) if row['price'] else 0,
                            'volume': int(row['volume']) if row['volume'] else 0,
                            'amount': float(row['amount']) if row['amount'] else 0,
                            'change_pct': float(row['change_pct']) if row['change_pct'] else 0,
                        }
            cls._realtime_cache_ts = time.time()
            print(f"   📡 已预加载 {len(cls._realtime_cache)} 只股票的实时行情")
        except Exception as e:
            print(f"   ⚠ 预加载实时行情失败: {e}")

    @classmethod
    def _append_today_realtime(cls, df, stock_code):
        """
        用 adata 实时行情补充当日数据行。
        如果 baostock 返回的最新日期不是今天，且当前在交易时段，
        则从 adata 分时数据或批量缓存中合成当日 OHLCV 并追加到 df 末尾。
        """
        if df is None or df.empty:
            return df

        today_str = datetime.now().strftime('%Y-%m-%d')
        last_date = str(df.iloc[-1]['日期'])

        # 如果已经包含今天数据，无需补充
        if last_date >= today_str:
            return df

        # 非交易时段也不补充
        if not cls._is_trading_hours():
            return df

        # 方式1：如果有批量预加载的实时缓存（选股场景），直接用
        if stock_code in cls._realtime_cache and (time.time() - cls._realtime_cache_ts) < 600:
            rt = cls._realtime_cache[stock_code]
            price = rt['price']
            if price <= 0:
                return df
            prev_close = float(df.iloc[-1]['收盘'])
            change_pct = rt['change_pct'] if rt['change_pct'] else (
                (price - prev_close) / prev_close * 100 if prev_close > 0 else 0
            )
            today_row = pd.DataFrame([{
                '日期': today_str,
                '开盘': price,  # 批量接口无OHLC，用当前价近似
                '最高': price,
                '最低': price,
                '收盘': price,
                '成交量': rt['volume'],
                '成交额': rt['amount'],
                '换手率': 0.0,
                '涨跌幅': round(change_pct, 2),
            }])
            df = pd.concat([df, today_row], ignore_index=True)
            return df

        # 方式2：单只股票分析场景，用分时数据合成完整 OHLCV
        ad = _get_adata()
        if ad is None:
            return df

        try:
            min_df = ad.stock.market.get_market_min(stock_code=stock_code)
            if min_df is not None and not min_df.empty:
                trade_df = min_df[min_df['volume'] > 0]
                if trade_df.empty:
                    latest_price = float(min_df.iloc[-1]['price'])
                    today_row = pd.DataFrame([{
                        '日期': today_str,
                        '开盘': latest_price,
                        '最高': latest_price,
                        '最低': latest_price,
                        '收盘': latest_price,
                        '成交量': 0,
                        '成交额': 0.0,
                        '换手率': 0.0,
                        '涨跌幅': 0.0,
                    }])
                else:
                    open_price = float(trade_df.iloc[0]['price'])
                    close_price = float(trade_df.iloc[-1]['price'])
                    high_price = float(trade_df['price'].max())
                    low_price = float(trade_df['price'].min())
                    total_volume = int(trade_df['volume'].sum())
                    total_amount = float(trade_df['amount'].sum())

                    prev_close = float(df.iloc[-1]['收盘'])
                    change_pct = (close_price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0

                    today_row = pd.DataFrame([{
                        '日期': today_str,
                        '开盘': open_price,
                        '最高': high_price,
                        '最低': low_price,
                        '收盘': close_price,
                        '成交量': total_volume,
                        '成交额': total_amount,
                        '换手率': 0.0,
                        '涨跌幅': round(change_pct, 2),
                    }])

                df = pd.concat([df, today_row], ignore_index=True)
                return df
        except Exception:
            pass

        # 降级：用实时行情
        try:
            rt_df = ad.stock.market.list_market_current(code_list=[stock_code])
            if rt_df is not None and not rt_df.empty:
                price = float(rt_df.iloc[0]['price'])
                volume = int(rt_df.iloc[0]['volume']) if rt_df.iloc[0]['volume'] else 0
                amount = float(rt_df.iloc[0]['amount']) if rt_df.iloc[0]['amount'] else 0.0
                change_pct = float(rt_df.iloc[0]['change_pct']) if rt_df.iloc[0]['change_pct'] else 0.0

                today_row = pd.DataFrame([{
                    '日期': today_str,
                    '开盘': price,
                    '最高': price,
                    '最低': price,
                    '收盘': price,
                    '成交量': volume,
                    '成交额': amount,
                    '换手率': 0.0,
                    '涨跌幅': change_pct,
                }])
                df = pd.concat([df, today_row], ignore_index=True)
        except Exception:
            pass

        return df

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
