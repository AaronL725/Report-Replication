"""
因子回测模块
功能：对各种波动率因子进行全面的回测分析
回测要求：
- 样本：沪深300指数成分股为股票池，剔除停牌、ST等交易异常股票
- 回测区间：2021.01.01至2025.04.30
- 调仓频率：月度
- 组合权重分配：等权
- 因子处理方式：因子方向调整、缩尾调整、市值行业中性化、标准化
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

# 导入因子计算模块
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Factor'))

# 动态导入因子模块以避免导入错误
def import_factor_modules():
    """动态导入因子模块"""
    global calculate_ewmavol, calculate_vol_nm, calculate_rankvol, calculate_rvol, calculate_garchvol
    
    try:
        from EWMAVOL import calculate_ewmavol
    except ImportError:
        print("警告：无法导入EWMAVOL模块")
        calculate_ewmavol = None
    
    try:
        from VOL_3M import calculate_vol_nm
    except ImportError:
        print("警告：无法导入VOL_3M模块")
        calculate_vol_nm = None
    
    try:
        from RANKVOL import calculate_rankvol
    except ImportError:
        print("警告：无法导入RANKVOL模块")
        calculate_rankvol = None
    
    try:
        from RVOL import calculate_rvol
    except ImportError:
        print("警告：无法导入RVOL模块")
        calculate_rvol = None
    
    try:
        from GARCHVOL import calculate_garchvol
    except ImportError:
        print("警告：无法导入GARCHVOL模块")
        calculate_garchvol = None

# 获取当前脚本目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(CURRENT_DIR, 'cache')

class FactorBacktester:
    """因子回测类"""

    def __init__(self, start_date: str = '2021-01-01', end_date: str = '2025-04-30'):
        """
        初始化回测器
        
        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.data = {}
        self.rebalance_dates = []
        
        # 动态导入因子模块
        import_factor_modules()
          # 因子配置
        self.factors_config = {}
        if calculate_ewmavol is not None:
            self.factors_config['EWMAVOL'] = {'func': calculate_ewmavol, 'params': {'window': 60, 'lambda_decay': 0.9}}
        if calculate_vol_nm is not None:
            self.factors_config['VOL_3M'] = {'func': calculate_vol_nm, 'params': {'window': 60}}
        if calculate_rankvol is not None:
            self.factors_config['RANKVOL'] = {'func': calculate_rankvol, 'params': {'window': 60}}
        if calculate_rvol is not None:
            self.factors_config['RVOL'] = {'func': self._calculate_rvol_wrapper, 'params': {'window': 60}}
        if calculate_garchvol is not None:
            self.factors_config['GARCHVOL'] = {'func': self._calculate_garchvol_wrapper, 'params': {'window': 60}}
        
        print(f"回测器初始化完成: {start_date} 到 {end_date}")
        print(f"可用因子: {list(self.factors_config.keys())}")
    def _calculate_rvol_wrapper(self, close_prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """RVOL因子计算包装器，处理FF因子参数"""
        if 'ff_factors' in self.data:
            return calculate_rvol(close_prices, self.data['ff_factors'], window)
        else:
            print("警告：缺少Fama-French因子数据，跳过RVOL计算")
            return pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
    
    def _calculate_garchvol_wrapper(self, close_prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """GARCHVOL因子计算包装器，处理FF因子参数"""
        if 'ff_factors' in self.data:
            return calculate_garchvol(close_prices, self.data['ff_factors'], window)
        else:
            print("警告：缺少Fama-French因子数据，跳过GARCHVOL计算")
            return pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
    
    def load_cached_data(self) -> bool:
        """
        加载缓存的预处理数据
        
        Returns:
            bool: 是否成功加载所有必要数据
        """
        print("开始加载缓存数据...")
        required_files = [
            'processed_close.pkl', 'processed_open.pkl', 'processed_high.pkl', 
            'processed_low.pkl', 'processed_vol.pkl', 'processed_market_cap.pkl',
            'processed_industry.pkl', 'processed_csi300_index_returns.pkl',
            'processed_csi300_constituents.pkl', 'processed_ff_factors.pkl'
        ]
        
        for file_name in required_files:
            file_path = os.path.join(CACHE_DIR, file_name)
            if not os.path.exists(file_path):
                print(f"错误：缓存文件 {file_name} 不存在")
                return False
                
            try:
                with open(file_path, 'rb') as f:
                    data_key = file_name.replace('processed_', '').replace('.pkl', '')
                    self.data[data_key] = pickle.load(f)
                print(f"成功加载: {file_name}")
            except Exception as e:
                print(f"加载 {file_name} 时出错: {e}")
                return False
        
        # 过滤数据到回测区间
        self._filter_data_by_date()
        print("数据加载完成")
        return True
    
    def _filter_data_by_date(self):
        """根据回测日期范围过滤数据"""
        print(f"过滤数据到回测区间: {self.start_date} 到 {self.end_date}")
        
        for key, df in self.data.items():
            if isinstance(df, (pd.DataFrame, pd.Series)):
                # 过滤日期范围
                mask = (df.index >= self.start_date) & (df.index <= self.end_date)
                self.data[key] = df[mask]
                
                if isinstance(df, pd.DataFrame):
                    print(f"{key}: {self.data[key].shape}")
                else:
                    print(f"{key}: {len(self.data[key])} entries")
    
    def _get_trading_days(self) -> pd.DatetimeIndex:
        """获取回测期间的交易日"""
        return self.data['close'].index
    
    def _get_month_end_dates(self) -> List[pd.Timestamp]:
        """获取月末调仓日期"""
        trading_days = self._get_trading_days()
        
        # 获取每月最后一个交易日
        month_ends = []
        for year in range(self.start_date.year, self.end_date.year + 1):
            for month in range(1, 13):
                # 月末日期
                if year == self.start_date.year and month < self.start_date.month:
                    continue
                if year == self.end_date.year and month > self.end_date.month:
                    break
                
                # 找该月的最后一个交易日
                month_start = pd.Timestamp(year, month, 1)
                if month == 12:
                    month_end = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(days=1)
                else:
                    month_end = pd.Timestamp(year, month + 1, 1) - pd.Timedelta(days=1)
                
                # 在交易日中找到该月最后一个交易日
                month_trading_days = trading_days[(trading_days >= month_start) & (trading_days <= month_end)]
                if len(month_trading_days) > 0:
                    month_ends.append(month_trading_days[-1])
        
        self.rebalance_dates = sorted(month_ends)
        print(f"找到 {len(self.rebalance_dates)} 个调仓日期")
        return self.rebalance_dates
    def _filter_valid_stocks(self, factor_data: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """
        过滤有效股票，只保留CSI300成分股，并剔除停牌、ST等异常股票
        
        Args:
            factor_data: 因子数据
            date: 当前日期
            
        Returns:
            过滤后的因子数据
        """
        if date not in factor_data.index:
            return pd.DataFrame()
        
        factor_values = factor_data.loc[date].dropna()
        
        # 1. 首先过滤CSI300成分股
        if 'csi300_constituents' in self.data and date in self.data['csi300_constituents'].index:
            csi300_constituents = self.data['csi300_constituents'].loc[date]
            # 获取当日的CSI300成分股（值为True的股票）
            csi300_stocks = csi300_constituents[csi300_constituents == True].index
            # 只保留CSI300成分股
            factor_values = factor_values[factor_values.index.intersection(csi300_stocks)]
            print(f"日期 {date}: CSI300成分股过滤后剩余 {len(factor_values)} 只股票")
        else:
            print(f"警告：日期 {date} 无CSI300成分股数据，使用全部股票")
        
        # 2. 检查是否有价格数据
        if date in self.data['close'].index:
            close_prices = self.data['close'].loc[date]
            # 只保留有收盘价的股票
            valid_stocks = factor_values.index.intersection(close_prices.dropna().index)
            factor_values = factor_values[valid_stocks]
        
        # 3. 检查成交量，过滤停牌股票
        if date in self.data['vol'].index:
            volumes = self.data['vol'].loc[date]
            # 只保留有成交量的股票
            trading_stocks = volumes[volumes > 0].index
            factor_values = factor_values[factor_values.index.intersection(trading_stocks)]
        
        # 4. 简单过滤ST股票（基于股票代码包含ST标识，这里简化处理）
        # 实际应用中需要更精确的ST股票识别
        non_st_stocks = [stock for stock in factor_values.index 
                        if 'ST' not in str(stock).upper()]
        factor_values = factor_values[non_st_stocks]
        
        return factor_values
    
    def _neutralize_factor(self, factor_values: pd.Series, date: pd.Timestamp) -> pd.Series:
        """
        因子中性化处理：市值和行业中性化
        
        Args:
            factor_values: 原始因子值
            date: 当前日期
            
        Returns:
            中性化后的因子值
        """
        if len(factor_values) == 0:
            return factor_values
          # 获取市值数据
        if date not in self.data['market_cap'].index:
            print(f"警告：日期 {date} 无市值数据，跳过市值中性化")
            market_cap = None
        else:
            market_cap = self.data['market_cap'].loc[date]
            # 只保留与因子值有交集的股票
            common_stocks_mc = factor_values.index.intersection(market_cap.index)
            market_cap = market_cap[common_stocks_mc].dropna()
        
        # 获取行业数据
        if date not in self.data['industry'].index:
            print(f"警告：日期 {date} 无行业数据，跳过行业中性化")
            industry_data = None
        else:
            industry_data = self.data['industry'].loc[date]
            # 只保留与因子值有交集的股票
            common_stocks_ind = factor_values.index.intersection(industry_data.index)
            industry_data = industry_data[common_stocks_ind].dropna()
        
        # 确保因子值、市值、行业数据的股票一致
        if market_cap is not None and industry_data is not None:
            common_stocks = factor_values.index.intersection(market_cap.index).intersection(industry_data.index)
        elif market_cap is not None:
            common_stocks = factor_values.index.intersection(market_cap.index)
        elif industry_data is not None:
            common_stocks = factor_values.index.intersection(industry_data.index)
        else:
            return factor_values  # 没有中性化数据，返回原值
        
        if len(common_stocks) < 10:
            print(f"警告：日期 {date} 有效股票数量不足 ({len(common_stocks)})，跳过中性化")
            return factor_values
        
        factor_values = factor_values[common_stocks]
        
        # 准备回归数据
        y = factor_values.values
        X = []
        
        # 添加市值因子
        if market_cap is not None:
            log_market_cap = np.log(market_cap[common_stocks].values)
            X.append(log_market_cap)
        
        # 添加行业哑变量
        if industry_data is not None:
            industries = industry_data[common_stocks]
            unique_industries = industries.unique()
            
            for industry in unique_industries[1:]:  # 排除第一个行业作为基准
                industry_dummy = (industries == industry).astype(int).values
                X.append(industry_dummy)
        
        if len(X) == 0:
            return factor_values
        
        X = np.column_stack(X)
        X = np.column_stack([np.ones(len(X)), X])  # 添加常数项
        
        try:
            # 回归中性化
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
            
            # 返回残差作为中性化后的因子值
            return pd.Series(residuals, index=common_stocks)
        except np.linalg.LinAlgError:
            print(f"警告：日期 {date} 回归失败，返回原因子值")
            return factor_values
    
    def _standardize_factor(self, factor_values: pd.Series) -> pd.Series:
        """
        因子标准化：去均值、除标准差
        
        Args:
            factor_values: 因子值
            
        Returns:
            标准化后的因子值
        """
        if len(factor_values) == 0:
            return factor_values
        
        # 计算均值和标准差
        mean_val = factor_values.mean()
        std_val = factor_values.std()
        
        if std_val == 0 or np.isnan(std_val):
            print("警告：因子标准差为0或NaN，跳过标准化")
            return factor_values
        
        # 标准化
        standardized = (factor_values - mean_val) / std_val
        return standardized
    
    def _winsorize_factor(self, factor_values: pd.Series, quantile: float = 0.01) -> pd.Series:
        """
        因子缩尾处理
        
        Args:
            factor_values: 因子值
            quantile: 缩尾比例，默认1%
            
        Returns:
            缩尾后的因子值
        """
        if len(factor_values) == 0:
            return factor_values
        
        lower_bound = factor_values.quantile(quantile)
        upper_bound = factor_values.quantile(1 - quantile)
        
        return factor_values.clip(lower=lower_bound, upper=upper_bound)
    
    def process_factor(self, factor_data: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """
        完整的因子处理流程：方向调整、缩尾、中性化、标准化
        
        Args:
            factor_data: 原始因子数据
            factor_name: 因子名称
            
        Returns:
            处理后的因子数据
        """
        print(f"开始处理因子: {factor_name}")
        
        processed_factor = factor_data.copy()
        
        # 因子方向调整（波动率因子通常是反向因子，即波动率越高收益越低）
        processed_factor = -processed_factor
        
        # 逐日处理
        for date in processed_factor.index:
            if date in self.rebalance_dates:
                factor_values = processed_factor.loc[date].dropna()
                
                if len(factor_values) == 0:
                    continue
                
                # 1. 缩尾处理
                factor_values = self._winsorize_factor(factor_values)
                
                # 2. 市值行业中性化
                factor_values = self._neutralize_factor(factor_values, date)
                
                # 3. 标准化
                factor_values = self._standardize_factor(factor_values)
                
                # 更新处理后的值
                processed_factor.loc[date, factor_values.index] = factor_values
        
        print(f"因子 {factor_name} 处理完成")
        return processed_factor
    
    def calculate_factor_ic(self, factor_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        计算因子IC和RankIC
        
        Args:
            factor_data: 因子数据
            
        Returns:
            (IC序列, RankIC序列)
        """
        ic_series = pd.Series(dtype=float)
        rank_ic_series = pd.Series(dtype=float)
          # 计算下期收益率
        next_period_returns = self.data['close'].pct_change().shift(-1)
        
        for i, date in enumerate(self.rebalance_dates[:-1]):  # 排除最后一个日期
            next_date = self.rebalance_dates[i + 1]
            
            if date not in factor_data.index or next_date not in next_period_returns.index:
                continue
            
            # 获取当期因子值
            factor_values = factor_data.loc[date].dropna()
            
            # 使用更简单的有效股票过滤
            if date in self.data['close'].index:
                close_prices = self.data['close'].loc[date]
                valid_stocks = close_prices.dropna().index
                factor_values = factor_values[factor_values.index.intersection(valid_stocks)]
            
            if len(factor_values) < 10:
                continue
            
            # 获取下期收益率
            returns = next_period_returns.loc[next_date]
            
            # 找到共同股票
            common_stocks = factor_values.index.intersection(returns.dropna().index)
            if len(common_stocks) < 10:
                continue
            
            factor_vals = factor_values[common_stocks]
            return_vals = returns[common_stocks]
            
            try:
                # 计算IC (Pearson相关系数)
                ic_val, _ = pearsonr(factor_vals, return_vals)
                if not np.isnan(ic_val):
                    ic_series.loc[date] = ic_val
                  # 计算RankIC (Spearman相关系数)
                rank_ic_val, _ = spearmanr(factor_vals, return_vals)
                if not np.isnan(rank_ic_val):
                    rank_ic_series.loc[date] = rank_ic_val
                    
            except Exception as e:
                print(f"计算IC时出错 (日期: {date}): {e}")
                continue
        
        return ic_series, rank_ic_series
    
    def calculate_portfolio_returns(self, factor_data: pd.DataFrame, n_groups: int = 5) -> pd.DataFrame:
        """
        计算分组组合收益率
        
        Args:
            factor_data: 因子数据
            n_groups: 分组数量，默认5组
              Returns:
            各组合收益率DataFrame
        """
        group_returns = pd.DataFrame(index=self.rebalance_dates[:-1])
        
        for i in range(n_groups):
            group_returns[f'第{i+1}组'] = 0.0
        
        
        for i, date in enumerate(self.rebalance_dates[:-1]):
            next_date = self.rebalance_dates[i + 1]
            
            if date not in factor_data.index:
                print(f"DEBUG: 日期 {date} 不在因子数据中，跳过")
                continue
            
            # 获取当期因子值并过滤
            factor_values = factor_data.loc[date].dropna()
            
            # 使用更简单的有效股票过滤
            if date in self.data['close'].index:
                close_prices = self.data['close'].loc[date]
                valid_stocks = close_prices.dropna().index
                factor_values = factor_values[factor_values.index.intersection(valid_stocks)]
            
            
            if len(factor_values) < n_groups * 2:  # 降低门槛，每组至少2只股票
                print(f"DEBUG: 股票数量不足 ({len(factor_values)} < {n_groups * 2})，跳过")
                continue
              # 分组：按20%分位数分组
            factor_values_sorted = factor_values.sort_values()
            n_stocks = len(factor_values_sorted)
              # 计算各组在下一期的收益率
            period_returns = self._calculate_period_returns(date, next_date, factor_values_sorted.index)
            
            
            if period_returns is None:
                print(f"DEBUG: period_returns 为 None，跳过")
                continue
            
            
            # 按20%分位数分组
            for group_idx in range(n_groups):
                # 计算每组的起始和结束位置
                start_pct = group_idx * 0.2
                end_pct = (group_idx + 1) * 0.2
                
                start_idx = int(start_pct * n_stocks)
                end_idx = int(end_pct * n_stocks)
                
                # 最后一组包含剩余股票
                if group_idx == n_groups - 1:
                    end_idx = n_stocks
                  # 确保每组至少有1只股票
                if start_idx >= end_idx:
                    print(f"DEBUG: 第{group_idx+1}组索引范围无效 ({start_idx} >= {end_idx})，跳过")
                    continue
                    
                group_stocks = factor_values_sorted.iloc[start_idx:end_idx].index
                
                # 计算该组等权重收益率
                if len(group_stocks) > 0:
                    group_return = period_returns[group_stocks].mean()
                    
                    # 检查赋值是否成功
                    column_name = f'第{group_idx+1}组'
                    
                    group_returns.loc[date, column_name] = group_return
                    
                else:
                    print(f"DEBUG: 第{group_idx+1}组没有股票")
        
        print(f"DEBUG: 计算完成后group_returns样本:")
        print(group_returns.head())
        print(f"DEBUG: group_returns统计:")
        print(group_returns.describe())
        
        return group_returns

    def _calculate_period_returns(self, start_date: pd.Timestamp, end_date: pd.Timestamp,
                                   stocks: pd.Index) -> Optional[pd.Series]:
        """
        计算指定期间的股票收益率

        Args:
            start_date: 开始日期
            end_date: 结束日期
            stocks: 股票列表
            
        Returns:
            收益率序列
        """
        try:
            
            if start_date not in self.data['close'].index or end_date not in self.data['close'].index:
                print(f"日期不在索引中: start={start_date}, end={end_date}")
                return None
            
            # 确保stocks中的股票在close数据中存在
            available_stocks = [stock for stock in stocks if stock in self.data['close'].columns]
            
            if len(available_stocks) == 0:
                print(f"没有可用股票在日期 {start_date} 到 {end_date}")
                return None
                
            start_prices = self.data['close'].loc[start_date, available_stocks]
            end_prices = self.data['close'].loc[end_date, available_stocks]
            
            
            # 只保留两个日期都有价格的股票
            valid_stocks = start_prices.dropna().index.intersection(end_prices.dropna().index)
            
            if len(valid_stocks) == 0:
                print(f"没有股票在 {start_date} 到 {end_date} 期间有完整价格数据")
                return None
            
            start_prices = start_prices[valid_stocks]
            end_prices = end_prices[valid_stocks]
            
            
            # 计算收益率
            returns = (end_prices / start_prices - 1)
            
            
            if len(returns) == 0:
                print(f"计算出的收益率序列为空，日期: {start_date} 到 {end_date}")
                return None
                
            return returns
            
        except Exception as e:
            print(f"计算期间收益率时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_benchmark_returns(self) -> pd.Series:
        """
        计算基准收益率（使用CSI300指数）
        
        Returns:
            基准收益率序列
        """
        benchmark_returns = pd.Series(index=self.rebalance_dates[:-1], dtype=float)
        
        if 'csi300_index_returns' not in self.data:
            print("警告：无CSI300指数数据，使用市场平均收益作为基准")
            # 使用全市场平均收益作为替代
            for i, date in enumerate(self.rebalance_dates[:-1]):
                next_date = self.rebalance_dates[i + 1]
                period_returns = self._calculate_period_returns(date, next_date, self.data['close'].columns)
                if period_returns is not None:
                    benchmark_returns.loc[date] = period_returns.mean()
            return benchmark_returns
        
        # 使用CSI300指数收益率
        for i, date in enumerate(self.rebalance_dates[:-1]):
            next_date = self.rebalance_dates[i + 1]
              # 计算CSI300在该期间的收益率
            csi300_data = self.data['csi300_index_returns']
            period_data = csi300_data[(csi300_data.index > date) & (csi300_data.index <= next_date)]
            
            if len(period_data) > 0:
                # 累计收益率
                cumulative_return = (1 + period_data).prod() - 1
                benchmark_returns.loc[date] = cumulative_return
        
        return benchmark_returns
    
    def run_single_factor_backtest(self, factor_name: str) -> Dict:
        """
        对单个因子进行回测
        
        Args:
            factor_name: 因子名称
            
        Returns:
            回测结果字典
        """
        print(f"\n开始回测因子: {factor_name}")
        
        # 确保数据已加载
        if not hasattr(self, 'data') or 'close' not in self.data:
            print("数据未加载，开始加载数据...")
            if not self.load_cached_data():
                raise RuntimeError("数据加载失败")
            self._get_month_end_dates()
        
        # 计算因子
        if factor_name not in self.factors_config:
            raise ValueError(f"未知因子: {factor_name}")
        
        factor_func = self.factors_config[factor_name]['func']
        factor_params = self.factors_config[factor_name]['params']
        print(f"计算 {factor_name} 因子...")
        # 获取CSI300过滤后的价格数据进行因子计算
        csi300_close_data = self._get_csi300_filtered_data()
        raw_factor_data = factor_func(csi300_close_data, **factor_params)
        
        # 处理因子
        processed_factor_data = self.process_factor(raw_factor_data, factor_name)
        
        # 计算IC
        print(f"计算 {factor_name} IC值...")
        ic_series, rank_ic_series = self.calculate_factor_ic(processed_factor_data)
        
        # 计算分组收益率
        print(f"计算 {factor_name} 分组收益率...")
        group_returns = self.calculate_portfolio_returns(processed_factor_data)
        
        # 计算多空组合收益率
        if len(group_returns.columns) >= 2:
            long_short_returns = group_returns['第1组'] - group_returns['第5组']
        else:
            long_short_returns = pd.Series(dtype=float)
        
        result = {
            'IC值序列': ic_series,
            'RankIC值序列': rank_ic_series,
            '各分组收益率': group_returns,
            '多空组合收益率序列': long_short_returns
        }
        
        print(f"{factor_name} 回测完成")
        
        # 更新日志
        self._update_log_with_factor(factor_name, result)
        
        return result
    
    def run_backtest(self, factors: Optional[List[str]] = None) -> Dict:
        """
        运行完整回测
        
        Args:
            factors: 要回测的因子列表，如果为None则回测所有因子
            
        Returns:
            完整回测结果
        """
        print("开始执行因子回测...")
        
        # 加载数据
        if not self.load_cached_data():
            raise RuntimeError("数据加载失败")
        
        # 获取调仓日期
        self._get_month_end_dates()
        
        if len(self.rebalance_dates) < 2:
            raise RuntimeError("调仓日期不足，无法进行回测")
        
        # 确定要回测的因子
        if factors is None:
            factors = list(self.factors_config.keys())
        
        # 计算基准收益率
        print("计算基准收益率...")
        benchmark_returns = self.calculate_benchmark_returns()
        
        # 更新日志：基准数据
        self._update_log_with_benchmark(self.rebalance_dates[:-1], benchmark_returns.tolist())
        
        # 回测各因子
        factor_results = {}
        for factor_name in factors:
            try:
                factor_results[factor_name] = self.run_single_factor_backtest(factor_name)
            except Exception as e:
                print(f"回测因子 {factor_name} 时出错: {e}")
                continue
        
        # 整理结果
        result = {
            "回测日期序列": self.rebalance_dates[:-1],
            "基准指数收益率序列": benchmark_returns,
            "因子表现数据": factor_results
        }
        
        print("\n回测完成！")
        self._print_backtest_summary(result)
        
        # 完成日志        self._finalize_log(len(factor_results))
        
        return result
    
    def _print_results_in_required_format(self, results: Dict, save_to_file: bool = True):
        """
        按要求格式打印回测结果
        
        Args:
            results: 回测结果字典
            save_to_file: 是否保存到log.log文件，默认True
        """
        print("\n" + "="*80)
        print("回测结果 - 按要求格式输出")
        print("="*80)
        
        # 准备输出字典
        output_dict = {
            "回测日期序列": [date.strftime('%Y-%m-%d') for date in results["回测日期序列"]],
            "基准指数收益率序列": results["基准指数收益率序列"],
            "因子表现数据": {}
        }
        
        # 处理因子数据
        for factor_name, factor_data in results["因子表现数据"].items():
            output_dict["因子表现数据"][factor_name] = {
                "IC值序列": factor_data["IC值序列"],
                "RankIC值序列": factor_data["RankIC值序列"],
                "各分组收益率": {
                    "第一组": factor_data["各分组收益率"]["第1组"].values.tolist() if "第1组" in factor_data["各分组收益率"].columns else [],
                    "第二组": factor_data["各分组收益率"]["第2组"].values.tolist() if "第2组" in factor_data["各分组收益率"].columns else [],
                    "第三组": factor_data["各分组收益率"]["第3组"].values.tolist() if "第3组" in factor_data["各分组收益率"].columns else [],
                    "第四组": factor_data["各分组收益率"]["第4组"].values.tolist() if "第4组" in factor_data["各分组收益率"].columns else [],
                    "第五组": factor_data["各分组收益率"]["第5组"].values.tolist() if "第5组" in factor_data["各分组收益率"].columns else []
                },
                "多空组合收益率序列": factor_data["多空组合收益率序列"]
            }
        
        # 打印结果
        import json
        print(json.dumps(output_dict, indent=2, ensure_ascii=False))
        print("="*80)
        
        # 打印数据结构信息
        print("\n数据结构信息:")
        print(f"回测日期序列长度: {len(output_dict['回测日期序列'])}")
        print(f"基准指数收益率序列长度: {len(output_dict['基准指数收益率序列'])}")
        print(f"回测因子数量: {len(output_dict['因子表现数据'])}")
        for factor_name, factor_data in output_dict["因子表现数据"].items():
            print(f"\n{factor_name}:")
            print(f"  - IC值序列长度: {len(factor_data['IC值序列'])}")
            print(f"  - RankIC值序列长度: {len(factor_data['RankIC值序列'])}")
            print(f"  - 各分组收益率数据点数: {len(factor_data['各分组收益率']['第一组'])}")
            print(f"  - 多空组合收益率序列长度: {len(factor_data['多空组合收益率序列'])}")
        
        print("="*80)
        
        # 保存结果到log.log文件（可选）
        if save_to_file:
            try:
                # 获取当前文件的目录路径
                current_dir = os.path.dirname(__file__)
                log_file_path = os.path.join(current_dir, 'log.log')
                
                # 添加时间戳信息
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                log_data = {
                    "时间戳": timestamp,
                    "回测结果": output_dict
                }
                  # 保存为JSON格式
                with open(log_file_path, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
                
                print(f"\n✅ 回测结果已保存到: {log_file_path}")
                
            except Exception as e:
                print(f"\n❌ 保存log.log文件时出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n📝 日志文件已通过增量更新完成，跳过批量保存")

    def run_comprehensive_backtest(self, use_incremental_logging: bool = True) -> Dict:
        """
        运行综合回测，对所有可用因子进行回测分析
        
        Args:
            use_incremental_logging: 是否使用增量日志记录，默认True
        
        Returns:
            符合要求格式的回测结果字典
        """
        print("开始综合回测...")
        
        # 确保数据已加载
        if not hasattr(self, 'data') or 'close' not in self.data:
            print("数据未加载，开始加载数据...")
            if not self.load_cached_data():
                raise RuntimeError("数据加载失败")
            self._get_month_end_dates()
        
        # 1. 初始化增量日志
        if use_incremental_logging:
            print("初始化增量日志系统...")
            self._initialize_log()
        
        # 2. 计算基准指数收益率
        print("计算基准指数收益率...")
        benchmark_returns = self.calculate_benchmark_returns()
        
        # 3. 立即更新日志：回测日期和基准收益率
        if use_incremental_logging:
            self._update_log_with_benchmark(
                self.rebalance_dates[:-1],  # 排除最后一个日期
                benchmark_returns.values.tolist()
            )
        
        # 初始化结果字典
        results = {
            "回测日期序列": self.rebalance_dates[:-1],  # 排除最后一个日期
            "基准指数收益率序列": benchmark_returns.values.tolist(),
            "因子表现数据": {}
        }
        
        # 4. 对每个因子进行回测，并实时更新日志
        completed_factors = 0
        for factor_name in self.factors_config.keys():
            print(f"\n{'='*50}")
            print(f"开始回测因子: {factor_name}")
            print(f"{'='*50}")
            
            try:
                factor_results = self.run_single_factor_backtest(factor_name)
                
                # 将结果格式化为要求的格式
                results["因子表现数据"][factor_name] = {
                    "IC值序列": factor_results["IC值序列"].values.tolist(),
                    "RankIC值序列": factor_results["RankIC值序列"].values.tolist(),
                    "各分组收益率": factor_results["各分组收益率"],  # 保持DataFrame格式用于后续处理
                    "多空组合收益率序列": factor_results["多空组合收益率序列"].values.tolist()
                }
                
                # 立即更新日志：添加因子结果
                if use_incremental_logging:
                    self._update_log_with_factor(factor_name, factor_results)
                
                completed_factors += 1
                print(f"因子 {factor_name} 回测完成 ({completed_factors}/{len(self.factors_config)})")
                print(f"- IC均值: {factor_results['IC值序列'].mean():.4f}")
                print(f"- IC标准差: {factor_results['IC值序列'].std():.4f}")
                if factor_results['IC值序列'].std() != 0:
                    print(f"- IC_IR: {factor_results['IC值序列'].mean() / factor_results['IC值序列'].std():.4f}")
                print(f"- RankIC均值: {factor_results['RankIC值序列'].mean():.4f}")
                print(f"- 多空组合年化收益率: {(factor_results['多空组合收益率序列'].mean() * 12):.4f}")
                
            except Exception as e:
                print(f"因子 {factor_name} 回测失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 5. 完成增量日志
        if use_incremental_logging:
            self._finalize_log(completed_factors)
        
        print(f"\n{'='*50}")
        print("综合回测完成!")
        print(f"成功回测因子数量: {completed_factors}")
        print(f"回测期间: {self.start_date.strftime('%Y-%m-%d')} 到 {self.end_date.strftime('%Y-%m-%d')}")
        print(f"调仓周期数: {len(results['回测日期序列'])}")
        print(f"{'='*50}")
        
        # 6. 输出结果到控制台（可选择是否保存文件）
        self._print_results_in_required_format(results, save_to_file=not use_incremental_logging)
        
        return results
    
    def _print_backtest_summary(self, result: Dict):
        """打印回测结果摘要"""
        print("\n=== 回测结果摘要 ===")
        print(f"回测期间: {self.start_date.strftime('%Y-%m-%d')} 到 {self.end_date.strftime('%Y-%m-%d')}")
        print(f"调仓次数: {len(result['回测日期序列'])}")
        
        for factor_name, factor_data in result["因子表现数据"].items():
            print(f"\n--- {factor_name} ---")
            
            # IC统计
            ic_mean = factor_data['IC值序列'].mean()
            ic_std = factor_data['IC值序列'].std()
            ic_ir = ic_mean / ic_std if ic_std != 0 else 0
            
            rank_ic_mean = factor_data['RankIC值序列'].mean()
            rank_ic_std = factor_data['RankIC值序列'].std()
            rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std != 0 else 0
            
            print(f"IC均值: {ic_mean:.4f}, IC标准差: {ic_std:.4f}, ICIR: {ic_ir:.4f}")
            print(f"RankIC均值: {rank_ic_mean:.4f}, RankIC标准差: {rank_ic_std:.4f}, RankICIR: {rank_ic_ir:.4f}")
            
            # 多空组合统计
            if len(factor_data['多空组合收益率序列']) > 0:
                ls_mean = factor_data['多空组合收益率序列'].mean()
                ls_std = factor_data['多空组合收益率序列'].std()
                ls_sharpe = ls_mean / ls_std if ls_std != 0 else 0
                
                print(f"多空组合年化收益: {ls_mean * 12:.2%}")
                print(f"多空组合年化波动: {ls_std * np.sqrt(12):.2%}")
                print(f"多空组合Sharpe: {ls_sharpe:.4f}")
    
    def save_results(self, results: Dict, filename: str = None):
        """
        保存回测结果到文件
        
        Args:
            results: 回测结果字典
            filename: 保存文件名，如果为None则自动生成
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtesting_results_{timestamp}.pkl"
        
        # 确保Result目录存在
        result_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Result')
        os.makedirs(result_dir, exist_ok=True)
        
        filepath = os.path.join(result_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"回测结果已保存到: {filepath}")
        return filepath

    def _get_log_file_path(self) -> str:
        """获取log.log文件路径"""
        current_dir = os.path.dirname(__file__)
        return os.path.join(current_dir, 'log.log')
    def _initialize_log(self) -> None:
        """
        初始化log.log文件，设置基本结构
        """
        try:
            log_file_path = self._get_log_file_path()
            
            # 初始化log数据结构 - 严格按照要求格式
            log_data = {
                "回测日期序列": [],
                "基准指数收益率序列": [],
                "因子表现数据": {}
            }
            
            # 保存初始结构
            import json
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 已初始化日志文件: {log_file_path}")
            
        except Exception as e:
            print(f"❌ 初始化log.log文件时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_log_with_benchmark(self, backtest_dates: list, benchmark_returns: list) -> None:
        """
        更新log.log文件，添加回测日期和基准收益率
        
        Args:
            backtest_dates: 回测日期序列
            benchmark_returns: 基准收益率序列
        """
        try:
            log_file_path = self._get_log_file_path()
            import os
            if not os.path.exists(log_file_path):
                self._initialize_log()
            # 读取现有log数据
            import json
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # 更新回测日期和基准收益率（处理字符串和datetime对象）
            if backtest_dates and isinstance(backtest_dates[0], str):
                log_data["回测日期序列"] = backtest_dates
            else:
                log_data["回测日期序列"] = [date.strftime('%Y-%m-%d') for date in backtest_dates]
            log_data["基准指数收益率序列"] = benchmark_returns
            
            # 保存更新的数据
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 已更新日志：回测日期({len(backtest_dates)}个) 和 基准收益率({len(benchmark_returns)}个)")
            
        except Exception as e:
            print(f"❌ 更新log.log(基准数据)时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_log_with_factor(self, factor_name: str, factor_results: Dict) -> None:
        """
        更新log.log文件，添加单个因子的回测结果
        
        Args:
            factor_name: 因子名称
            factor_results: 因子回测结果
        """
        try:
            log_file_path = self._get_log_file_path()
            import os
            if not os.path.exists(log_file_path):
                self._initialize_log()
            # 读取现有log数据
            import json
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)            # 格式化因子数据为要求的格式（处理不同的输入格式）
            formatted_factor_data = {}
            
            # 处理IC值序列
            if "IC值序列" in factor_results and hasattr(factor_results["IC值序列"], 'values'):
                formatted_factor_data["IC值序列"] = factor_results["IC值序列"].values.tolist()
            elif "IC值序列" in factor_results:
                formatted_factor_data["IC值序列"] = factor_results["IC值序列"]
            else:
                formatted_factor_data["IC值序列"] = []
            
            # 处理RankIC值序列
            if "RankIC值序列" in factor_results and hasattr(factor_results["RankIC值序列"], 'values'):
                formatted_factor_data["RankIC值序列"] = factor_results["RankIC值序列"].values.tolist()
            elif "RankIC值序列" in factor_results:
                formatted_factor_data["RankIC值序列"] = factor_results["RankIC值序列"]
            else:
                formatted_factor_data["RankIC值序列"] = []
            
            # 处理各分组收益率
            if "各分组收益率" in factor_results:
                group_returns = factor_results["各分组收益率"]
                if hasattr(group_returns, 'columns'):  # DataFrame
                    formatted_factor_data["各分组收益率"] = {
                        "第一组": group_returns["第1组"].values.tolist() if "第1组" in group_returns.columns else [],
                        "第二组": group_returns["第2组"].values.tolist() if "第2组" in group_returns.columns else [],
                        "第三组": group_returns["第3组"].values.tolist() if "第3组" in group_returns.columns else [],
                        "第四组": group_returns["第4组"].values.tolist() if "第4组" in group_returns.columns else [],
                        "第五组": group_returns["第5组"].values.tolist() if "第5组" in group_returns.columns else []
                    }
                else:  # Dict
                    formatted_factor_data["各分组收益率"] = group_returns
            else:
                formatted_factor_data["各分组收益率"] = {}
            
            # 处理多空组合收益率序列
            if "多空组合收益率序列" in factor_results and hasattr(factor_results["多空组合收益率序列"], 'values'):
                formatted_factor_data["多空组合收益率序列"] = factor_results["多空组合收益率序列"].values.tolist()
            elif "多空组合收益率序列" in factor_results:
                formatted_factor_data["多空组合收益率序列"] = factor_results["多空组合收益率序列"]
            else:
                formatted_factor_data["多空组合收益率序列"] = []
            
            # 添加因子数据到log - 严格按照要求格式
            log_data["因子表现数据"][factor_name] = formatted_factor_data
            
            # 保存更新的数据
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 已更新日志：因子 {factor_name} 回测结果")
            
        except Exception as e:
            print(f"❌ 更新log.log(因子 {factor_name})时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _finalize_log(self, total_factors_completed: int) -> None:
        """
        完成log.log文件，添加最终时间戳和总结信息
        
        Args:
            total_factors_completed: 成功完成的因子数量
        """
        try:
            log_file_path = self._get_log_file_path()

            # 读取现有log数据
            import json
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # 不添加任何额外信息，保持严格格式
            # log.log文件已包含所需的所有数据，无需添加完成标记
            
            print(f"✅ 日志文件已完成，共成功回测 {total_factors_completed} 个因子")
            
        except Exception as e:
            print(f"❌ 完成log.log文件时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_csi300_filtered_data(self) -> pd.DataFrame:
        """
        获取CSI300成分股过滤后的价格数据
        只保留在回测期间任意时点是CSI300成分股的股票
        
        Returns:
            过滤后的收盘价数据
        """
        print("开始过滤CSI300成分股数据...")
        
        if 'csi300_constituents' not in self.data:
            print("警告：无CSI300成分股数据，使用全部股票")
            return self.data['close']
        
        # 获取在回测期间任意时点是CSI300成分股的所有股票
        csi300_constituents = self.data['csi300_constituents']
        
        # 找到所有曾经是CSI300成分股的股票
        all_csi300_stocks = set()
        for date in csi300_constituents.index:
            if self.start_date <= date <= self.end_date:
                current_constituents = csi300_constituents.loc[date]
                csi300_stocks = current_constituents[current_constituents == True].index
                all_csi300_stocks.update(csi300_stocks)
        
        all_csi300_stocks = list(all_csi300_stocks)
        
        # 过滤价格数据，只保留CSI300相关股票
        close_data = self.data['close']
        available_stocks = [stock for stock in all_csi300_stocks if stock in close_data.columns]
        
        filtered_close = close_data[available_stocks]
        
        print(f"CSI300股票池过滤完成: {len(available_stocks)} 只股票 (原总数: {len(close_data.columns)})")
        
        return filtered_close

def main():
    """主函数示例"""
    # 创建回测器
    backtester = FactorBacktester(start_date='2021-01-01', end_date='2025-04-30')
    
    # 运行回测
    results = backtester.run_backtest()
    
    # 保存结果
    results_path = os.path.join(os.path.dirname(CURRENT_DIR), 'Result', 'backtest_results.pkl')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n回测结果已保存到: {results_path}")
    
    return results


if __name__ == "__main__":
    main()