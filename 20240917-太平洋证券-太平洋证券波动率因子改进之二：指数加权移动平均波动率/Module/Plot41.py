"""
绘图模块 - Plot41.py
功能：绘制EWMAVOL因子沪深300内选股分年表现图表
图表内容：按年份显示EWMAVOL因子在沪深300成分股中的选股表现
要求：
- 样本：沪深300为股票池，剔除停牌、ST等交易异常股票
- 回测区间：2008.12.31至2024.06.07
- 调仓频率：月度
- 组合权重分配：等权
- 因子处理方式：因子方向调整、缩尾调整、市值行业中性化、标准化
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from typing import Dict, List, Tuple
import sys
import warnings
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# 导入因子模块
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Factor'))
try:
    from EWMAVOL import calculate_ewmavol
except ImportError:
    print("警告：无法导入EWMAVOL模块")
    calculate_ewmavol = None

# --- 全局设置 ---
# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 优先使用微软雅黑
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"设置中文字体失败 (尝试了Microsoft YaHei, SimHei): {e}。可能显示为方块。")
    plt.rcParams['font.sans-serif'] = ['sans-serif']  # 降级到通用无衬线体
    plt.rcParams['axes.unicode_minus'] = False

# --- 路径设置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
CACHE_DIR = os.path.join(CURRENT_DIR, 'cache')
RESULT_DIR = os.path.join(PROJECT_ROOT, "Result")

# 确保Result目录存在
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
    
# 图表颜色设置
COLORS = plt.cm.tab10.colors
COLOR_REPORT_HEADER_BG = '#FDEADA'  # 研报表头背景色 (一种浅橙黄色)
COLOR_REPORT_CELL_BG = '#FFFFFF'    # 研报数据单元格背景色 (白色)
COLOR_REPORT_FACTOR_BG = '#F2F2F2'  # 研报因子名称列背景色 (浅灰色)
COLOR_REPORT_RED_TEXT = '#FF0000'   # 红色文字
COLOR_REPORT_DARK_YELLOW_TEXT = '#B8860B'  # 深黄色文字
COLOR_REPORT_BORDER = '#BFBFBF'     # 研报表格边框颜色 (稍深一点的灰色)

# --- 数据加载函数 ---
def load_cached_data(data_type: str) -> pd.DataFrame:
    """
    加载缓存的数据文件
    Args:
        data_type: 数据类型（如'close', 'csi300_constituents'等）
    Returns:
        pd.DataFrame: 加载的数据框
    """
    cache_file = os.path.join(CACHE_DIR, f'processed_{data_type}.pkl')
    if not os.path.exists(cache_file):
        print(f"错误：缓存文件 {cache_file} 不存在")
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"成功加载缓存文件: {data_type}")
        return data
    except Exception as e:
        print(f"加载缓存文件 {cache_file} 时出错: {e}")
        return None

def calculate_monthly_returns(
        factor_data: pd.DataFrame, 
        close_data: pd.DataFrame, 
        constituents_data: pd.DataFrame,
        benchmark_returns: pd.Series,
        market_cap_data: pd.DataFrame = None,
        industry_data: pd.DataFrame = None,
        start_date: str = '2009-01-01', 
        end_date: str = '2024-06-07',
        n_groups: int = 5,
        rebalance_freq: str = 'M'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    根据因子值计算分组月度收益
    
    Args:
        factor_data: 因子值DataFrame，索引为日期，列为股票代码
        close_data: 收盘价DataFrame，索引为日期，列为股票代码
        constituents_data: 沪深300成分股DataFrame，索引为日期，列为股票代码，值为布尔值
        benchmark_returns: 基准收益率Series
        market_cap_data: 市值数据DataFrame，用于中性化处理
        industry_data: 行业数据DataFrame，用于中性化处理
        start_date: 回测开始日期
        end_date: 回测结束日期
        n_groups: 分组数量
        rebalance_freq: 调仓频率，'M'为月度
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series]: (分组月度收益率, 分组月度基准收益率, 多空组合收益率)
    """
    print("计算分组月度收益...")
    
    # 转换日期格式
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # 确保数据在回测区间内
    factor_data = factor_data[(factor_data.index >= start_date) & (factor_data.index <= end_date)]
    close_data = close_data[(close_data.index >= start_date) & (close_data.index <= end_date)]
    constituents_data = constituents_data[(constituents_data.index >= start_date) & (constituents_data.index <= end_date)]
    benchmark_returns = benchmark_returns[(benchmark_returns.index >= start_date) & (benchmark_returns.index <= end_date)]
    
    if market_cap_data is not None:
        market_cap_data = market_cap_data[(market_cap_data.index >= start_date) & (market_cap_data.index <= end_date)]
    
    if industry_data is not None:
        industry_data = industry_data[(industry_data.index >= start_date) & (industry_data.index <= end_date)]
    
    # 计算每日收益率
    daily_returns = close_data.pct_change()
    
    # 获取月度调仓日期
    if rebalance_freq == 'M':
        # 获取每月第一个交易日作为调仓日
        trading_days = factor_data.index
        rebalance_dates = []
        for year in range(start_date.year, end_date.year + 1):
            for month in range(1, 13):
                month_start = pd.Timestamp(year, month, 1)
                month_trading_days = [d for d in trading_days if d.year == year and d.month == month]
                if month_trading_days:
                    rebalance_dates.append(min(month_trading_days))
        rebalance_dates = sorted(set(rebalance_dates))
    else:
        raise ValueError(f"不支持的调仓频率: {rebalance_freq}")
    
    # 存储月度收益率
    monthly_group_returns = []
    monthly_benchmark_returns = []
    monthly_long_short_returns = []
    monthly_dates = []
    
    # 对每个调仓日进行回测
    for i, current_date in enumerate(rebalance_dates):
        if i == len(rebalance_dates) - 1:
            # 最后一个调仓日，持有至回测结束
            next_rebalance_date = end_date
        else:
            next_rebalance_date = rebalance_dates[i + 1]
            
        # 获取当前调仓日的因子值
        if current_date not in factor_data.index:
            continue
            
        current_factor = factor_data.loc[current_date]
          # 只考虑沪深300成分股
        if constituents_data is not None and current_date in constituents_data.index:
            current_constituents = constituents_data.loc[current_date]
            valid_stocks = current_constituents[current_constituents].index.tolist()
            current_factor = current_factor[valid_stocks].dropna()
        else:
            current_factor = current_factor.dropna()
        
        if len(current_factor) < n_groups:
            continue
        
        # 市值和行业中性化处理
        if market_cap_data is not None and industry_data is not None and current_date in market_cap_data.index and current_date in industry_data.index:
            try:
                current_factor = neutralize_factor(
                    current_factor, 
                    market_cap_data.loc[current_date], 
                    industry_data.loc[current_date]
                )
            except Exception as e:
                print(f"中性化处理失败: {e}")
        
        # 缩尾处理（去除极端值）
        lower_bound = current_factor.quantile(0.01)
        upper_bound = current_factor.quantile(0.99)
        current_factor = current_factor[(current_factor >= lower_bound) & (current_factor <= upper_bound)]
        
        if len(current_factor) < n_groups:
            continue
        
        # 标准化处理
        current_factor = (current_factor - current_factor.mean()) / current_factor.std()
        
        # 因子方向调整（EWMAVOL因子需要反转，波动率低的股票预期收益高）
        current_factor = -current_factor
        
        # 划分分组（因子值从大到小排序，组1为因子值最高的组）
        try:
            quantile_values = [1/n_groups * i for i in range(n_groups+1)]
            groups = pd.qcut(current_factor.rank(method='first'), q=quantile_values, labels=range(1, n_groups+1))
        except Exception as e:
            print(f"分组失败: {e}")
            continue
        
        # 构建组合权重（等权重）
        weights = {}
        for group in range(1, n_groups+1):
            group_stocks = groups[groups == group].index.tolist()
            if group_stocks:
                group_weight = 1.0 / len(group_stocks)
                weights[group] = {stock: group_weight for stock in group_stocks}
            else:
                weights[group] = {}
        
        # 计算持有期内的收益率
        hold_period = pd.date_range(current_date, next_rebalance_date, freq='D')
        hold_period = [date for date in hold_period if date in daily_returns.index and date != current_date]
        
        if not hold_period:
            continue
        
        # 计算月度组合收益率
        month_group_returns = {}
        for group in range(1, n_groups+1):
            if group not in weights or not weights[group]:
                month_group_returns[group] = 0
                continue
                
            group_stocks = list(weights[group].keys())
            group_weights = np.array(list(weights[group].values()))
            
            # 计算组合收益
            group_daily_returns = []
            for date in hold_period:
                if date in daily_returns.index:
                    stock_returns = daily_returns.loc[date, group_stocks].fillna(0).values
                    if len(stock_returns) > 0:
                        group_return = np.sum(stock_returns * group_weights)
                    else:
                        group_return = 0
                    group_daily_returns.append(group_return)
            
            # 计算月度总收益率
            if group_daily_returns:
                cumulative_return = np.prod([1 + r for r in group_daily_returns]) - 1
                month_group_returns[group] = cumulative_return
            else:
                month_group_returns[group] = 0
        
        # 计算基准月度收益率
        benchmark_daily_returns = []
        for date in hold_period:
            if date in benchmark_returns.index:
                benchmark_daily_returns.append(benchmark_returns.loc[date])
        
        if benchmark_daily_returns:
            benchmark_monthly_return = np.prod([1 + r for r in benchmark_daily_returns]) - 1
        else:
            benchmark_monthly_return = 0
        
        # 计算多空组合收益率（第1组 - 第5组）
        long_short_return = month_group_returns.get(1, 0) - month_group_returns.get(n_groups, 0)
        
        # 存储结果
        monthly_group_returns.append(month_group_returns)
        monthly_benchmark_returns.append(benchmark_monthly_return)
        monthly_long_short_returns.append(long_short_return)
        monthly_dates.append(current_date)
    
    # 转换为DataFrame
    group_returns_df = pd.DataFrame(monthly_group_returns, index=monthly_dates)
    benchmark_returns_series = pd.Series(monthly_benchmark_returns, index=monthly_dates)
    long_short_returns_series = pd.Series(monthly_long_short_returns, index=monthly_dates)
    
    return group_returns_df, benchmark_returns_series, long_short_returns_series

def neutralize_factor(factor_series, market_cap_series, industry_series):
    """
    对因子进行市值和行业中性化处理
    
    Args:
        factor_series: 因子值Series，索引为股票代码
        market_cap_series: 市值Series，索引为股票代码
        industry_series: 行业Series，索引为股票代码，值为行业代码
    
    Returns:
        pd.Series: 中性化后的因子值
    """
    try:
        # 确保所有Series有相同的股票，并且去除NaN值
        factor_clean = factor_series.dropna()
        market_cap_clean = market_cap_series.dropna()
        industry_clean = industry_series.dropna()
        
        # 确保market_cap_series中的值是数值型
        market_cap_clean = pd.to_numeric(market_cap_clean, errors='coerce').dropna()
        
        # 找到共同的股票
        common_stocks = factor_clean.index.intersection(market_cap_clean.index).intersection(industry_clean.index)
        
        if len(common_stocks) < 10:  # 至少需要10只股票进行回归
            print(f"警告：共同股票数量不足({len(common_stocks)})，跳过中性化处理")
            return factor_series
            
        factor_values = factor_clean[common_stocks]
        market_cap_values = market_cap_clean[common_stocks]
        industry_values = industry_clean[common_stocks]
        
        # 对市值取对数
        market_cap_values = market_cap_values.replace(0, np.nan)
        market_cap_values = market_cap_values[market_cap_values > 0]  # 只保留正值
        
        if len(market_cap_values) < 10:
            print("警告：有效市值数据不足，跳过中性化处理")
            return factor_series
            
        # 重新确定共同股票
        common_stocks = factor_values.index.intersection(market_cap_values.index).intersection(industry_values.index)
        factor_values = factor_values[common_stocks]
        market_cap_values = market_cap_values[common_stocks]
        industry_values = industry_values[common_stocks]
        
        log_market_cap = np.log(market_cap_values)
        
        # 创建行业哑变量
        industry_dummies = pd.get_dummies(industry_values, prefix='industry')
        
        # 合并自变量
        X = pd.concat([log_market_cap.rename('log_market_cap'), industry_dummies], axis=1)
        X = sm.add_constant(X)
        
        # 确保X中没有NaN值
        X = X.dropna()
        factor_values = factor_values[X.index]
        
        if len(X) < 10:
            print("警告：回归数据不足，跳过中性化处理")
            return factor_series
          # 回归模型
        try:
            # 确保数据类型为数值型
            X_clean = X.astype(float)
            factor_clean = factor_values.astype(float)
            
            model = sm.OLS(factor_clean, X_clean)
            results = model.fit()
            
            # 提取残差作为中性化后的因子
            neutralized_factor = pd.Series(results.resid, index=factor_values.index)
        except Exception as e:
            print(f"回归计算失败: {e}")
            return factor_series
          # 对于不在回归中的股票，保留原来的因子值
        missing_stocks = factor_series.index.difference(neutralized_factor.index)
        if len(missing_stocks) > 0:
            neutralized_factor = pd.concat([neutralized_factor, factor_series[missing_stocks]])
        
        return neutralized_factor
        
    except Exception as e:
        print(f"中性化处理出现错误: {e}")
        return factor_series

def calculate_comprehensive_statistics(group_returns_df, benchmark_returns, long_short_returns):
    """
    计算全面的统计指标，包括年度和整体统计
    
    Args:
        group_returns_df: 分组月度收益率DataFrame
        benchmark_returns: 基准月度收益率Series
        long_short_returns: 多空组合月度收益率Series
    
    Returns:
        Dict: 包含年度和整体统计指标的字典
    """
    print("计算全面统计指标...")
    
    # 确保数据对齐
    common_dates = group_returns_df.index.intersection(benchmark_returns.index).intersection(long_short_returns.index)
    group_returns = group_returns_df.loc[common_dates]
    benchmark = benchmark_returns.loc[common_dates]
    long_short = long_short_returns.loc[common_dates]
    
    # 按年份分组
    years = sorted(set([date.year for date in common_dates]))
    
    statistics = {}
    
    # 计算每年的统计指标
    for year in years:
        year_mask = [date.year == year for date in common_dates]
        year_dates = common_dates[year_mask]
        
        if len(year_dates) == 0:
            continue
            
        year_group_returns = group_returns.loc[year_dates]
        year_benchmark = benchmark.loc[year_dates]
        year_long_short = long_short.loc[year_dates]
        
        statistics[year] = {}
        
        # 多头组合统计（第1组）
        if 1 in year_group_returns.columns:
            long_returns = year_group_returns[1].dropna()
            long_benchmark = year_benchmark.loc[long_returns.index]
            
            # 年化收益率
            annual_return = (1 + long_returns.mean()) ** 12 - 1
            
            # 超额收益率
            excess_returns = long_returns - long_benchmark
            annual_excess_return = (1 + excess_returns.mean()) ** 12 - 1
            
            # 夏普比率
            if long_returns.std() != 0:
                sharpe_ratio = long_returns.mean() / long_returns.std() * np.sqrt(12)
            else:
                sharpe_ratio = 0
            
            # 超额回撤（这里简化为超额收益的最大回撤）
            excess_cumulative = (1 + excess_returns).cumprod()
            excess_drawdown = (excess_cumulative / excess_cumulative.cummax() - 1).min()
            
            # 相对胜率
            win_rate = (excess_returns > 0).mean()
            
            # 信息比率
            if excess_returns.std() != 0:
                info_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
            else:
                info_ratio = 0
            
            statistics[year]['long'] = {
                'annual_return': annual_return,
                'excess_return': annual_excess_return,
                'sharpe_ratio': sharpe_ratio,
                'excess_drawdown': excess_drawdown,
                'win_rate': win_rate,
                'info_ratio': info_ratio
            }
        
        # 多空组合统计
        if len(year_long_short) > 0:
            ls_returns = year_long_short.dropna()
            ls_benchmark = year_benchmark.loc[ls_returns.index]
            
            # 年化收益率
            annual_return = (1 + ls_returns.mean()) ** 12 - 1
            
            # 超额收益率
            excess_returns = ls_returns - ls_benchmark
            annual_excess_return = (1 + excess_returns.mean()) ** 12 - 1
            
            # 夏普比率
            if ls_returns.std() != 0:
                sharpe_ratio = ls_returns.mean() / ls_returns.std() * np.sqrt(12)
            else:
                sharpe_ratio = 0
            
            # 最大回撤
            cumulative = (1 + ls_returns).cumprod()
            max_drawdown = (cumulative / cumulative.cummax() - 1).min()
            
            # 胜率
            win_rate = (ls_returns > 0).mean()
            
            # 信息比率
            if excess_returns.std() != 0:
                info_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
            else:
                info_ratio = 0
            
            statistics[year]['long_short'] = {
                'annual_return': annual_return,
                'excess_return': annual_excess_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'info_ratio': info_ratio
            }
    
    # 计算整体统计
    if 1 in group_returns.columns:
        long_returns = group_returns[1].dropna()
        long_benchmark = benchmark.loc[long_returns.index]
        
        # 年化收益率
        annual_return = (1 + long_returns.mean()) ** 12 - 1
        
        # 超额收益率
        excess_returns = long_returns - long_benchmark
        annual_excess_return = (1 + excess_returns.mean()) ** 12 - 1
        
        # 夏普比率
        if long_returns.std() != 0:
            sharpe_ratio = long_returns.mean() / long_returns.std() * np.sqrt(12)
        else:
            sharpe_ratio = 0
        
        # 超额回撤
        excess_cumulative = (1 + excess_returns).cumprod()
        excess_drawdown = (excess_cumulative / excess_cumulative.cummax() - 1).min()
        
        # 相对胜率
        win_rate = (excess_returns > 0).mean()
        
        # 信息比率
        if excess_returns.std() != 0:
            info_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
        else:
            info_ratio = 0
        
        statistics['overall'] = {
            'long': {
                'annual_return': annual_return,
                'excess_return': annual_excess_return,
                'sharpe_ratio': sharpe_ratio,
                'excess_drawdown': excess_drawdown,
                'win_rate': win_rate,
                'info_ratio': info_ratio
            }
        }
    
    # 多空组合整体统计
    if len(long_short) > 0:
        ls_returns = long_short.dropna()
        ls_benchmark = benchmark.loc[ls_returns.index]
        
        # 年化收益率
        annual_return = (1 + ls_returns.mean()) ** 12 - 1
        
        # 超额收益率
        excess_returns = ls_returns - ls_benchmark
        annual_excess_return = (1 + excess_returns.mean()) ** 12 - 1
        
        # 夏普比率
        if ls_returns.std() != 0:
            sharpe_ratio = ls_returns.mean() / ls_returns.std() * np.sqrt(12)
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        cumulative = (1 + ls_returns).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        
        # 胜率
        win_rate = (ls_returns > 0).mean()
        
        # 信息比率
        if excess_returns.std() != 0:
            info_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
        else:
            info_ratio = 0
        
        if 'overall' not in statistics:
            statistics['overall'] = {}
            
        statistics['overall']['long_short'] = {
            'annual_return': annual_return,
            'excess_return': annual_excess_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'info_ratio': info_ratio
        }
    
    return statistics



def generate_table_image(statistics, output_path):
    """
    生成表格图片
    
    Args:
        statistics: 统计指标字典
        output_path: 输出图片路径
    """
    print("生成表格图片...")
    
    try:
        # 创建图表
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('off')
        
        # 准备表格数据
        headers = ['年份', '多头年化收益', '多头超额收益', '多头夏普比率', '多头超额回撤', '多头相对胜率', '多头信息比率',
                  '多空年化收益', '多空超额收益', '多空夏普比率', '多空最大回撤', '多空胜率', '多空信息比率']
        
        table_data = []
        table_data.append(headers)
          # 添加年度数据
        years = sorted([year for year in statistics.keys() if isinstance(year, int) and year >= 2009])
        
        for year in years:
            if year not in statistics:
                continue
                
            row = [str(year)]
            
            # 多头数据
            if 'long' in statistics[year]:
                long_data = statistics[year]['long']
                row.extend([
                    f"{long_data['annual_return']*100:.1f}%",
                    f"{long_data['excess_return']*100:.1f}%", 
                    f"{long_data['sharpe_ratio']:.2f}",
                    f"{abs(long_data['excess_drawdown'])*100:.1f}%",

                    f"{long_data['win_rate']*100:.1f}%",
                    f"{long_data['info_ratio']:.2f}"
                ])
            else:
                row.extend(["-", "-", "-", "-", "-", "-"])
            
            # 多空数据  
            if 'long_short' in statistics[year]:
                ls_data = statistics[year]['long_short']
                row.extend([
                    f"{ls_data['annual_return']*100:.1f}%",
                    f"{ls_data['excess_return']*100:.1f}%",
                    f"{ls_data['sharpe_ratio']:.2f}",
                    f"{abs(ls_data['max_drawdown'])*100:.1f}%",

                    f"{ls_data['win_rate']*100:.1f}%",
                    f"{ls_data['info_ratio']:.2f}"
                ])
            else:
                row.extend(["-", "-", "-", "-", "-", "-"])
                
            table_data.append(row)
        
        # 添加整体数据
        if 'overall' in statistics:
            row = ['整体']
            
            # 多头整体数据
            if 'long' in statistics['overall']:
                long_data = statistics['overall']['long']
                row.extend([
                    f"{long_data['annual_return']*100:.1f}%",
                    f"{long_data['excess_return']*100:.1f}%", 
                    f"{long_data['sharpe_ratio']:.2f}",
                    f"{abs(long_data['excess_drawdown'])*100:.1f}%",

                    f"{long_data['win_rate']*100:.1f}%",
                    f"{long_data['info_ratio']:.2f}"
                ])
            else:
                row.extend(["-", "-", "-", "-", "-", "-"])
            
            # 多空整体数据
            if 'long_short' in statistics['overall']:
                ls_data = statistics['overall']['long_short']
                row.extend([
                    f"{ls_data['annual_return']*100:.1f}%",
                    f"{ls_data['excess_return']*100:.1f}%",
                    f"{ls_data['sharpe_ratio']:.2f}",
                    f"{abs(ls_data['max_drawdown'])*100:.1f}%",

                    f"{ls_data['win_rate']*100:.1f}%",
                    f"{ls_data['info_ratio']:.2f}"
                ])
            else:
                row.extend(["-", "-", "-", "-", "-", "-"])
                
            table_data.append(row)
        
        # 创建表格
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                        cellLoc='center', loc='center')
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        # 设置标题
        plt.title('EWMAVOL因子沪深300内选股分年表现统计表', fontsize=16, fontweight='bold', pad=20)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"表格图片已保存至: {output_path}")
        
    except Exception as e:
        print(f"生成表格图片时出错: {e}")
        # 如果图片生成失败，创建一个简单的文本文件作为备份
        backup_file = output_path.replace('.png', '_backup.txt')
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write("表格图片生成失败，但数据处理成功完成。\n")
                f.write("统计数据已成功计算，但无法生成图片文件。\n")
            print(f"备份文本文件已保存至: {backup_file}")
        except:
            print("无法创建备份文件")

def main():
    """主函数：加载数据，计算因子分年表现，生成markdown表格"""    # 设置回测参数
    factor_name = "EWMAVOL"
    start_date = '2009-01-01'
    end_date = '2024-06-07'
    n_groups = 5
    rebalance_freq = 'M'
    
    print(f"开始生成{factor_name}因子沪深300内选股分年表现表格...")
    
    # 加载必要的数据
    print("加载数据...")
    close_data = load_cached_data('close')
    market_cap_data = load_cached_data('market_cap')
    industry_data = load_cached_data('industry')
    csi300_constituents = load_cached_data('csi300_constituents')
    csi300_index_returns = load_cached_data('csi300_index_returns')
    
    if close_data is None or csi300_constituents is None or csi300_index_returns is None:
        print("错误：无法加载必要的数据")
        return
    
    print(f"数据加载完成:")
    print(f"  - 收盘价数据: {close_data.shape}")
    print(f"  - CSI300成分股数据: {csi300_constituents.shape}")
    print(f"  - CSI300指数收益率: {csi300_index_returns.shape}")
    if market_cap_data is not None:
        print(f"  - 市值数据: {market_cap_data.shape}")
    if industry_data is not None:
        print(f"  - 行业数据: {industry_data.shape}")
    
    # 计算EWMAVOL因子
    if calculate_ewmavol is None:
        print("错误：无法导入EWMAVOL因子计算函数")
        return
        
    try:
        print(f"计算{factor_name}因子...")
        factor_data = calculate_ewmavol(close_data, window=60, lambda_decay=0.9)
        print(f"{factor_name}因子计算完成，数据维度: {factor_data.shape}")
    except Exception as e:
        print(f"计算{factor_name}因子时出错: {e}")
        return
    
    # 计算分组收益
    print("计算分组月度收益...")
    group_returns_df, benchmark_returns_series, long_short_returns_series = calculate_monthly_returns(
        factor_data, close_data, csi300_constituents, csi300_index_returns, 
        market_cap_data, industry_data,
        start_date=start_date, end_date=end_date, n_groups=n_groups, rebalance_freq=rebalance_freq
    )
    
    print(f"分组收益计算完成:")
    print(f"  - 分组收益数据维度: {group_returns_df.shape}")
    print(f"  - 基准收益数据长度: {len(benchmark_returns_series)}")
    print(f"  - 多空收益数据长度: {len(long_short_returns_series)}")
    
    # 计算综合统计数据
    print("计算综合统计指标...")
    comprehensive_stats = calculate_comprehensive_statistics(
        group_returns_df, benchmark_returns_series, long_short_returns_series
    )
      # 生成并保存表格图片
    image_output_file = os.path.join(RESULT_DIR, f"{factor_name}因子沪深300内选股分年表现.png")
    generate_table_image(comprehensive_stats, image_output_file)
    
    print("表格生成完成！")

if __name__ == "__main__":
    main()