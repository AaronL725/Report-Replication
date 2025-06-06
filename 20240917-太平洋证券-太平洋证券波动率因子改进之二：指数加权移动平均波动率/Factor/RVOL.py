# filepath: RVOL.py
"""
RVOL因子计算模块
功能：计算特质波动率因子
定义：过去N天（研报默认为60天）日收益率序列对Fama-French三因子模型回归残差的波动率
"""

import pandas as pd
import numpy as np
import warnings
from multiprocessing import Pool, cpu_count
import functools
warnings.filterwarnings('ignore')


def fast_linear_regression(X, y):
    """
    使用numpy实现快速线性回归，比sklearn.LinearRegression更快
    
    Args:
        X: 自变量矩阵 (n_samples, n_features)
        y: 因变量向量 (n_samples,)
    
    Returns:
        residuals: 回归残差
    """
    try:
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # 使用numpy的最小二乘法求解
        coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        
        # 计算预测值和残差
        predictions = X_with_intercept @ coeffs
        residuals = y - predictions
        
        return residuals
    except:
        return None


def calculate_single_stock_rvol(args):
    """
    计算单只股票的RVOL因子 - 用于并行计算
    
    Args:
        args: (stock_idx, stock_code, stock_excess_returns_np, ff_regressors_np, 
               window, min_obs_reg, dates_len)
    
    Returns:
        (stock_code, rvol_values)
    """
    stock_idx, stock_code, stock_excess_returns_np, ff_regressors_np, window, min_obs_reg, dates_len = args
    
    # 预分配结果数组
    stock_rvol_values = np.full(dates_len, np.nan)
    
    # 滚动窗口计算
    for i in range(window - 1, dates_len):
        start_pos = i - window + 1
        end_pos = i + 1
        
        # 获取当前窗口数据
        current_stock_ret_window = stock_excess_returns_np[start_pos:end_pos]
        current_ff_factors_window = ff_regressors_np[start_pos:end_pos, :]
        
        # 处理NaN值
        valid_stock_mask = ~np.isnan(current_stock_ret_window)
        valid_ff_mask = ~np.isnan(current_ff_factors_window).any(axis=1)
        common_valid_mask = valid_stock_mask & valid_ff_mask
        
        if np.sum(common_valid_mask) < min_obs_reg:
            continue
            
        y_clean = current_stock_ret_window[common_valid_mask]
        X_clean = current_ff_factors_window[common_valid_mask, :]
        
        # 快速线性回归
        residuals = fast_linear_regression(X_clean, y_clean)
        
        if residuals is not None and len(residuals) >= 2:
            stock_rvol_values[i] = np.std(residuals, ddof=1)
    
    return stock_code, stock_rvol_values
    """
    计算RVOL因子：特质波动率因子

    Args:
        close_prices (pd.DataFrame): 收盘价数据，索引为日期，列为股票代码
        ff_factors (pd.DataFrame): Fama-French三因子数据，索引为日期，
                                   必须包含列: 'Mkt-RF', 'SMB', 'HML', 'RF' (假设为百分比形式)
        window (int): 滚动窗口大小，默认为60天

    Returns:
        pd.DataFrame: RVOL因子值，索引为日期，列为股票代码
    """
    
    # 数据验证
    if not isinstance(close_prices, pd.DataFrame) or close_prices.empty:
        raise ValueError("输入`close_prices`必须是非空的Pandas DataFrame。")
    if not isinstance(ff_factors, pd.DataFrame) or ff_factors.empty:
        raise ValueError("输入`ff_factors`必须是非空的Pandas DataFrame。")
    if not isinstance(window, int) or window <= 1: # 回归至少需要几个点
        raise ValueError("参数`window`必须是大于1的整数。")
    
    required_ff_cols = ['Mkt-RF', 'SMB', 'HML', 'RF']
    if not all(col in ff_factors.columns for col in required_ff_cols):
        raise ValueError(f"Fama-French因子数据必须包含以下列: {required_ff_cols}")

    print(f"开始计算RVOL因子，使用{window}天窗口...")
    
    # 1. 计算日收益率
    returns = close_prices.pct_change()

    # 2. 数据对齐与预处理
    # 合并收益率和FF因子，基于共同的日期索引
    common_idx = returns.index.intersection(ff_factors.index)
    if len(common_idx) < window:
        print(f"警告：共同交易日数量({len(common_idx)})少于窗口大小({window})，无法计算RVOL。")
        return pd.DataFrame(index=close_prices.index, columns=close_prices.columns, dtype=float)
        
    returns_aligned = returns.loc[common_idx]
    ff_factors_aligned = ff_factors.loc[common_idx]

    # 3. 计算超额收益率 (个股收益率 - 无风险利率)
    # 假设输入的RF是百分比形式，例如2%表示为2.0，需转换为小数0.02
    rf_daily_decimal = ff_factors_aligned['RF'] / 100.0
    excess_returns_df = returns_aligned.subtract(rf_daily_decimal, axis=0)
    
    # 准备Fama-French回归的自变量 (也转换为小数形式)
    ff_regressors_df = ff_factors_aligned[['Mkt-RF', 'SMB', 'HML']] / 100.0

    rvol_results = pd.DataFrame(index=excess_returns_df.index, columns=excess_returns_df.columns, dtype=float)

def calculate_rvol(close_prices: pd.DataFrame, ff_factors: pd.DataFrame, window: int = 60, 
                   use_parallel: bool = True, n_jobs: int = None) -> pd.DataFrame:
    """
    计算RVOL因子：特质波动率因子 (优化版本)

    Args:
        close_prices (pd.DataFrame): 收盘价数据，索引为日期，列为股票代码
        ff_factors (pd.DataFrame): Fama-French三因子数据，索引为日期，
                                   必须包含列: 'Mkt-RF', 'SMB', 'HML', 'RF' (假设为百分比形式)
        window (int): 滚动窗口大小，默认为60天
        use_parallel (bool): 是否使用并行计算，默认True
        n_jobs (int): 并行进程数，默认为CPU核心数

    Returns:
        pd.DataFrame: RVOL因子值，索引为日期，列为股票代码
    """
    
    # 数据验证
    if not isinstance(close_prices, pd.DataFrame) or close_prices.empty:
        raise ValueError("输入`close_prices`必须是非空的Pandas DataFrame。")
    if not isinstance(ff_factors, pd.DataFrame) or ff_factors.empty:
        raise ValueError("输入`ff_factors`必须是非空的Pandas DataFrame。")
    if not isinstance(window, int) or window <= 1:
        raise ValueError("参数`window`必须是大于1的整数。")
    
    required_ff_cols = ['Mkt-RF', 'SMB', 'HML', 'RF']
    if not all(col in ff_factors.columns for col in required_ff_cols):
        raise ValueError(f"Fama-French因子数据必须包含以下列: {required_ff_cols}")

    print(f"开始计算RVOL因子，使用{window}天窗口...")
    
    # 1. 数据预处理 - 一次性完成所有转换
    returns = close_prices.pct_change()
    common_idx = returns.index.intersection(ff_factors.index)
    
    if len(common_idx) < window:
        print(f"警告：共同交易日数量({len(common_idx)})少于窗口大小({window})，无法计算RVOL。")
        return pd.DataFrame(index=close_prices.index, columns=close_prices.columns, dtype=float)
        
    returns_aligned = returns.loc[common_idx]
    ff_factors_aligned = ff_factors.loc[common_idx]
    
    # 2. 预计算所有需要的数据
    rf_daily_decimal = ff_factors_aligned['RF'].values / 100.0
    excess_returns_df = returns_aligned.subtract(pd.Series(rf_daily_decimal, index=common_idx), axis=0)
    ff_regressors_np = ff_factors_aligned[['Mkt-RF', 'SMB', 'HML']].values / 100.0
    
    min_obs_reg = max(10, int(window * 0.7))
    total_stocks = len(excess_returns_df.columns)
    dates_len = len(excess_returns_df)
    
    print(f"总共需要处理 {total_stocks} 只股票的RVOL计算...")
    
    # 3. 并行计算或串行计算
    if use_parallel and total_stocks > 10:  # 股票数量较多时才使用并行
        if n_jobs is None:
            n_jobs = min(cpu_count(), total_stocks)
        
        # 准备并行计算的参数
        args_list = []
        for stock_idx, stock_code in enumerate(excess_returns_df.columns):
            stock_excess_returns_np = excess_returns_df[stock_code].values
            args_list.append((stock_idx, stock_code, stock_excess_returns_np, 
                            ff_regressors_np, window, min_obs_reg, dates_len))
        
        print(f"使用 {n_jobs} 个进程进行并行计算...")
        
        # 并行计算
        with Pool(n_jobs) as pool:
            results = pool.map(calculate_single_stock_rvol, args_list)
        
        # 整理结果
        rvol_results = pd.DataFrame(index=excess_returns_df.index, 
                                   columns=excess_returns_df.columns, dtype=float)
        for stock_code, rvol_values in results:
            rvol_results[stock_code] = rvol_values
    
    else:
        # 串行计算（优化版）
        rvol_results = pd.DataFrame(index=excess_returns_df.index, 
                                   columns=excess_returns_df.columns, dtype=float)
        
        for stock_idx, stock_code in enumerate(excess_returns_df.columns):
            if (stock_idx + 1) % 50 == 0 or stock_idx == total_stocks - 1:
                print(f"RVOL计算进度: {stock_idx + 1}/{total_stocks} ({(stock_idx + 1)/total_stocks*100:.1f}%)")
            
            stock_excess_returns_np = excess_returns_df[stock_code].values
            _, rvol_values = calculate_single_stock_rvol(
                (stock_idx, stock_code, stock_excess_returns_np, 
                 ff_regressors_np, window, min_obs_reg, dates_len)
            )
            rvol_results[stock_code] = rvol_values

    # 4. 年化处理
    rvol_annualized = rvol_results * np.sqrt(252)
    rvol_annualized = rvol_annualized.replace([np.inf, -np.inf], np.nan)
    
    print(f"RVOL因子计算完成，数据维度: {rvol_annualized.shape}")
    if rvol_annualized.notna().any().any():
        print(f"有效数据日期范围: {rvol_annualized.dropna(how='all').index.min()} 至 {rvol_annualized.dropna(how='all').index.max()}")
    else:
        print("警告：未能计算出任何有效的RVOL因子值。")

    return rvol_annualized


def get_factor_data(close_data: pd.DataFrame, ff_factors: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
    """
    因子计算的标准化接口

    Args:
        close_data (pd.DataFrame): 收盘价数据
        ff_factors (pd.DataFrame): Fama-French三因子数据
        **kwargs: 其他参数，可包含 'window'

    Returns:
        pd.DataFrame: 计算得到的RVOL因子数据
    """
    if ff_factors is None:
        raise ValueError("RVOL因子计算需要提供Fama-French三因子数据 (ff_factors)。")
        
    window = kwargs.get('window', 60) # 研报默认为60天
    return calculate_rvol(close_data, ff_factors, window=window)


if __name__ == "__main__":
    print("RVOL 因子模块测试")

    dates_rng = pd.date_range(start='2020-01-01', periods=100, freq='B')
    num_stocks = 3
    stock_names = [f'STOCK_{i+1}' for i in range(num_stocks)]

    np.random.seed(42)
    price_movements = np.random.randn(len(dates_rng), num_stocks) * 0.02 + 0.0005
    simulated_prices = 100 * (1 + price_movements).cumprod(axis=0)
    test_price_data = pd.DataFrame(simulated_prices, index=dates_rng, columns=stock_names)
    test_price_data = test_price_data.clip(lower=1.0)

    # 模拟Fama-French因子数据 (假设为百分比形式)
    ff_data = {
        'Mkt-RF': np.random.normal(0.03, 0.5, len(dates_rng)), # 假设日均0.03%，标准差0.5%
        'SMB': np.random.normal(0.01, 0.3, len(dates_rng)),
        'HML': np.random.normal(0.005, 0.2, len(dates_rng)),
        'RF': np.full(len(dates_rng), 2.0 / 252) # 假设年化无风险利率2%，转换为日度百分比
    }
    test_ff_factors = pd.DataFrame(ff_data, index=dates_rng)

    print("\n模拟价格数据 (前5行):")
    print(test_price_data.head())
    print("\n模拟FF因子数据 (前5行):")
    print(test_ff_factors.head())
    
    try:
        print("\n测试计算 RVOL (60天窗口)...")
        result_rvol = get_factor_data(test_price_data, ff_factors=test_ff_factors, window=60)
        print("RVOL 计算成功!")
        print(f"结果维度: {result_rvol.shape}")
        print("因子值 (最后5行):")
        print(result_rvol.tail())
        if result_rvol.notna().sum().sum() == 0:
            print("警告: RVOL结果全为NaN。")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
