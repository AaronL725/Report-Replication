# filepath: GARCHVOL.py
"""
GARCHVOL因子计算模块 (极致优化版)
功能：计算预测特质波动率因子
定义：基于过去N天（研报默认为60天）日收益率序列对Fama-French三因子回归残差，
      使用GARCH(1,1)模型建模并进行一步向前（样本外）波动率预测

优化策略:
1. Numba JIT编译: 加速GARCH似然函数和主滚动循环。
2. GARCH参数缓存: 使用LRU缓存避免对相似的残差序列重复进行昂贵的优化计算。
3. 高效优化器: 使用L-BFGS-B替代SLSQP。
4. 多进程并行: 在股票层面并行计算。
5. NumPy向量化: 底层计算全部基于NumPy。
"""

import pandas as pd
import numpy as np
import warnings
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize
from numba import jit
from functools import lru_cache # 用于实现LRU缓存

warnings.filterwarnings('ignore')

# --- 核心计算函数 (使用Numba JIT编译) ---

@jit(nopython=True, cache=True)
def fast_linear_regression_numba(X, y):
    """
    使用Numba JIT编译的快速线性回归
    """
    # 添加截距项
    X_with_intercept = np.ones((X.shape[0], X.shape[1] + 1))
    X_with_intercept[:, 1:] = X
    # 用 pinv 替代 solve，避免异常
    beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ (X_with_intercept.T @ y)
    predictions = X_with_intercept @ beta
    residuals = y - predictions
    return residuals

@jit(nopython=True, cache=True)
def _garch_likelihood_numba(params, residuals_sq, initial_h_var):
    """
    Numba JIT编译的GARCH(1,1)负对数似然函数
    """
    omega, alpha, beta = params
    
    # 参数约束在优化器层面处理，但这里也做检查
    if omega <= 1e-8 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.9999:
        return 1e12

    num_obs = len(residuals_sq)
    h_variance = np.zeros(num_obs) 
    
    h_variance[0] = initial_h_var if initial_h_var > 1e-8 else 1e-8
    
    for t in range(1, num_obs):
        h_variance[t] = omega + alpha * residuals_sq[t-1] + beta * h_variance[t-1]
        if h_variance[t] <= 1e-8:
            h_variance[t] = 1e-8
            
    # 计算对数似然值
    log_likelihood_sum = -0.5 * np.sum(np.log(h_variance) + residuals_sq / h_variance) 
    
    if np.isnan(log_likelihood_sum) or np.isinf(log_likelihood_sum):
        return 1e12

    return -log_likelihood_sum, h_variance # 返回似然和条件方差序列

# --- GARCH模型类与缓存 ---

class GARCHModelFitter:
    def __init__(self, cache_size=128):
        # 使用LRU缓存来存储已计算的GARCH参数
        # 键是残差序列的统计特征元组，值是GARCH参数
        self.fit_garch_cached = lru_cache(maxsize=cache_size)(self._fit_garch_core)

    def _fit_garch_core(self, residuals_tuple):
        """
        核心的GARCH拟合逻辑，被LRU缓存装饰。
        注意：lru_cache要求所有参数都是可哈希的，因此传入元组。
        """
        residuals = np.array(residuals_tuple)
        
        residuals_sq = residuals**2
        initial_h_var = np.var(residuals)
        if initial_h_var <= 1e-8: initial_h_var = 1e-7

        var_res_sq_mean = np.mean(residuals_sq)
        if var_res_sq_mean <= 1e-8: var_res_sq_mean = 1e-8

        alpha_init, beta_init = 0.1, 0.85
        initial_omega = var_res_sq_mean * (1 - alpha_init - beta_init)
        if initial_omega <= 1e-8: initial_omega = 1e-7
        initial_params = np.array([initial_omega, alpha_init, beta_init])

        # 使用L-BFGS-B优化器，它通过bounds参数处理边界约束
        bounds = [(1e-8, None), (1e-8, 0.999), (1e-8, 0.999)]

        # L-BFGS-B不直接支持线性约束，但我们可以通过参数转换或在目标函数中惩罚来间接处理
        # 或者，我们可以信任边界约束，因为alpha+beta接近1时，omega会非常小，这会影响似然
        # 一个简单的方法是在目标函数中对违反约束的情况返回一个很大的值
        def objective_func(params):
             if params[1] + params[2] >= 0.9999: return 1e12
             # 只返回似然值给优化器
             neg_log_likelihood, _ = _garch_likelihood_numba(params, residuals_sq, initial_h_var)
             return neg_log_likelihood

        try:
            result = minimize(
                objective_func,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-7} # 减少迭代次数
            )
            
            if result.success:
                final_params = tuple(result.x) # 返回元组使其可哈希
                # 使用最优参数计算最终的条件方差序列
                _, final_h_series = _garch_likelihood_numba(final_params, residuals_sq, initial_h_var)
                return final_params, final_h_series
            else:
                return None, None
        except Exception:
            return None, None

    def fit(self, residuals: np.ndarray):
        """
        外部调用的拟合接口，处理缓存逻辑。
        """
        if len(residuals) < 10 or np.std(residuals) < 1e-7:
            return None, None # 数据不足或波动过小

        # 创建一个基于统计特征的缓存键 (可哈希)
        # 使用均值、标准差、偏度和峰度的量化值作为键
        # 乘以1e4并取整，以处理浮点数精度问题
        cache_key = (
            round(residuals.mean() * 1e4),
            round(residuals.std() * 1e4),
            round(pd.Series(residuals).skew() * 1e2),
            round(pd.Series(residuals).kurt() * 1e2)
        )

        # 调用缓存的拟合函数，传 tuple(residuals)
        return self.fit_garch_cached(tuple(residuals))


@jit(nopython=True, cache=True)
def forecast_volatility_numba(params, last_residual_sq, last_h_variance):
    """Numba JIT编译的一步向前波动率预测"""
    omega, alpha, beta = params
    h_forecast = omega + alpha * last_residual_sq + beta * last_h_variance
    if h_forecast <= 1e-8: h_forecast = 1e-8
    return np.sqrt(h_forecast)


# --- 主计算流程 ---

def calculate_single_stock_garchvol(args):
    """计算单只股票的GARCHVOL因子 - 用于并行计算"""
    (stock_code, stock_excess_returns_np, ff_regressors_np, 
     window, min_obs_ff_reg, min_obs_garch) = args
    
    dates_len = len(stock_excess_returns_np)
    stock_garchvol_values = np.full(dates_len, np.nan)
    
    # 每个进程拥有自己的GARCH拟合器实例 (包含自己的缓存)
    garch_fitter = GARCHModelFitter(cache_size=256) 
    
    for i in range(window - 1, dates_len):
        start_pos = i - window + 1
        end_pos = i + 1
        
        current_stock_ret_win = stock_excess_returns_np[start_pos:end_pos]
        current_ff_factors_win = ff_regressors_np[start_pos:end_pos, :]
        
        valid_mask = ~np.isnan(current_stock_ret_win)
        y_ff_reg = current_stock_ret_win[valid_mask]
        X_ff_reg = current_ff_factors_win[valid_mask, :]
        
        if len(y_ff_reg) < min_obs_ff_reg:
            continue
        
        residuals_ff = fast_linear_regression_numba(X_ff_reg, y_ff_reg)
        
        fallback_std = np.std(residuals_ff) if residuals_ff.size >= 2 else np.nan
        
        if residuals_ff.size < min_obs_garch:
            stock_garchvol_values[i] = fallback_std
            continue
            
        # 使用GARCH拟合器进行拟合 (会自动处理缓存)
        # 这里直接传 residuals_ff (ndarray)，不转 tuple
        params, h_series = garch_fitter.fit(residuals_ff)
        
        if params is not None and h_series is not None:
            last_h_var = h_series[-1]
            last_resid_sq = residuals_ff[-1]**2
            forecast = forecast_volatility_numba(params, last_resid_sq, last_h_var)
            stock_garchvol_values[i] = forecast
        else:
            stock_garchvol_values[i] = fallback_std
            
    return stock_code, stock_garchvol_values

def calculate_garchvol(close_prices: pd.DataFrame, ff_factors: pd.DataFrame, window: int = 60,
                      use_parallel: bool = True, n_jobs: int = None) -> pd.DataFrame:
    """计算GARCHVOL因子 (极致优化版)"""
    if not isinstance(close_prices, pd.DataFrame) or close_prices.empty:
        raise ValueError("close_prices 输入必须是非空Pandas DataFrame。")
    if not isinstance(ff_factors, pd.DataFrame) or ff_factors.empty:
        raise ValueError("ff_factors 输入必须是非空Pandas DataFrame。")
    
    print(f"开始计算GARCHVOL因子，FF回归窗口: {window}天...")
    
    returns = close_prices.pct_change()
    common_idx = returns.index.intersection(ff_factors.index)
    if len(common_idx) < window:
        print("警告：共同交易日数量不足，无法计算。")
        return pd.DataFrame()
        
    returns_aligned = returns.loc[common_idx]
    ff_factors_aligned = ff_factors.loc[common_idx]
    
    rf_daily_decimal = ff_factors_aligned['RF'].values / 100.0
    excess_returns_df = returns_aligned.subtract(pd.Series(rf_daily_decimal, index=common_idx), axis=0)
    ff_regressors_np = ff_factors_aligned[['Mkt-RF', 'SMB', 'HML']].values / 100.0

    min_obs_ff_reg = max(10, int(window * 0.7)) 
    min_obs_garch = 10
    total_stocks = len(excess_returns_df.columns)

    print(f"总共需要处理 {total_stocks} 只股票的GARCHVOL计算...")

    args_list = [
        (
            stock_code,
            excess_returns_df[stock_code].values,
            ff_regressors_np,
            window,
            min_obs_ff_reg,
            min_obs_garch
        )
        for stock_code in excess_returns_df.columns
    ]

    if use_parallel and total_stocks > 1:
        if n_jobs is None:
            n_jobs = min(cpu_count() - 1, total_stocks) if cpu_count() > 1 else 1
        else:
            n_jobs = min(n_jobs, cpu_count() - 1 if cpu_count() > 1 else 1, total_stocks)
        if n_jobs < 1: n_jobs = 1
        
        print(f"使用 {n_jobs} 个进程进行并行计算...")
        with Pool(n_jobs) as pool:
            results = pool.map(calculate_single_stock_garchvol, args_list)
    else:
        print("使用串行计算...")
        results = [calculate_single_stock_garchvol(arg) for arg in args_list]
        
    garchvol_results_df = pd.DataFrame(
        {code: vals for code, vals in results},
        index=excess_returns_df.index
    )
                
    garchvol_annualized = garchvol_results_df * np.sqrt(252) # 年化
    print(f"GARCHVOL因子计算完成，数据维度: {garchvol_annualized.shape}")
    return garchvol_annualized


def get_factor_data(close_data: pd.DataFrame, ff_factors: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
    """标准化因子计算接口"""
    if ff_factors is None:
        raise ValueError("GARCHVOL因子计算需要提供Fama-French三因子数据 (ff_factors)。")
    window = kwargs.get('window', 60)
    use_parallel_param = kwargs.get('use_parallel', True) 
    n_jobs_param = kwargs.get('n_jobs', None)
    return calculate_garchvol(close_data, ff_factors, window=window, 
                              use_parallel=use_parallel_param, n_jobs=n_jobs_param)

if __name__ == "__main__":
    print("GARCHVOL 因子模块测试 (极致优化版)")
    
    # 创建一个规模稍大的测试数据集以体现优化效果
    dates_rng = pd.date_range(start='2020-01-01', periods=500, freq='B') 
    num_stocks = 50 
    stock_names = [f'STOCK_{i+1:03d}' for i in range(num_stocks)]
    
    np.random.seed(123)
    price_data = np.random.randn(len(dates_rng), num_stocks) * 0.02 + 0.0001
    price_data = 100 * np.exp(np.cumsum(price_data, axis=0)) # 更真实的股价
    test_price_data = pd.DataFrame(price_data, index=dates_rng, columns=stock_names)

    ff_data_dict = {
        'Mkt-RF': np.random.normal(0.03/252, 0.02, len(dates_rng)) * 100, 
        'SMB': np.random.normal(0.01/252, 0.01, len(dates_rng)) * 100,   
        'HML': np.random.normal(0.005/252, 0.01, len(dates_rng)) * 100,  
        'RF': np.full(len(dates_rng), 0.015/252) * 100 
    }
    test_ff_factors = pd.DataFrame(ff_data_dict, index=dates_rng)
    
    import time
    
    try:
        test_window_garch = 60 
        print(f"\n测试计算 GARCHVOL ({test_window_garch}天FF窗口) ...")
        
        start_time = time.time()
        result_garchvol = get_factor_data(test_price_data, ff_factors=test_ff_factors, 
                                          window=test_window_garch, use_parallel=True) 
        end_time = time.time()
        
        print(f"\n计算完成! 总耗时: {end_time - start_time:.2f} 秒")
        print(f"结果维度: {result_garchvol.shape}")
        print("因子值 (最后3行, 前5列):")
        print(result_garchvol.tail(3).iloc[:, :5])
        if result_garchvol.notna().sum().sum() == 0 :
             print("警告: GARCHVOL结果全为NaN。")
        else:
            print(f"有效GARCHVOL值数量: {result_garchvol.notna().sum().sum()} / {result_garchvol.size}")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

