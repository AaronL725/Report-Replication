# filepath: EWMAVOL.py
"""
EWMAVOL因子计算模块
功能：计算指数加权移动平均波动率因子
定义：使用EWMA模型对过去N天（研报默认为60天）日收益率赋予不同权重，计算波动率。
      σ²ₜ = (1-λ)(rₜ-uₜ)² + λσ²ₜ₋₁
      研报指明 λ=0.9, 历史期 L=60 天。
      uₜ 在研报中定义为 "t日的所有样本收益率均值"，此处解释为与滚动计算一致的个股自身60日滚动收益率均值。
"""

import pandas as pd
import numpy as np
from scipy.signal import lfilter, lfiltic
import warnings
warnings.filterwarnings('ignore')


def calculate_ewmavol(close_prices: pd.DataFrame, window: int = 60, lambda_decay: float = 0.9) -> pd.DataFrame:
    """
    计算EWMAVOL因子：指数加权移动平均波动率 (使用 lfilter 优化)

    Args:
        close_prices (pd.DataFrame): 收盘价数据，索引为日期，列为股票代码
        window (int): 用于计算滚动均值uₜ的窗口大小，默认为60天
        lambda_decay (float): EWMA的衰减因子 (λ)，默认为0.9

    Returns:
        pd.DataFrame: EWMAVOL因子值，索引为日期，列为股票代码
    """
    
    if not isinstance(close_prices, pd.DataFrame) or close_prices.empty:
        raise ValueError("输入`close_prices`必须是非空的Pandas DataFrame。")
    if not isinstance(window, int) or window <= 1:
        raise ValueError("参数`window`必须是大于1的整数。")
    if not (0 < lambda_decay < 1):
        raise ValueError("衰减因子`lambda_decay`必须在 (0, 1) 区间内。")
        
    print(f"开始计算EWMAVOL因子 (lfilter优化)，u_t窗口={window}天，衰减因子λ={lambda_decay}...")
    
    returns = close_prices.pct_change()
    
    # 1. 计算 u_t (个股收益率的滚动均值)
    min_obs_mean = int(window * 0.8)
    if min_obs_mean < 1: min_obs_mean = 1
    rolling_mean_returns = returns.rolling(window=window, min_periods=min_obs_mean).mean()

    # 2. 计算 (r_t - u_t)^2
    squared_deviations = (returns - rolling_mean_returns).pow(2)

    ewmavol_df = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    # EWMA 滤波器系数: y[n] - λ*y[n-1] = (1-λ)*x[n]
    # a[0]*y[n] + a[1]*y[n-1] + ... = b[0]*x[n] + b[1]*x[n-1] + ...
    # 这里 y[n] 是 sigma_sq_curr, y[n-1] 是 sigma_sq_prev, x[n] 是 current_sq_dev_val
    # sigma_sq_curr - lambda_decay * sigma_sq_prev = (1 - lambda_decay) * current_sq_dev_val
    # 所以 a = [1, -lambda_decay], b = [1 - lambda_decay]
    b_coeffs = [1 - lambda_decay]
    a_coeffs = [1, -lambda_decay]

    for stock_col in returns.columns:
        stock_sq_dev_series = squared_deviations[stock_col].dropna() # 输入信号 x[n]
        
        if len(stock_sq_dev_series) < 1:
            continue

        # 初始化 sigma_sq_prev (即 y[-1] for lfiltic)
        # 使用 (r_t-u_t)^2 序列中第一个有效值对应日期的之前`window`期收益率的方差来初始化。
        first_valid_sq_dev_date = stock_sq_dev_series.index[0]
        try:
            loc_of_first_valid_date = returns.index.get_loc(first_valid_sq_dev_date)
        except KeyError:
            continue
            
        start_loc_for_initial_var = max(0, loc_of_first_valid_date - window) # 注意这里是-window，不是-window+1
        initial_returns_for_var_calc = returns[stock_col].iloc[start_loc_for_initial_var:loc_of_first_valid_date].dropna() # 取第一个有效sq_dev之前的收益

        if len(initial_returns_for_var_calc) >= 2: # 方差至少需要2个点
            sigma_sq_initial_y_minus_1 = initial_returns_for_var_calc.var()
        elif not stock_sq_dev_series.empty:
            # 如果历史收益不足，用第一个有效的(r-u)^2作为y[0]的近似，那么y[-1]可以设为它或一个默认值
            # 或者，更简单地，用第一个(r-u)^2来直接启动，lfilter的zi可以设为0，然后修正第一个输出值
            sigma_sq_initial_y_minus_1 = stock_sq_dev_series.iloc[0] 
        else:
            sigma_sq_initial_y_minus_1 = 1e-8 

        if pd.isna(sigma_sq_initial_y_minus_1) or sigma_sq_initial_y_minus_1 <= 1e-8:
            sigma_sq_initial_y_minus_1 = 1e-8
            
        # 计算 lfilter 的初始状态 zi
        # zi = y[-1] * (-a[1]) for first order filter if a=[1, a1]
        # Here a[1] = -lambda_decay, so -a[1] = lambda_decay
        # zi = [sigma_sq_initial_y_minus_1 * lambda_decay] # This is the state for y[-1]
        # Alternatively, use lfiltic to compute zi based on past y and x values
        # lfiltic(b, a, y_past, x_past=None)
        # Here, y_past would be [sigma_sq_initial_y_minus_1]
        # x_past is not directly available for the period before stock_sq_dev_series starts.
        # A common approach for EWMA is to initialize the first EWMA value itself.
        # Let's use the direct zi calculation based on y[-1]
        
        zi = [sigma_sq_initial_y_minus_1 * lambda_decay] # This is the state related to y[-1] for the filter
                                                        # It's y[-1] * a_coeffs[1] if a_coeffs[0]=1, but signs matter.
                                                        # For y[n] = (1-L)x[n] + L y[n-1],
                                                        # zi should be such that the first output y[0] is correct.
                                                        # y[0] = (1-L)x[0] + L y[-1].
                                                        # lfilter default zi is 0.
                                                        # It's often easier to run lfilter and then adjust the first point,
                                                        # or initialize y[0] and then run lfilter on x[1:].
                                                        # For simplicity, let's use the definition of zi as state y[-1]*(-a[1])
        
        # If we consider the state of the filter to be y[-1] (the previous output),
        # then zi = lfiltic(b, a, y=[sigma_sq_initial_y_minus_1]) should work.
        # The state vector zi for lfilter contains the N-1 previous values of the output
        # and M-1 previous values of the input. For a 1st order filter (N=2, M=1), zi is a scalar.
        # zi = [y_prev_output * (-a_coeffs[1])] = [sigma_sq_initial_y_minus_1 * lambda_decay]
        # Let's test this. If it's off, the first value can be manually set.
        
        # An alternative to complex zi:
        # 1. Calculate first sigma_sq_0 manually:
        #    sigma_sq_0 = (1-lambda_decay)*stock_sq_dev_series.iloc[0] + lambda_decay*sigma_sq_initial_y_minus_1
        # 2. Run lfilter on stock_sq_dev_series.iloc[1:] with zi derived from sigma_sq_0
        # This is more robust.

        if stock_sq_dev_series.empty:
            continue
            
        # Manual calculation for the first point
        current_ewma_var = (1 - lambda_decay) * stock_sq_dev_series.iloc[0] + \
                           lambda_decay * sigma_sq_initial_y_minus_1
        ewmavol_df.loc[stock_sq_dev_series.index[0], stock_col] = np.sqrt(max(current_ewma_var, 1e-10))

        # For subsequent points, use lfilter if there are more points
        if len(stock_sq_dev_series) > 1:
            # The state for lfilter should be based on the *actual* previous output (current_ewma_var)
            zi_for_lfilter = [current_ewma_var * lambda_decay] # state = y_previous_actual * coeff_for_y_previous
            
            # Apply lfilter to the rest of the series x[1:]
            remaining_sq_dev = stock_sq_dev_series.iloc[1:].to_numpy()
            if len(remaining_sq_dev) > 0:
                filtered_sigma_sq_remaining = lfilter(b_coeffs, a_coeffs, remaining_sq_dev, zi=zi_for_lfilter)[0]
                
                valid_sqrt_sigma_sq_remaining = np.sqrt(np.maximum(filtered_sigma_sq_remaining, 1e-10))
                ewmavol_df.loc[stock_sq_dev_series.index[1:], stock_col] = valid_sqrt_sigma_sq_remaining
    
    # 年化处理
    ewmavol_df_annualized = ewmavol_df * np.sqrt(252)
    ewmavol_df_annualized = ewmavol_df_annualized.replace([np.inf, -np.inf], np.nan)
    
    print(f"EWMAVOL因子计算完成，数据维度: {ewmavol_df_annualized.shape}")
    if ewmavol_df_annualized.notna().any().any():
        print(f"有效数据日期范围: {ewmavol_df_annualized.dropna(how='all').index.min()} 至 {ewmavol_df_annualized.dropna(how='all').index.max()}")
    else:
        print("警告：未能计算出任何有效的EWMAVOL因子值。")
            
    return ewmavol_df_annualized


def get_factor_data(close_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    window = kwargs.get('window', 60) 
    lambda_decay = kwargs.get('lambda_decay', 0.9) 
    return calculate_ewmavol(close_data, window=window, lambda_decay=lambda_decay)


if __name__ == "__main__":
    print("EWMAVOL 因子模块测试 (lfilter优化)")
    
    dates_rng = pd.date_range(start='2020-01-01', periods=100, freq='B')
    num_stocks = 3
    stock_names = [f'STOCK_{i+1}' for i in range(num_stocks)]
    
    np.random.seed(42)
    price_data_dict = {}
    for stock in stock_names:
        s0 = np.random.uniform(80, 120)
        dt = 1/252
        mu_annual = np.random.uniform(0.05, 0.20)
        sigma_annual = np.random.uniform(0.15, 0.40)
        price_path = [s0]
        for _ in range(1, len(dates_rng)):
            drift = (mu_annual - 0.5 * sigma_annual**2) * dt
            diffusion = sigma_annual * np.sqrt(dt) * np.random.normal(0,1)
            price_path.append(price_path[-1] * np.exp(drift + diffusion))
        price_data_dict[stock] = price_path
        
    test_price_data = pd.DataFrame(price_data_dict, index=dates_rng)
    test_price_data = test_price_data.clip(lower=0.1)

    print("\n模拟价格数据 (前5行):")
    print(test_price_data.head())
    
    try:
        test_window_ewma = 20
        test_lambda_ewma = 0.94
        print(f"\n测试计算 EWMAVOL (u_t窗口={test_window_ewma}天, λ={test_lambda_ewma})...")
        
        result_ewmavol = get_factor_data(test_price_data, window=test_window_ewma, lambda_decay=test_lambda_ewma)
        print("EWMAVOL 计算成功!")
        print(f"结果维度: {result_ewmavol.shape}")
        print("因子值 (最后5行):")
        print(result_ewmavol.tail())
        if result_ewmavol.notna().sum().sum() == 0 :
             print("警告: EWMAVOL结果全为NaN。")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
