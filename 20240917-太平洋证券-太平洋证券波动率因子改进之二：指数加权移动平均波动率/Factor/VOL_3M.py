# filepath: VOL_3M.py
"""
VOL_3M因子计算模块
功能：计算基础波动率因子
定义：过去N天日收益率的标准差（研报中默认为60天，对应VOL_3M中的“3M”近似）
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def calculate_vol_nm(close_prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    计算VOL_NM因子（例如VOL_3M对应window=60）：过去N天日收益率的标准差

    Args:
        close_prices (pd.DataFrame): 收盘价数据，索引为日期，列为股票代码
        window (int): 滚动窗口大小，例如60天代表过去约3个月

    Returns:
        pd.DataFrame: 因子值，索引为日期，列为股票代码
    """
    
    # 数据验证
    if not isinstance(close_prices, pd.DataFrame) or close_prices.empty:
        raise ValueError("输入`close_prices`必须是非空的Pandas DataFrame。")
    if not isinstance(window, int) or window <= 1:
        raise ValueError("参数`window`必须是大于1的整数。")
    
    print(f"开始计算VOL_{window}D因子，使用{window}天窗口...")
    
    # 计算日收益率 (当日收盘价/前一日收盘价 - 1)
    # 使用 fill_method=None (默认) 来确保 pct_change 在数据开始处产生NaN
    returns = close_prices.pct_change() 
    
    # 计算滚动标准差
    # min_periods 参数确保在窗口期数据不足时，若数据点少于min_periods则结果为NaN
    # 研报中通常要求一定比例的有效数据，例如80%
    min_obs = int(window * 0.8) 
    if min_obs < 2: # 标准差至少需要2个点
        min_obs = 2
        
    vol_factor = returns.rolling(window=window, min_periods=min_obs).std()
    
    # 年化处理（假设一年252个交易日）
    # 波动率是标准差，年化时乘以 sqrt(周期数)
    vol_factor_annualized = vol_factor * np.sqrt(252)
    
    # 清理可能因计算产生的无效值 (例如，如果某窗口内returns全为NaN，std可能为NaN或0)
    # replace Inf/-Inf just in case, though std() on numbers should not produce them.
    vol_factor_annualized = vol_factor_annualized.replace([np.inf, -np.inf], np.nan)
    
    print(f"VOL_{window}D因子计算完成，数据维度: {vol_factor_annualized.shape}")
    if vol_factor_annualized.notna().any().any():
        print(f"有效数据日期范围: {vol_factor_annualized.dropna(how='all').index.min()} 至 {vol_factor_annualized.dropna(how='all').index.max()}")
    else:
        print("警告：未能计算出任何有效的因子值。")
        
    return vol_factor_annualized


def get_factor_data(close_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    因子计算的标准化接口

    Args:
        close_data (pd.DataFrame): 收盘价数据
        **kwargs: 其他参数，可包含 'window' (例如60代表VOL_3M)

    Returns:
        pd.DataFrame: 计算得到的因子数据
    """
    # 研报中VOL_3M通常指过去60个交易日的波动率
    window = kwargs.get('window', 60) 
    return calculate_vol_nm(close_data, window=window)


if __name__ == "__main__":
    print("VOL_NM (例如 VOL_3M) 因子模块测试")
    
    # 创建模拟测试数据
    dates_rng = pd.date_range(start='2020-01-01', periods=100, freq='B') # 'B' 表示工作日
    num_stocks = 3
    stock_names = [f'STOCK_{i+1}' for i in range(num_stocks)]
    
    # 生成随机价格数据
    np.random.seed(42)
    price_movements = np.random.randn(len(dates_rng), num_stocks) * 0.02 + 0.0005 # 模拟每日小幅波动
    simulated_prices = 100 * (1 + price_movements).cumprod(axis=0)
    
    test_price_data = pd.DataFrame(simulated_prices, index=dates_rng, columns=stock_names)
    test_price_data = test_price_data.clip(lower=1.0) # 确保价格为正

    print("\n模拟价格数据 (前5行):")
    print(test_price_data.head())

    try:
        # 测试计算 VOL_3M (60天窗口)
        print("\n测试计算 VOL_3M (60天窗口)...")
        result_vol_3m = get_factor_data(test_price_data, window=60)
        print("VOL_3M 计算成功!")
        print(f"结果维度: {result_vol_3m.shape}")
        print("因子值 (最后5行):")
        print(result_vol_3m.tail())
        if result_vol_3m.notna().sum().sum() == 0:
            print("警告: VOL_3M结果全为NaN，请检查输入数据和窗口期。")

        # 测试计算 VOL_1M (20天窗口)
        print("\n测试计算 VOL_1M (20天窗口)...")
        result_vol_1m = get_factor_data(test_price_data, window=20)
        print("VOL_1M 计算成功!")
        print(f"结果维度: {result_vol_1m.shape}")
        print("因子值 (最后5行):")
        print(result_vol_1m.tail())
        if result_vol_1m.notna().sum().sum() == 0:
            print("警告: VOL_1M结果全为NaN，请检查输入数据和窗口期。")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
