# filepath: RANKVOL.py
"""
RANKVOL因子计算模块
功能：计算分位数波动率因子
定义：过去N天内（研报默认为60天），每日个股收益率在全市场股票中的排序分位数序列的标准差
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def calculate_rankvol(close_prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    计算RANKVOL因子：分位数波动率因子

    Args:
        close_prices (pd.DataFrame): 收盘价数据，索引为日期，列为股票代码
        window (int): 滚动窗口大小，默认为60天

    Returns:
        pd.DataFrame: RANKVOL因子值，索引为日期，列为股票代码
    """
    
    # 数据验证
    if not isinstance(close_prices, pd.DataFrame) or close_prices.empty:
        raise ValueError("输入`close_prices`必须是非空的Pandas DataFrame。")
    if not isinstance(window, int) or window <= 1:
        raise ValueError("参数`window`必须是大于1的整数。")
        
    print(f"开始计算RANKVOL因子，使用{window}天窗口...")
    
    # 1. 计算日收益率
    returns = close_prices.pct_change()
    if returns.empty:
        print("警告：计算得到的日收益率数据为空。")
        return pd.DataFrame(index=close_prices.index, columns=close_prices.columns, dtype=float)

    # 2. 计算每日收益率的排序分位数 (0到1之间)
    #    研报定义："计算全部股票日收益率的排序分位数"
    #    (rank - 1) / (N_valid - 1) 确保结果在 [0, 1] 区间
    #    Pandas rank(pct=True) 通常是 rank / N_valid，结果在 (0, 1] 区间 (近似)
    #    为精确复现 (rank-1)/(N-1)，我们手动计算。
    
    print("计算每日收益率的截面排序分位数...")
    
    def get_rank_percentile(daily_returns_series: pd.Series) -> pd.Series:
        """辅助函数：计算单日截面收益率的排序分位数"""
        valid_returns = daily_returns_series.dropna()
        if len(valid_returns) < 2: # 如果有效股票数少于2，无法计算 (N-1) 分母
            # 返回一个与输入Series索引相同，但全为NaN的Series
            return pd.Series(np.nan, index=daily_returns_series.index, dtype=float) 
            
        ranks = valid_returns.rank(method='average', ascending=True) # 平均排名处理相同值
        # 计算 (rank - 1) / (N_valid - 1)
        num_valid = len(valid_returns)
        percentiles_on_valid = (ranks - 1) / (num_valid - 1)
        
        # 将计算得到的分位数放回原始Series的对应位置
        result_series = pd.Series(np.nan, index=daily_returns_series.index, dtype=float)
        result_series.loc[valid_returns.index] = percentiles_on_valid
        return result_series

    # 使用 apply 在 axis=1 (逐行，即逐日) 上应用此函数
    # result_type='broadcast' 确保结果的列与原始returns的列一致
    rank_percentiles_df = returns.apply(get_rank_percentile, axis=1, result_type='broadcast')

    if rank_percentiles_df.empty:
        print("警告：计算得到的排序分位数数据为空。")
        return pd.DataFrame(index=close_prices.index, columns=close_prices.columns, dtype=float)
        
    # 3. 计算分位数时间序列的滚动标准差
    print("计算分位数序列的滚动标准差...")
    min_obs_rank = int(window * 0.8)
    if min_obs_rank < 2:
        min_obs_rank = 2
        
    rankvol_factor = rank_percentiles_df.rolling(window=window, min_periods=min_obs_rank).std()
    
    # 无需年化，因为分位数本身是0-1的值，其标准差也是一个相对波动指标
    # 研报中通常直接使用此标准差作为因子值

    rankvol_factor = rankvol_factor.replace([np.inf, -np.inf], np.nan)
    
    print(f"RANKVOL因子计算完成，数据维度: {rankvol_factor.shape}")
    if rankvol_factor.notna().any().any():
        print(f"有效数据日期范围: {rankvol_factor.dropna(how='all').index.min()} 至 {rankvol_factor.dropna(how='all').index.max()}")
    else:
        print("警告：未能计算出任何有效的RANKVOL因子值。")
        
    return rankvol_factor


def get_factor_data(close_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    因子计算的标准化接口

    Args:
        close_data (pd.DataFrame): 收盘价数据
        **kwargs: 其他参数，可包含 'window'

    Returns:
        pd.DataFrame: 计算得到的RANKVOL因子数据
    """
    window = kwargs.get('window', 60) # 研报默认为60天
    return calculate_rankvol(close_data, window=window)


if __name__ == "__main__":
    print("RANKVOL 因子模块测试")
    
    dates_rng = pd.date_range(start='2020-01-01', periods=100, freq='B')
    num_stocks = 20 # RANKVOL需要较多股票进行截面排序才有意义
    stock_names = [f'STOCK_{i+1}' for i in range(num_stocks)]
    
    np.random.seed(42)
    price_movements = np.random.randn(len(dates_rng), num_stocks) * 0.02 + 0.0005
    simulated_prices = 100 * (1 + price_movements).cumprod(axis=0)
    
    test_price_data = pd.DataFrame(simulated_prices, index=dates_rng, columns=stock_names)
    test_price_data = test_price_data.clip(lower=1.0)

    # 引入一些NaN值模拟真实数据
    for col in test_price_data.columns:
        nan_indices = np.random.choice(test_price_data.index, size=int(0.1 * len(test_price_data)), replace=False)
        test_price_data.loc[nan_indices, col] = np.nan
        
    print("\n模拟价格数据 (部分):")
    print(test_price_data.head())

    try:
        print("\n测试计算 RANKVOL (60天窗口)...")
        result_rankvol = get_factor_data(test_price_data, window=60)
        print("RANKVOL 计算成功!")
        print(f"结果维度: {result_rankvol.shape}")
        print("因子值 (最后5行，前3列):")
        print(result_rankvol.tail().iloc[:, :3])
        if result_rankvol.notna().sum().sum() == 0 :
             print("警告: RANKVOL结果全为NaN。")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
