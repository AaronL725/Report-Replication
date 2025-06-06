'''
基准指数数据获取脚本 (Benchmark Index Data)
功能：从Tushare获取沪深300指数的历史价格数据，并计算每日收益率，保存为CSV文件。
用途：用于计算"超额收益"、"信息比率"、"相对胜率"和"超额最大回撤"等相对于市场基准的性能指标。
'''

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# 加载Tushare Token
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
if not TUSHARE_TOKEN:
    print("未在.env文件中找到TUSHARE_TOKEN，请在Data文件夹下的.env文件中添加TUSHARE_TOKEN=你的token")
    exit()

# 配置参数
CSI300_INDEX_CODE = '399300.SZ'  # 沪深300指数代码
REPORT_START_DATE_STR = "20201231"
REPORT_END_DATE_STR = "20250430"
DATA_FETCH_START_DATE_STR = "20200901"
DATA_FETCH_END_DATE_STR = REPORT_END_DATE_STR
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "day")
OUTPUT_FILENAME = "csi300_index.csv"

# 初始化Tushare Pro
try:
    pro = ts.pro_api(TUSHARE_TOKEN)
    pro.trade_cal(exchange='', start_date='20200101', end_date='20200101')
    print("Tushare Pro API 初始化成功。")
except Exception as e:
    print(f"初始化 Tushare Pro API 时出错: {e}")
    exit()

# 获取指定区间的交易日
def get_trade_dates(start_date_str, end_date_str):
    try:
        df_cal = pro.trade_cal(exchange='', start_date=start_date_str, end_date=end_date_str)
        trade_dates = df_cal[df_cal['is_open'] == 1]['cal_date'].tolist()
        trade_dates.sort()
        print(f"获取到 {start_date_str} 和 {end_date_str} 之间的 {len(trade_dates)} 个交易日。")
        return trade_dates
    except Exception as e:
        print(f"获取交易日历时出错: {e}")
        return []

def fetch_csi300_index_data():
    """
    获取沪深300指数的历史价格数据。
    """
    try:
        print(f"正在获取沪深300指数数据: {DATA_FETCH_START_DATE_STR} 到 {DATA_FETCH_END_DATE_STR}")
        df_index = pro.index_daily(
            ts_code=CSI300_INDEX_CODE,
            start_date=DATA_FETCH_START_DATE_STR,
            end_date=DATA_FETCH_END_DATE_STR,
            fields='ts_code,trade_date,close'
        )
        print(f"获取到 {len(df_index)} 条沪深300指数数据")
        return df_index
    except Exception as e:
        print(f"获取沪深300指数数据时出错: {e}。正在重试一次...")
        try:
            import time
            time.sleep(0.3)
            df_index = pro.index_daily(
                ts_code=CSI300_INDEX_CODE,
                start_date=DATA_FETCH_START_DATE_STR,
                end_date=DATA_FETCH_END_DATE_STR,
                fields='ts_code,trade_date,close'
            )
            return df_index
        except Exception as e_retry:
            print(f"重试失败: {e_retry}。")
            return pd.DataFrame()

def calculate_daily_returns(df_index, trade_dates):
    """
    计算指数的每日收益率并格式化为与其他数据一致的格式
    """
    if df_index.empty:
        print("指数数据为空，无法计算收益率")
        return pd.Series(dtype=float)
    
    print("正在计算沪深300指数每日收益率...")
    
    # 确保数据按日期排序
    df_index = df_index.copy()
    df_index['trade_date'] = df_index['trade_date'].astype(str)
    df_index = df_index.sort_values('trade_date')
    
    # 计算每日收益率
    df_index['close'] = pd.to_numeric(df_index['close'])
    df_index['returns'] = df_index['close'].pct_change()
    
    # 删除第一个NaN值（第一天没有前一天的数据）
    df_index = df_index.dropna()
    
    # 将trade_date格式化为与其他数据一致的格式
    formatted_dates = []
    for date_str in df_index['trade_date']:
        date_obj = pd.to_datetime(date_str, format='%Y%m%d')
        formatted_dates.append(date_obj.strftime('%Y-%m-%d') + ' 15:00:00')
    
    # 创建收益率Series
    returns_series = pd.Series(
        df_index['returns'].values,
        index=pd.to_datetime(formatted_dates),
        name='CSI300_Returns'
    )
    
    print(f"计算完成，共 {len(returns_series)} 个交易日的收益率")
    print(f"收益率统计:")
    print(f"  平均日收益率: {returns_series.mean():.6f}")
    print(f"  收益率标准差: {returns_series.std():.6f}")
    print(f"  最大单日收益: {returns_series.max():.6f}")
    print(f"  最小单日收益: {returns_series.min():.6f}")
    
    return returns_series


# --- 主脚本逻辑 ---
if __name__ == "__main__":
    print("开始获取沪深300基准指数数据...")

    # 创建输出目录 (如果不存在)
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"输出目录 '{OUTPUT_DIR}' 已准备好。")
    except Exception as e:
        print(f"创建输出目录 '{OUTPUT_DIR}' 时出错: {e}")
        exit()

    # 1. 获取数据提取周期的所有交易日
    all_fetch_trade_dates = get_trade_dates(DATA_FETCH_START_DATE_STR, DATA_FETCH_END_DATE_STR)
    if not all_fetch_trade_dates:
        print("未找到交易日。正在退出。")
        exit()

    # 2. 获取沪深300指数历史数据
    index_data = fetch_csi300_index_data()
    if index_data.empty:
        print("没有获取到指数数据。正在退出。")
        exit()

    # 3. 计算每日收益率
    returns_series = calculate_daily_returns(index_data, all_fetch_trade_dates)
    if returns_series.empty:
        print("无法计算收益率。正在退出。")
        exit()

    # 4. 保存数据
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    try:
        # 将Series转换为DataFrame以保存到CSV
        returns_df = pd.DataFrame(returns_series)
        returns_df.to_csv(output_path)
        print(f"成功保存沪深300基准指数数据到: {output_path}")
        
        # 显示文件信息
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"文件大小: {file_size:.2f} KB")
        print(f"数据维度: {returns_df.shape}")
        
    except Exception as e:
        print(f"保存数据时出错: {e}")
        
    print("基准指数数据获取完成。")