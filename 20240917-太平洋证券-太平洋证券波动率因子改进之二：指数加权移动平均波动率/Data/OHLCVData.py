'''
数据获取脚本
功能：从Tushare获取指定时间段内的股票日度行情数据，并将其保存为CSV文件。
'''

import tushare as ts
import pandas as pd
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
REPORT_START_DATE_STR = "20201231"
REPORT_END_DATE_STR = "20250430"
DATA_FETCH_START_DATE_STR = "20200901"
DATA_FETCH_END_DATE_STR = REPORT_END_DATE_STR
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "day")
FIELDS_TO_EXTRACT = {
    'open': 'open.csv',
    'high': 'high.csv',
    'low': 'low.csv',
    'close': 'close.csv',
    'vol': 'vol.csv',
}

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

def fetch_daily_data_for_date(trade_date_str):
    """
    获取指定交易日所有股票的日度行情数据。
    """
    try:
        df_daily = pro.daily(trade_date=trade_date_str,
                             fields='ts_code,trade_date,open,high,low,close,vol')
        return df_daily
    except Exception as e:
        print(f"获取 {trade_date_str} 的日度数据时出错: {e}。正在重试一次...")
        try:
            import time
            time.sleep(0.3)
            df_daily = pro.daily(trade_date=trade_date_str,
                                 fields='ts_code,trade_date,open,high,low,close,vol')
            return df_daily
        except Exception as e_retry:
            print(f"为 {trade_date_str} 重试失败: {e_retry}。跳过此日期。")
            return pd.DataFrame()


# --- 主脚本逻辑 ---
if __name__ == "__main__":
    print("开始数据获取流程...")

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

    # 2. 获取每个交易日所有股票的日度数据
    all_daily_data_list = []
    print(f"正在获取 {len(all_fetch_trade_dates)} 个交易日的日度数据...")
    for i, trade_date in enumerate(all_fetch_trade_dates):
        print(f"处理日期 {i+1}/{len(all_fetch_trade_dates)}: {trade_date}")
        daily_data_df = fetch_daily_data_for_date(trade_date)
        if not daily_data_df.empty:
            all_daily_data_list.append(daily_data_df)

    if not all_daily_data_list:
        print("没有获取到数据。正在退出。")
        exit()

    print("合并所有数据...")
    full_data_df = pd.concat(all_daily_data_list, ignore_index=True)
    print(f"合并后的数据维度: {full_data_df.shape}")
    if full_data_df.empty:
        print("合并后的数据为空。正在退出。")
        exit()

    # 3. 获取所有唯一股票代码的排序列表
    all_stock_tickers = sorted(full_data_df['ts_code'].unique())
    print(f"找到 {len(all_stock_tickers)} 个唯一的股票代码。")

    # 4. 将 'trade_date' 格式化为 'YYYY-MM-DD 15:00:00'
    full_data_df['trade_date'] = full_data_df['trade_date'].astype(str)
    unique_trade_dates_yyyymmdd = sorted(full_data_df['trade_date'].unique())
    formatted_dates = pd.to_datetime(unique_trade_dates_yyyymmdd, format='%Y%m%d')
    formatted_date_index_str = [d.strftime('%Y-%m-%d') + ' 15:00:00' for d in formatted_dates]
    
    # 5. 为每个所需字段透视并保存数据
    for field, filename in FIELDS_TO_EXTRACT.items():
        print(f"保存 {filename} ...")
        try:
            pivoted_df = full_data_df.pivot_table(index='trade_date', columns='ts_code', values=field)
            pivoted_df = pivoted_df.reindex(columns=all_stock_tickers)
            pivoted_df = pivoted_df.reindex(index=unique_trade_dates_yyyymmdd)
            pivoted_df.index = pd.Series(formatted_date_index_str, name='date_time_index')
            pivoted_df.index.name = None
            output_path = os.path.join(OUTPUT_DIR, filename)
            pivoted_df.to_csv(output_path, na_rep='')
            print(f"成功保存 {output_path}")
        except Exception as e:
            print(f"处理或保存 {filename} 时出错: {e}")
    print("数据导出结束。")
