'''
行业分类数据获取脚本
功能：从Tushare获取指定时间段内的股票行业分类数据，并将其保存为CSV文件。
用途：用于因子预处理中的行业中性化步骤。
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
# 行业分类数据字段配置
FIELDS_TO_EXTRACT = {
    'industry': 'industry.csv',  # 行业分类代码
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

def get_all_stock_codes():
    """
    获取所有股票代码列表
    """
    try:
        # 获取股票基本信息
        df_basic = pro.stock_basic(exchange='', list_status='L', 
                                  fields='ts_code,symbol,name,area,industry,list_date')
        stock_codes = df_basic['ts_code'].tolist()
        print(f"获取到 {len(stock_codes)} 个股票代码。")
        return df_basic
    except Exception as e:
        print(f"获取股票基本信息时出错: {e}")
        return pd.DataFrame()

def get_industry_classification():
    """
    获取申万行业分类数据
    """
    try:
        # 获取申万一级行业分类
        df_industry = pro.index_classify(level='L1', src='SW2021')
        print(f"获取到 {len(df_industry)} 个申万一级行业分类。")
        return df_industry
    except Exception as e:
        print(f"获取行业分类数据时出错: {e}")
        return pd.DataFrame()

def get_stock_industry_mapping():
    """
    获取股票与行业的映射关系
    """
    try:
        # 获取申万行业成分股
        industry_df = get_industry_classification()
        stock_industry_map = {}
        
        for _, row in industry_df.iterrows():
            index_code = row['index_code']
            try:
                # 获取该行业的成分股
                members_df = pro.index_member(index_code=index_code)
                if not members_df.empty:
                    for _, member_row in members_df.iterrows():
                        stock_code = member_row['con_code']
                        stock_industry_map[stock_code] = index_code
                        
                import time
                time.sleep(0.3)
            except Exception as e:
                print(f"获取行业 {index_code} 成分股时出错: {e}")
                continue
                
        print(f"获取到 {len(stock_industry_map)} 个股票的行业映射关系。")
        return stock_industry_map
    except Exception as e:
        print(f"获取股票行业映射时出错: {e}")
        return {}

def create_industry_dataframe(stock_industry_map, trade_dates, all_stock_codes):
    """
    创建行业分类数据的DataFrame
    """
    # 将交易日转换为日期格式
    date_index = pd.to_datetime(trade_dates, format='%Y%m%d')
    formatted_date_index = [d.strftime('%Y-%m-%d') + ' 15:00:00' for d in date_index]
    
    # 创建空的DataFrame
    industry_df = pd.DataFrame(index=formatted_date_index, columns=all_stock_codes)
    
    # 填充行业数据（行业分类通常不会频繁变动，所以每个股票在所有日期都使用相同的行业代码）
    for stock_code in all_stock_codes:
        if stock_code in stock_industry_map:
            industry_df[stock_code] = stock_industry_map[stock_code]
        else:
            # 如果没有找到行业分类，可以设置为默认值或留空
            industry_df[stock_code] = None
            
    return industry_df


# --- 主脚本逻辑 ---
if __name__ == "__main__":
    print("开始行业分类数据获取流程...")

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

    # 2. 获取所有股票代码
    stock_basic_df = get_all_stock_codes()
    if stock_basic_df.empty:
        print("未获取到股票基本信息。正在退出。")
        exit()
    
    all_stock_codes = sorted(stock_basic_df['ts_code'].tolist())
    print(f"找到 {len(all_stock_codes)} 个股票代码。")

    # 3. 获取股票行业映射关系
    print("正在获取股票行业分类映射关系...")
    stock_industry_map = get_stock_industry_mapping()
    if not stock_industry_map:
        print("警告: 未获取到行业映射关系，将使用股票基本信息中的行业数据。")
        # 使用股票基本信息中的行业字段作为备选
        stock_industry_map = {}
        for _, row in stock_basic_df.iterrows():
            stock_industry_map[row['ts_code']] = row['industry']

    # 4. 创建行业分类数据DataFrame
    print("创建行业分类数据...")
    industry_df = create_industry_dataframe(stock_industry_map, all_fetch_trade_dates, all_stock_codes)
    
    if industry_df.empty:
        print("创建的行业数据为空。正在退出。")
        exit()

    print(f"行业数据维度: {industry_df.shape}")

    # 5. 保存行业分类数据
    for field, filename in FIELDS_TO_EXTRACT.items():
        print(f"保存 {filename} ...")
        try:
            output_path = os.path.join(OUTPUT_DIR, filename)
            industry_df.to_csv(output_path, na_rep='')
            print(f"成功保存 {output_path}")
            
            # 显示一些统计信息
            unique_industries = industry_df.stack().dropna().unique()
            print(f"共包含 {len(unique_industries)} 个不同的行业分类")
            print(f"行业分类示例: {list(unique_industries[:5])}")
            
        except Exception as e:
            print(f"处理或保存 {filename} 时出错: {e}")
    
    print("行业分类数据导出结束。")