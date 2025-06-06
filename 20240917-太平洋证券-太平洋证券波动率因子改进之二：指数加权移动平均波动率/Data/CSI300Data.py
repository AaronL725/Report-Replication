#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沪深300成分股数据获取脚本 (CSI 300 Constituents Data)
功能：从Tushare获取沪深300指数成分股的历史变动数据，并保存为CSV文件。
用途：用于生成研报图表 "4.1 EWMAVOL因子沪深300内选股分年表现"
"""

import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time

# 加载Tushare Token
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
if not TUSHARE_TOKEN:
    print("未在.env文件中找到TUSHARE_TOKEN，请在Data文件夹下的.env文件中添加TUSHARE_TOKEN=你的token")
    exit()

# 配置参数
CSI300_INDEX_CODE = '399300.SZ'  # 沪深300指数代码
REPORT_START_DATE_STR = "20200901"
REPORT_END_DATE_STR = "20250430"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "day")
OUTPUT_FILENAME = "csi300_constituents_daily.csv"

# 初始化Tushare Pro
try:
    pro = ts.pro_api(TUSHARE_TOKEN)
    # 测试连接
    pro.trade_cal(exchange='', start_date='20200101', end_date='20200101')
    print("Tushare Pro API 初始化成功。")
except Exception as e:
    print(f"初始化 Tushare Pro API 时出错: {e}")
    exit()

def get_trade_dates(start_date_str, end_date_str):
    """获取指定区间的交易日"""
    try:
        df_cal = pro.trade_cal(exchange='', start_date=start_date_str, end_date=end_date_str)
        trade_dates = df_cal[df_cal['is_open'] == 1]['cal_date'].tolist()
        trade_dates.sort()
        print(f"获取到 {start_date_str} 和 {end_date_str} 之间的 {len(trade_dates)} 个交易日。")
        return trade_dates
    except Exception as e:
        print(f"获取交易日历时出错: {e}")
        return []

def get_csi300_constituents_for_period(start_date_str, end_date_str):
    """
    获取指定时间段内的沪深300成分股数据
    """
    try:
        print(f"正在获取 {start_date_str} 到 {end_date_str} 的沪深300成分股数据...")
        df_constituents = pro.index_weight(
            index_code=CSI300_INDEX_CODE, 
            start_date=start_date_str, 
            end_date=end_date_str
        )
        
        if df_constituents.empty:
            print(f"警告: 期间 {start_date_str}-{end_date_str} 未获取到沪深300成分股数据")
            return pd.DataFrame()
        
        print(f"获取到 {len(df_constituents)} 条成分股记录")
        return df_constituents
        
    except Exception as e:
        print(f"获取沪深300成分股数据时出错: {e}")
        return pd.DataFrame()

def get_all_csi300_constituents():
    """
    分批获取所有时间段的沪深300成分股数据
    """
    all_constituents = []
    
    # 按年分批获取数据，避免一次性获取过多数据导致超时
    start_year = int(REPORT_START_DATE_STR[:4])
    end_year = int(REPORT_END_DATE_STR[:4])
    
    for year in range(start_year, end_year + 1):
        year_start = f"{year}0101"
        year_end = f"{year}1231"
        
        # 调整起始和结束日期
        if year == start_year:
            year_start = REPORT_START_DATE_STR
        if year == end_year:
            year_end = REPORT_END_DATE_STR
            
        print(f"获取 {year} 年数据...")
        year_data = get_csi300_constituents_for_period(year_start, year_end)
        
        if not year_data.empty:
            all_constituents.append(year_data)
        
        # 添加延迟避免频率限制
        time.sleep(0.3)
    
    if all_constituents:
        result = pd.concat(all_constituents, ignore_index=True)
        print(f"总共获取到 {len(result)} 条成分股记录")
        return result
    else:
        print("警告: 未获取到任何沪深300成分股数据")
        return pd.DataFrame()

def create_constituents_matrix(df_constituents, trade_dates):
    """
    创建成分股布尔矩阵
    参数:
        df_constituents: 成分股历史数据 (包含trade_date和con_code列)
        trade_dates: 交易日列表
    返回:
        DataFrame: 布尔型矩阵，行为日期，列为股票代码
    """
    if df_constituents.empty:
        print("警告: 成分股数据为空，无法创建矩阵")
        return pd.DataFrame()
    
    print("正在创建成分股布尔矩阵...")
    
    # 获取所有股票代码
    all_stocks = sorted(df_constituents['con_code'].unique())
    print(f"发现 {len(all_stocks)} 个不同的成分股")
    
    # 创建日期索引（格式化为 YYYY-MM-DD 15:00:00）
    formatted_dates = []
    for date_str in trade_dates:
        date_obj = pd.to_datetime(date_str, format='%Y%m%d')
        formatted_dates.append(date_obj.strftime('%Y-%m-%d') + ' 15:00:00')
    
    # 初始化布尔矩阵
    constituents_matrix = pd.DataFrame(
        False, 
        index=pd.to_datetime(formatted_dates),
        columns=all_stocks
    )
    
    # 填充成分股数据
    print("正在填充成分股矩阵...")
    
    # 将trade_date转换为datetime格式以便比较
    df_constituents = df_constituents.copy()
    df_constituents['trade_date'] = pd.to_datetime(df_constituents['trade_date'], format='%Y%m%d')
    
    # 为每个交易日确定成分股
    for i, trade_date_str in enumerate(trade_dates):
        if i % 500 == 0:  # 每500个交易日打印一次进度
            print(f"处理进度: {i+1}/{len(trade_dates)}")
            
        trade_date = pd.to_datetime(trade_date_str, format='%Y%m%d')
        formatted_date = pd.to_datetime(trade_date.strftime('%Y-%m-%d') + ' 15:00:00')
        
        # 找到该日期或之前最近的成分股数据
        # 使用前向填充的方式：如果某日没有成分股变动数据，则使用之前最近的成分股列表
        relevant_data = df_constituents[df_constituents['trade_date'] <= trade_date]
        
        if not relevant_data.empty:
            # 获取最新的成分股列表
            latest_date = relevant_data['trade_date'].max()
            current_constituents = relevant_data[
                relevant_data['trade_date'] == latest_date
            ]['con_code'].unique()
            
            # 在矩阵中标记这些股票为True
            for stock in current_constituents:
                if stock in constituents_matrix.columns:
                    constituents_matrix.loc[formatted_date, stock] = True
    
    print(f"成分股矩阵创建完成，维度: {constituents_matrix.shape}")
    
    # 计算统计信息
    daily_constituent_counts = constituents_matrix.sum(axis=1)
    print(f"平均每日成分股数量: {daily_constituent_counts.mean():.1f}")
    print(f"成分股数量范围: {daily_constituent_counts.min()} - {daily_constituent_counts.max()}")
    
    return constituents_matrix

def save_constituents_data(constituents_matrix, output_path):
    """
    保存成分股数据到CSV文件
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 将布尔值转换为1/0以节省空间和便于读取
        constituents_matrix_int = constituents_matrix.astype(int)
        
        # 保存到CSV，格式与DataFetcher.py保持一致
        constituents_matrix_int.to_csv(output_path, na_rep='0')
        print(f"成功保存沪深300成分股数据到: {output_path}")
        
        # 显示文件信息
        file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
        print(f"文件大小: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"保存成分股数据时出错: {e}")
        return False

def load_existing_constituents_data(file_path):
    """
    加载已存在的成分股数据
    """
    try:
        if os.path.exists(file_path):
            print(f"找到现有成分股数据文件: {file_path}")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            print(f"加载的数据维度: {df.shape}")
            return df
        else:
            print("未找到现有成分股数据文件")
            return None
    except Exception as e:
        print(f"加载现有成分股数据时出错: {e}")
        return None

# --- 主脚本逻辑 ---
if __name__ == "__main__":
    print("=" * 60)
    print("开始获取沪深300成分股数据...")
    print("数据用途: 生成研报图表 '4.1 EWMAVOL因子沪深300内选股分年表现'")
    print("=" * 60)
    
    # 创建输出目录
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"输出目录 '{OUTPUT_DIR}' 已准备好。")
    except Exception as e:
        print(f"创建输出目录 '{OUTPUT_DIR}' 时出错: {e}")
        exit()
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # 检查是否已存在数据文件
    existing_data = load_existing_constituents_data(output_path)
    if existing_data is not None:
        user_input = input("发现现有数据文件，是否重新获取？(y/N): ").lower().strip()
        if user_input != 'y':
            print("使用现有数据文件，退出程序。")
            exit()
    
    # 1. 获取交易日
    print("\n1. 获取交易日历...")
    trade_dates = get_trade_dates(REPORT_START_DATE_STR, REPORT_END_DATE_STR)
    if not trade_dates:
        print("未找到交易日，退出程序。")
        exit()
    
    # 2. 获取沪深300成分股历史数据
    print("\n2. 获取沪深300成分股历史数据...")
    constituents_data = get_all_csi300_constituents()
    
    if constituents_data.empty:
        print("警告: 未找到沪深300成分股数据，无法生成'EWMAVOL因子沪深300内选股分年表现'图表。")
        print("建议检查Tushare权限或网络连接。")
        exit()
    
    # 3. 创建成分股布尔矩阵
    print("\n3. 创建成分股布尔矩阵...")
    constituents_matrix = create_constituents_matrix(constituents_data, trade_dates)
    
    if constituents_matrix.empty:
        print("无法创建成分股矩阵，退出程序。")
        exit()
    
    # 4. 保存数据
    print("\n4. 保存成分股数据...")
    success = save_constituents_data(constituents_matrix, output_path)
    
    if success:
        print("\n" + "=" * 60)
        print("沪深300成分股数据获取完成！")
        print(f"数据文件: {output_path}")
        print(f"数据期间: {REPORT_START_DATE_STR} - {REPORT_END_DATE_STR}")
        print(f"数据维度: {constituents_matrix.shape}")
        print("数据格式: 布尔型矩阵 (1/0)，行为日期，列为股票代码")
        print("=" * 60)
    else:
        print("数据保存失败，请检查错误信息。")