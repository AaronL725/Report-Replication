import pandas as pd
import numpy as np
import os
import pickle # 用于缓存数据

# --- 获取项目路径配置 ---
# 获取当前脚本文件的绝对路径
CURRENT_FILE_PATH = os.path.abspath(__file__)
# 获取当前脚本所在的目录
CURRENT_DIR = os.path.dirname(CURRENT_FILE_PATH)
# 获取项目根目录（当前目录的上一级）
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# 设置数据目录（根目录下的Data/day目录）
INPUT_DATA_DIR = os.path.join(PROJECT_ROOT, 'Data', 'day')
# 设置缓存目录（在当前Module目录下创建cache子目录）
CACHE_DIR = os.path.join(CURRENT_DIR, 'cache')
# 缓存文件名模板
CACHE_FILENAME_TEMPLATE = 'processed_{data_type}.pkl'

# 需要加载和处理的OHLCV数据类型
DATA_TYPES = ['open', 'close', 'high', 'low', 'vol']

# Fama-French因子文件名和缓存时使用的数据类型名
FAMA_FRENCH_FILENAME = 'F-F_Research_Data_Factors_daily.csv'
FAMA_FRENCH_DATA_TYPE_NAME = 'ff_factors'

# 新增辅助数据文件名和缓存时使用的数据类型名
INDUSTRY_FILENAME = 'industry.csv'
INDUSTRY_DATA_TYPE_NAME = 'industry'

MARKET_CAP_FILENAME = 'market_cap.csv'
MARKET_CAP_DATA_TYPE_NAME = 'market_cap'

CSI300_CONSTITUENTS_FILENAME = 'csi300_constituents_daily.csv'
CSI300_CONSTITUENTS_DATA_TYPE_NAME = 'csi300_constituents'

CSI300_INDEX_FILENAME = 'csi300_index.csv'
CSI300_INDEX_DATA_TYPE_NAME = 'csi300_index_returns'


def load_single_ohlcv_csv(data_dir: str, data_type: str) -> pd.DataFrame | None:
    """
    加载单个股票行情CSV文件 (open, close, high, low, vol)，并进行基础的格式转换。
    Args:
        data_dir (str): CSV文件所在的目录。
        data_type (str): 数据类型 ('open', 'close', 'high', 'low', 'vol')。
    Returns:
        pd.DataFrame or None: 加载并初步处理后的DataFrame，如果文件不存在则返回None。
    """
    file_path = os.path.join(data_dir, f'{data_type}.csv')
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 未找到。")
        return None

    print(f"正在加载OHLCV文件: {file_path} ...")
    try:
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index) # 假设索引已经是 'YYYY-MM-DD HH:MM:SS' 格式
        df.columns = df.columns.str.strip() # 清理列名
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"文件 {file_path} 加载完成，数据类型已转换为数值型。")
        return df
    except Exception as e:
        print(f"加载或初步处理文件 {file_path} 时发生错误: {e}")
        return None

def clean_ohlcv_data(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """
    对已加载的股票行情DataFrame进行数据清洗。
    Args:
        df (pd.DataFrame): 原始数据DataFrame。
        data_type (str): 数据类型。
    Returns:
        pd.DataFrame: 清洗后的DataFrame。
    """
    print(f"开始OHLCV数据清洗: {data_type} ...")
    original_shape = df.shape

    if data_type in ['open', 'close', 'high', 'low']:
        df[df <= 0] = np.nan # 价格不能为负或零
        df.ffill(inplace=True) # 使用前向填充处理NaN
    elif data_type == 'vol':
        df[df < 0] = np.nan # 成交量不能为负
        df.fillna(0, inplace=True) # NaN成交量填充为0

    print(f"OHLCV数据清洗完成: {data_type}. 原始维度: {original_shape}, 清洗后维度: {df.shape}")
    return df

def align_ohlcv_dataframes(data_frames_dict: dict) -> dict:
    """
    对齐多个股票行情数据类型的DataFrame，使其具有相同的日期索引和股票代码列。
    Args:
        data_frames_dict (dict): 包含多个DataFrame的字典，键为数据类型。
    Returns:
        dict: 对齐后的DataFrame字典。
    """
    valid_dfs = [df for df in data_frames_dict.values() if df is not None and not df.empty]
    if not valid_dfs:
        print("错误：没有有效的OHLCV DataFrame进行对齐。")
        return {key: None for key in data_frames_dict.keys()}
    # ... (rest of the align_ohlcv_dataframes function remains the same)
    if len(valid_dfs) == 1:
        print("只有一个有效的OHLCV DataFrame，无需对齐，但将确保排序。")
        df_single = valid_dfs[0]
        df_type_single = [k for k, v in data_frames_dict.items() if v is df_single][0]
        return {df_type_single: df_single.sort_index(axis=0).sort_index(axis=1)}

    print("开始对齐所有加载的OHLCV数据帧...")
    common_index = valid_dfs[0].index
    for df in valid_dfs[1:]:
        common_index = common_index.intersection(df.index)
    common_index = common_index.sort_values()

    common_columns = valid_dfs[0].columns
    for df in valid_dfs[1:]:
        common_columns = common_columns.intersection(df.columns)
    common_columns = common_columns.sort_values()
    
    if common_index.empty or common_columns.empty:
        print("错误：无法找到共同的日期索引或股票代码列进行对齐OHLCV数据。")
        return {key: None for key in data_frames_dict.keys()}

    aligned_data = {}
    for data_type, df in data_frames_dict.items():
        if df is not None:
            aligned_df = df.reindex(index=common_index, columns=common_columns)
            aligned_data[data_type] = aligned_df
        else:
            aligned_data[data_type] = None
    
    print(f"OHLCV数据对齐完成。共同日期范围: {common_index.min()} 到 {common_index.max()}, 共同股票数量: {len(common_columns)}")
    return aligned_data


def load_and_process_fama_french(data_dir: str, filename: str) -> pd.DataFrame | None:
    """加载并处理Fama-French因子数据文件。"""
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        print(f"错误：Fama-French因子文件 {file_path} 未找到。")
        return None
    print(f"正在加载Fama-French因子文件: {file_path} ...")
    try:
        df = pd.read_csv(file_path, index_col=0, skipfooter=1, engine='python')
        df.columns = df.columns.str.strip()
        # 修复：设置时间为15:00:00以匹配股票行情数据
        df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d') + pd.Timedelta(hours=15)
        df.index.name = 'date'
        factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RF']
        for col in factor_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
            else:
                print(f"警告：Fama-French数据中缺少预期的列: {col}。")
        print(f"Fama-French因子文件 {file_path} 加载和初步处理完成。")
        return df
    except Exception as e:
        print(f"加载或处理Fama-French因子文件 {file_path} 时发生错误: {e}")
        return None

def load_general_stock_dataframe(data_dir: str, filename: str, data_name: str, value_dtype=None, fill_na_method='ffill') -> pd.DataFrame | None:
    """
    加载通用的股票相关DataFrame数据 (如市值、行业、成分股)。
    Args:
        data_dir (str): CSV文件所在的目录。
        filename (str): CSV文件名。
        data_name (str): 数据描述名称 (用于打印日志)。
        value_dtype : 目标数据类型 (例如 bool for constituents)。
        fill_na_method (str or None): 填充NaN值的方法 ('ffill', 'bfill', or None).
    Returns:
        pd.DataFrame or None: 处理后的DataFrame。
    """
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        print(f"错误：{data_name}文件 {file_path} 未找到。")
        return None

    print(f"正在加载 {data_name} 文件: {file_path} ...")
    try:
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index) # 假设索引是 'YYYY-MM-DD HH:MM:SS'
        df.columns = df.columns.str.strip() # 清理列名中的空格

        if value_dtype == 'numeric':
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        elif value_dtype == bool:
             # pd.read_csv 通常能正确解析 'True'/'False' 字符串为布尔值
             # 如果不是，可能需要显式转换 df.astype(bool) 或 df.replace({'True':True, 'False':False})
             pass # 假设read_csv已正确处理或后续步骤处理

        if fill_na_method == 'ffill':
            df.ffill(inplace=True)
        elif fill_na_method == 'bfill':
            df.bfill(inplace=True)
        
        print(f"{data_name} 文件 {file_path} 加载和初步处理完成。")
        return df
    except Exception as e:
        print(f"加载或处理 {data_name} 文件 {file_path} 时发生错误: {e}")
        return None

def load_index_series_data(data_dir: str, filename: str, data_name: str, fill_na_method='ffill') -> pd.Series | None:
    """
    加载指数数据 (如指数收益率)。
    Args:
        data_dir (str): CSV文件所在的目录。
        filename (str): CSV文件名。
        data_name (str): 数据描述名称。
        fill_na_method (str or None): 填充NaN值的方法。
    Returns:
        pd.Series or None: 处理后的Series。
    """
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        print(f"错误：{data_name}文件 {file_path} 未找到。")
        return None
    print(f"正在加载 {data_name} 文件: {file_path} ...")
    try:
        # 假设CSV是两列：第一列日期（索引），第二列是值，无头部
        series = pd.read_csv(file_path, index_col=0, header=None, names=['value']).squeeze("columns")
        series.index = pd.to_datetime(series.index) # 假设索引是 'YYYY-MM-DD HH:MM:SS'
        series = pd.to_numeric(series, errors='coerce')
        series.name = data_name # 给Series命名

        if fill_na_method == 'ffill':
            series.ffill(inplace=True)
        elif fill_na_method == 'bfill':
            series.bfill(inplace=True)

        print(f"{data_name} 文件 {file_path} 加载和初步处理完成。")
        return series
    except Exception as e:
        print(f"加载或处理 {data_name} 文件 {file_path} 时发生错误: {e}")
        return None


def save_data_to_cache(data_to_cache: dict, cache_dir: str):
    """将处理后的DataFrames/Series保存到缓存文件。"""
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
            print(f"缓存目录 {cache_dir} 已创建。")
        except Exception as e:
            print(f"创建缓存目录 {cache_dir} 失败: {e}。无法保存缓存。")
            return

    for data_type, data_item in data_to_cache.items(): # Changed df to data_item
        if data_item is not None and not data_item.empty:
            cache_file_path = os.path.join(cache_dir, CACHE_FILENAME_TEMPLATE.format(data_type=data_type))
            try:
                with open(cache_file_path, 'wb') as f:
                    pickle.dump(data_item, f)
                print(f"数据 {data_type} 已缓存到 {cache_file_path}")
            except Exception as e:
                print(f"缓存数据 {data_type} 到 {cache_file_path} 时发生错误: {e}")
        else:
            print(f"数据类型 {data_type} 为空或处理失败，不进行缓存。")

def run_data_loader():
    """主函数，执行数据加载、预处理、清洗和缓存。"""
    print("开始执行数据加载和预处理模块...")
    if not os.path.exists(CACHE_DIR):
        try:
            os.makedirs(CACHE_DIR)
            print(f"缓存目录 {CACHE_DIR} 已创建。")
        except Exception as e:
            print(f"创建缓存目录 {CACHE_DIR} 失败: {e}。")

    final_data_to_cache = {}

    # --- 1. 处理OHLCV股票行情数据 ---
    all_loaded_ohlcv_data = {}
    for data_type in DATA_TYPES:
        raw_df = load_single_ohlcv_csv(INPUT_DATA_DIR, data_type) # Renamed function
        if raw_df is not None:
            cleaned_df = clean_ohlcv_data(raw_df, data_type) # Renamed function
            all_loaded_ohlcv_data[data_type] = cleaned_df
        else:
            all_loaded_ohlcv_data[data_type] = None

    valid_ohlcv_data_for_alignment = {k: v for k, v in all_loaded_ohlcv_data.items() if v is not None}
    if not valid_ohlcv_data_for_alignment:
        print("警告：所有OHLCV数据文件均未能成功加载或清洗。")
    else:
        aligned_and_processed_ohlcv_data = align_ohlcv_dataframes(valid_ohlcv_data_for_alignment) # Renamed
        for dt_key, df_val in aligned_and_processed_ohlcv_data.items():
            if df_val is not None and not df_val.empty:
                 final_data_to_cache[dt_key] = df_val
        if not any(dt_key in final_data_to_cache for dt_key in DATA_TYPES):
             print("警告：OHLCV数据对齐后无有效数据可加入缓存列表。")

    # --- 2. 处理Fama-French因子数据 ---
    ff_df = load_and_process_fama_french(INPUT_DATA_DIR, FAMA_FRENCH_FILENAME)
    if ff_df is not None and not ff_df.empty:
        final_data_to_cache[FAMA_FRENCH_DATA_TYPE_NAME] = ff_df
    else:
        print(f"Fama-French因子数据 ({FAMA_FRENCH_DATA_TYPE_NAME}) 加载失败或为空。")

    # --- 3. 处理新增辅助数据 ---
    # 行业数据
    industry_df = load_general_stock_dataframe(INPUT_DATA_DIR, INDUSTRY_FILENAME, "行业", value_dtype=None, fill_na_method='ffill')
    if industry_df is not None and not industry_df.empty:
        final_data_to_cache[INDUSTRY_DATA_TYPE_NAME] = industry_df
    else:
        print(f"行业数据 ({INDUSTRY_DATA_TYPE_NAME}) 加载失败或为空。")

    # 市值数据
    market_cap_df = load_general_stock_dataframe(INPUT_DATA_DIR, MARKET_CAP_FILENAME, "市值", value_dtype='numeric', fill_na_method='ffill')
    if market_cap_df is not None and not market_cap_df.empty:
        final_data_to_cache[MARKET_CAP_DATA_TYPE_NAME] = market_cap_df
    else:
        print(f"市值数据 ({MARKET_CAP_DATA_TYPE_NAME}) 加载失败或为空。")

    # CSI300成分股数据
    csi300_const_df = load_general_stock_dataframe(INPUT_DATA_DIR, CSI300_CONSTITUENTS_FILENAME, "CSI300成分股", value_dtype=bool, fill_na_method='ffill')
    if csi300_const_df is not None and not csi300_const_df.empty:
        # 确保布尔类型正确
        try:
            csi300_const_df = csi300_const_df.astype(bool)
        except Exception as e:
            print(f"警告: 转换CSI300成分股数据为布尔型失败: {e}. 数据可能不是标准的True/False字符串。")
        final_data_to_cache[CSI300_CONSTITUENTS_DATA_TYPE_NAME] = csi300_const_df
    else:
        print(f"CSI300成分股数据 ({CSI300_CONSTITUENTS_DATA_TYPE_NAME}) 加载失败或为空。")
        
    # CSI300指数收益数据
    csi300_index_returns_series = load_index_series_data(INPUT_DATA_DIR, CSI300_INDEX_FILENAME, "CSI300指数收益", fill_na_method='ffill')
    if csi300_index_returns_series is not None and not csi300_index_returns_series.empty:
        final_data_to_cache[CSI300_INDEX_DATA_TYPE_NAME] = csi300_index_returns_series
    else:
        print(f"CSI300指数收益数据 ({CSI300_INDEX_DATA_TYPE_NAME}) 加载失败或为空。")

    # --- 4. 保存所有收集到的数据到缓存 ---
    if not final_data_to_cache:
        print("没有数据可供缓存。流程结束。")
    else:
        print(f"准备缓存以下数据类型: {list(final_data_to_cache.keys())}")
        save_data_to_cache(final_data_to_cache, CACHE_DIR)

    print("数据加载、预处理和缓存流程执行完毕。")

if __name__ == '__main__':
    run_data_loader()
