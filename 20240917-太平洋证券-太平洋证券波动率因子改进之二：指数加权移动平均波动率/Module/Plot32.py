# filepath: Plot.py
"""
绘图模块
功能：读取log.log文件数据，绘制与研报格式一致的因子收益表现和因子IC表现图表，并保存到Result文件夹
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors # 用于颜色处理
from typing import Dict, List

# --- 全局设置 ---
# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 优先使用微软雅黑
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"设置中文字体失败 (尝试了Microsoft YaHei, SimHei): {e}。可能显示为方块。")
    plt.rcParams['font.sans-serif'] = ['sans-serif'] # 降级到通用无衬线体
    plt.rcParams['axes.unicode_minus'] = False


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(CURRENT_DIR, "log.log")
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) # 假设Plot.py在项目的子目录中
if "Factor" in PROJECT_ROOT or "Plot" in PROJECT_ROOT : # 修正项目根目录判断
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
RESULT_DIR = os.path.join(PROJECT_ROOT, "Result")


# 研报中的因子顺序 (表头从上到下)
REPORT_FACTOR_ORDER = ['VOL_3M', 'RANKVOL', 'RVOL', 'GARCHVOL', 'EWMAVOL']

# 研报颜色参考 (近似值)
COLOR_REPORT_HEADER_BG = '#FDEADA' # 研报表头背景色 (一种浅橙黄色)
COLOR_REPORT_CELL_BG = '#FFFFFF'   # 研报数据单元格背景色 (白色)
COLOR_REPORT_FACTOR_BG = '#F2F2F2' # 研报因子名称列背景色 (浅灰色)
COLOR_REPORT_RED_TEXT = '#FF0000'    # 红色文字
COLOR_REPORT_DARK_YELLOW_TEXT = '#B8860B' # 深黄色文字 (用于RankICIR)
COLOR_REPORT_BORDER = '#BFBFBF' # 研报表格边框颜色 (稍深一点的灰色)


# --- 数据加载与计算 ---

def load_log_data() -> Dict:
    """加载log.log文件中的数据"""
    try:
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("log.log 文件加载成功。")
        return data
    except FileNotFoundError:
        print(f"错误: log.log 文件未找到于 {LOG_FILE_PATH}")
        return None
    except json.JSONDecodeError:
        print(f"错误: log.log 文件格式无效，无法解析JSON。")
        return None
    except Exception as e:
        print(f"读取log.log文件时发生未知错误: {e}")
        return None

def calculate_annualized_return(period_returns: np.ndarray, periods_per_year: int = 252) -> float:
    """计算年化收益率 (几何平均)"""
    if len(period_returns) == 0:
        return 0.0
    valid_returns = period_returns[~np.isnan(period_returns)]
    if len(valid_returns) == 0:
        return 0.0
    
    product_of_returns = np.prod(1 + valid_returns)
    if product_of_returns <= 0 and np.any(valid_returns <= -1.0): 
        if -1.0 in valid_returns:
            return -1.0 
        print(f"警告: 收益率序列计算得到的累积乘积为 {product_of_returns}，年化收益可能不准确。")
        if product_of_returns <=0 : return np.nan

    num_periods = len(valid_returns)
    if num_periods == 0: return 0.0 
    annualized = (product_of_returns ** (periods_per_year / num_periods)) - 1
    return annualized

def calculate_sharpe_ratio(period_returns: np.ndarray, risk_free_rate_annual: float = 0.0, periods_per_year: int = 252) -> float:
    """计算夏普比率"""
    if len(period_returns) == 0:
        return 0.0
    valid_returns = period_returns[~np.isnan(period_returns)]
    if len(valid_returns) < 2: 
        return 0.0
        
    risk_free_rate_period = (1 + risk_free_rate_annual)**(1/periods_per_year) - 1
    excess_returns = valid_returns - risk_free_rate_period
    
    mean_excess_return = np.mean(excess_returns)
    std_dev_excess_return = np.std(excess_returns) 
    
    if std_dev_excess_return == 0: 
        return 0.0 if mean_excess_return == 0 else np.inf * np.sign(mean_excess_return)
        
    sharpe = (mean_excess_return / std_dev_excess_return) * np.sqrt(periods_per_year)
    return sharpe

def calculate_max_drawdown(period_returns: np.ndarray) -> float:
    """计算最大回撤"""
    if len(period_returns) == 0:
        return 0.0
    valid_returns = period_returns[~np.isnan(period_returns)]
    if len(valid_returns) == 0:
        return 0.0
        
    cumulative_net_value = np.cumprod(1 + valid_returns)
    historical_peaks = np.maximum.accumulate(np.insert(cumulative_net_value, 0, 1.0)) 
    
    if len(historical_peaks) > 1 and len(cumulative_net_value) > 0:
        running_max = np.maximum.accumulate(cumulative_net_value) 
        drawdowns = (running_max - cumulative_net_value) / running_max
        drawdowns[running_max == 0] = 0 
    else:
        drawdowns = np.array([0.0])

    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    return max_dd

def get_factor_performance_metrics(log_data: Dict) -> pd.DataFrame:
    if not log_data or "因子表现数据" not in log_data:
        print("错误: log_data 结构不符合预期，缺少'因子表现数据'。")
        return pd.DataFrame()

    factor_data_dict = log_data["因子表现数据"]
    benchmark_period_returns_series_raw = log_data.get("基准指数收益率序列", [])
    
    benchmark_period_returns_series = np.array(benchmark_period_returns_series_raw)
    benchmark_period_returns_series = benchmark_period_returns_series[~np.isnan(benchmark_period_returns_series)]

    all_metrics = {}
    periods_per_year = 252 

    for factor_name in REPORT_FACTOR_ORDER: 
        if factor_name not in factor_data_dict:
            print(f"警告: 在log数据中未找到因子 {factor_name} 的表现数据。")
            nan_metrics = {key: np.nan for key in [
                "多头_年化收益", "多头_超额收益", "多头_夏普比率", "多头_超额回撤", 
                "多头_相对胜率", "多头_信息比率", "多空_年化收益", "多空_超额收益", 
                "多空_夏普比率", "多空_最大回撤", "多空_胜率", "IC均值", "IC标准差", 
                "年化ICIR", "RankIC均值", "RankIC标准差", "年化RankICIR", "RankIC>0占比"
            ]}
            all_metrics[factor_name] = nan_metrics
            continue

        factor_specific_data = factor_data_dict[factor_name]
        metrics = {}
        
        grouped_returns = factor_specific_data.get("各分组收益率", {})
        long_portfolio_returns_list = grouped_returns.get("第一组", grouped_returns.get("Group_1_Period_Return", []))
        
        long_portfolio_period_returns = np.array(long_portfolio_returns_list)
        long_portfolio_period_returns = long_portfolio_period_returns[~np.isnan(long_portfolio_period_returns)]

        long_short_returns_list = factor_specific_data.get("多空组合收益率序列", [])
        long_short_portfolio_period_returns = np.array(long_short_returns_list)
        long_short_portfolio_period_returns = long_short_portfolio_period_returns[~np.isnan(long_short_portfolio_period_returns)]
            
        ic_series = np.array(factor_specific_data.get("IC值序列", []))
        ic_series = ic_series[~np.isnan(ic_series)]
        rank_ic_series = np.array(factor_specific_data.get("RankIC值序列", []))
        rank_ic_series = rank_ic_series[~np.isnan(rank_ic_series)]

        if len(long_portfolio_period_returns) > 0:
            metrics["多头_年化收益"] = calculate_annualized_return(long_portfolio_period_returns, periods_per_year)
            
            current_benchmark_returns = benchmark_period_returns_series
            current_long_portfolio_returns = long_portfolio_period_returns
            
            min_len = min(len(current_long_portfolio_returns), len(current_benchmark_returns))
            if min_len > 0:
                current_long_portfolio_returns_aligned = current_long_portfolio_returns[:min_len]
                current_benchmark_returns_aligned = current_benchmark_returns[:min_len]

                excess_long_returns = current_long_portfolio_returns_aligned - current_benchmark_returns_aligned
                metrics["多头_超额收益"] = calculate_annualized_return(excess_long_returns, periods_per_year)
                annual_mean_excess = np.mean(excess_long_returns) * periods_per_year
                annual_std_excess = np.std(excess_long_returns) * np.sqrt(periods_per_year)
                metrics["多头_信息比率"] = annual_mean_excess / annual_std_excess if annual_std_excess != 0 else 0.0
                
                long_net_value = np.cumprod(1 + current_long_portfolio_returns_aligned)
                bench_net_value = np.cumprod(1 + current_benchmark_returns_aligned)
                if len(long_net_value) > 0 and len(bench_net_value) > 0 and np.all(bench_net_value != 0):
                    excess_net_value_over_bench = long_net_value / bench_net_value
                    metrics["多头_超额回撤"] = calculate_max_drawdown(excess_net_value_over_bench - 1)
                else:
                    metrics["多头_超额回撤"] = np.nan
                metrics["多头_相对胜率"] = np.mean(current_long_portfolio_returns_aligned > current_benchmark_returns_aligned) if min_len > 0 else 0.0
            else: 
                metrics["多头_超额收益"] = np.nan
                metrics["多头_信息比率"] = np.nan
                metrics["多头_超额回撤"] = np.nan
                metrics["多头_相对胜率"] = np.nan

            metrics["多头_夏普比率"] = calculate_sharpe_ratio(long_portfolio_period_returns, 0.0, periods_per_year)
        else: 
            metrics["多头_年化收益"] = metrics["多头_超额收益"] = metrics["多头_夏普比率"] = \
            metrics["多头_超额回撤"] = metrics["多头_相对胜率"] = metrics["多头_信息比率"] = np.nan

        if len(long_short_portfolio_period_returns) > 0:
            metrics["多空_年化收益"] = calculate_annualized_return(long_short_portfolio_period_returns, periods_per_year)
            metrics["多空_超额收益"] = metrics["多空_年化收益"] 
            metrics["多空_夏普比率"] = calculate_sharpe_ratio(long_short_portfolio_period_returns, 0.0, periods_per_year)
            metrics["多空_最大回撤"] = calculate_max_drawdown(long_short_portfolio_period_returns)
            metrics["多空_胜率"] = np.mean(long_short_portfolio_period_returns > 0) if len(long_short_portfolio_period_returns) > 0 else 0.0
        else: 
            metrics["多空_年化收益"] = metrics["多空_超额收益"] = metrics["多空_夏普比率"] = \
            metrics["多空_最大回撤"] = metrics["多空_胜率"] = np.nan

        if len(ic_series) > 0:
            metrics["IC均值"] = np.mean(ic_series)
            metrics["IC标准差"] = np.std(ic_series)
            metrics["年化ICIR"] = (metrics["IC均值"] / metrics["IC标准差"]) if metrics["IC标准差"] != 0 else 0.0
        else:
            metrics["IC均值"] = metrics["IC标准差"] = metrics["年化ICIR"] = np.nan
            
        if len(rank_ic_series) > 0:
            metrics["RankIC均值"] = np.mean(rank_ic_series)
            metrics["RankIC标准差"] = np.std(rank_ic_series)
            metrics["年化RankICIR"] = (metrics["RankIC均值"] / metrics["RankIC标准差"]) if metrics["RankIC标准差"] != 0 else 0.0
            metrics["RankIC>0占比"] = np.mean(rank_ic_series > 0) if len(rank_ic_series) > 0 else 0.0
        else:
            metrics["RankIC均值"] = metrics["RankIC标准差"] = metrics["年化RankICIR"] = metrics["RankIC>0占比"] = np.nan
            
        all_metrics[factor_name] = metrics
        
    return pd.DataFrame(all_metrics).T.reindex(REPORT_FACTOR_ORDER, fill_value=np.nan)


# --- 绘图函数 ---

def _format_cell_value(value, data_type="float", decimals=2, text_color=None):
    """辅助函数：格式化单元格文本和颜色"""
    if pd.isna(value) or (isinstance(value, float) and (np.isinf(value))): 
        return {"text": "-", "color": "black"} 
    
    text_val = ""
    if data_type == "percent":
        text_val = f"{value * 100:.{decimals}f}%"
    elif data_type == "float":
         text_val = f"{value:.{decimals}f}"
    else: 
        text_val = str(value)

    color_to_use = text_color if text_color else "black"
    return {"text": text_val, "color": color_to_use}

def _create_report_table(fig, ax, 
                         col_level0_labels, # 新增：最顶层合并表头，例如 "多头表现"
                         col_level0_spans,  # 新增：最顶层合并表头的跨列数
                         col_level1_labels, # 原来的 col_level2_labels，即具体指标名
                         data_df, data_format_config, source_text, 
                         user_col_widths: List[float] = None):
    """通用函数创建研报风格表格，支持模拟的两层表头"""
    ax.axis('off')
    # 标题不再通过此函数设置，直接在调用处 ax.set_title()

    cell_text_list = []
    text_colours_list = [] 

    for factor_name_idx, factor_name in enumerate(data_df.index):
        row_text = [factor_name] # 第一列是因子名称
        row_text_colours = ["black"] 

        for col_config in data_format_config:
            col_key = col_config["key"]
            value = data_df.loc[factor_name, col_key] if factor_name in data_df.index else np.nan
            
            cell_text_color_val = "black" 
            if "text_color_rule" in col_config:
                rule = col_config["text_color_rule"]
                if rule == "red_if_excess_return": 
                    cell_text_color_val = COLOR_REPORT_RED_TEXT
                elif rule == "dark_yellow_if_rankicir":
                    cell_text_color_val = COLOR_REPORT_DARK_YELLOW_TEXT
            
            formatted_cell = _format_cell_value(value, 
                                               data_type=col_config.get("type", "float"),
                                               decimals=col_config.get("decimals", 1 if col_config.get("type") == "percent" else 2),
                                               text_color=cell_text_color_val)
            row_text.append(formatted_cell["text"])
            row_text_colours.append(formatted_cell["color"])
            
        cell_text_list.append(row_text)
        text_colours_list.append(row_text_colours)

    num_data_rows = len(data_df)
    # 总列数由 col_level1_labels (具体指标列) 决定，第一列是因子名
    num_total_render_cols = len(col_level1_labels) # 这包括了第一列的因子名占位符

    if user_col_widths and len(user_col_widths) == num_total_render_cols:
        col_widths = user_col_widths
    else: # 默认列宽计算
        default_factor_col_width = 0.12 
        if num_total_render_cols > 1:
            other_cols_total_width = 1.0 - default_factor_col_width
            avg_other_col_width = other_cols_total_width / (num_total_render_cols - 1)
            col_widths = [default_factor_col_width] + [avg_other_col_width] * (num_total_render_cols - 1)
        else: # 只有因子列
            col_widths = [1.0]

    # --- 创建表格主体 (数据行 + L1表头作为列标签) ---
    # L1表头 (具体指标名) 将作为colLabels传递给table
    the_table = ax.table(cellText=cell_text_list,
                         colLabels=col_level1_labels, # 使用L1表头作为列标签
                         colWidths=col_widths,
                         rowLabels=None, # 我们不在rowLabels放因子名，因子名是数据的第一列
                         loc='center', # 定位整个表格对象
                         cellLoc='center')
    
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(7.5) 

    # --- 格式化单元格 ---
    default_cell_height = 0.12 # 稍微增加行高以容纳两行文字的可能（虽然现在是一行）
    
    # 格式化表头单元格 (colLabels创建的)
    for j in range(num_total_render_cols):
        cell = the_table._cells[(0, j)] # 表头行是第0行
        cell.set_text_props(weight='bold', color='black', size=7.5)
        cell.set_facecolor(COLOR_REPORT_HEADER_BG)
        cell.set_edgecolor(COLOR_REPORT_BORDER)
        cell.set_linewidth(0.5)
        cell.set_height(default_cell_height)

    # 格式化数据单元格 (从第1行开始，因为第0行是表头了)
    for i in range(num_data_rows):
        for j in range(num_total_render_cols):
            cell = the_table._cells[(i + 1, j)] # 数据行现在是 i+1
            cell.set_text_props(color=text_colours_list[i][j]) # text_colours_list[i][j] 对应 cell_text_list[i][j]
            cell.set_facecolor(COLOR_REPORT_FACTOR_BG if j == 0 else COLOR_REPORT_CELL_BG) #第一列因子名背景色
            cell.set_edgecolor(COLOR_REPORT_BORDER)
            cell.set_linewidth(0.5)
            cell.set_height(default_cell_height)

    # --- 手动添加最顶层合并表头 (L0) ---
    # 这层表头绘制在由colLabels创建的表头之上
    # 需要获取表格的实际边界来定位
    fig.canvas.draw() # 确保表格布局完成
    
    # 获取由colLabels创建的表头行(table内部行号0)的y位置和高度
    # header_bbox = the_table._cells[(0,0)].get_bbox().transformed(ax.transData.inverted())
    # table_bbox = the_table.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    # header_y_pos = table_bbox.y1 # 表格的最顶部y坐标
    
    # 基于现有表头单元格定位新的顶层表头
    # (0,0) 是 colLabels 创建的表头的左上角单元格
    ref_cell_bbox = the_table.get_celld()[(0,0)].get_bbox()
    y0 = ref_cell_bbox.y1 # 在这个单元格的顶部之上绘制
    
    current_col_start_x = ref_cell_bbox.x0 # 第一个L0表头的起始x

    for k, (label_l0, span_l0) in enumerate(zip(col_level0_labels, col_level0_spans)):
        # 计算这个L0表头覆盖的列的总宽度和起始x位置
        width_l0 = 0
        # 起始x坐标是当前L0表头覆盖的第一列的左边界
        start_x_l0 = the_table.get_celld()[(0, sum(col_level0_spans[:k]))].get_bbox().x0
        
        for s_idx in range(span_l0):
            # col_idx_in_table = sum(col_level0_spans[:k]) + s_idx # 这个L0表头覆盖的L1表头的实际列索引
            # width_l0 += col_widths[col_idx_in_table] # 不再用col_widths, 直接用bbox计算
            pass

        # 结束x坐标是当前L0表头覆盖的最后一列的右边界
        end_x_l0 = the_table.get_celld()[(0, sum(col_level0_spans[:k+1])-1)].get_bbox().x1
        width_l0 = end_x_l0 - start_x_l0
        
        # 创建一个新的文本对象作为顶层表头
        ax.text((start_x_l0 + end_x_l0) / 2, # x居中
                y0 + default_cell_height * 0.5, # y位置在原表头之上
                label_l0,
                ha='center', va='center',
                fontsize=7.5, weight='bold', color='black',
                bbox=dict(boxstyle='square,pad=0', fc=COLOR_REPORT_HEADER_BG, ec=COLOR_REPORT_BORDER, lw=0.5),
                transform=ax.transData # 使用数据坐标
               )
    
    the_table.scale(1, 1.2) # 调整整体缩放以适应内容

    plt.figtext(0.03, 0.01, source_text, fontsize=6, color='dimgray', ha='left') 
    plt.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.08) # 增加顶部空间给手动添加的表头


def plot_factor_returns_table_report_style(metrics_df: pd.DataFrame, save_path: str = None):
    if metrics_df.empty:
        print("没有可供绘制的因子收益数据。")
        return

    fig, ax = plt.subplots(figsize=(10, 2.8)) 
    ax.set_title("因子收益表现", fontsize=10, weight='bold', loc='left', y=0.95) # 单独设置标题

    # 最顶层合并表头
    col_level0_labels = ["", "多头表现", "多空表现"]  # 第一个为空，对应因子列
    col_level0_spans = [1, 6, 5] 

    # 具体指标列名 (现在是L1表头)
    col_level1_labels = [ 
        "因子", # 第一列是因子名
        "年化收益", "超额收益", "夏普比率", "超额回撤", "相对胜率", "信息比率", # 多头
        "年化收益 ", "超额收益 ", "夏普比率 ", "最大回撤", "胜率"  # 多空
    ]
    
    col_widths_relative = [1.6, 1.1, 1.1, 0.9, 1.1, 1.1, 0.9, 1.1, 1.1, 0.9, 1.1, 0.9] 
    total_relative = sum(col_widths_relative)
    col_widths_normalized = [w / total_relative for w in col_widths_relative]

    data_format_config = [
        {"key": "多头_年化收益", "type": "percent", "decimals": 1},
        {"key": "多头_超额收益", "type": "percent", "decimals": 1, "text_color_rule": "red_if_excess_return"},
        {"key": "多头_夏普比率", "type": "float", "decimals": 2},
        {"key": "多头_超额回撤", "type": "percent", "decimals": 1},
        {"key": "多头_相对胜率", "type": "percent", "decimals": 1},
        {"key": "多头_信息比率", "type": "float", "decimals": 2},
        {"key": "多空_年化收益", "type": "percent", "decimals": 1},
        {"key": "多空_超额收益", "type": "percent", "decimals": 1, "text_color_rule": "red_if_excess_return"}, 
        {"key": "多空_夏普比率", "type": "float", "decimals": 2},
        {"key": "多空_最大回撤", "type": "percent", "decimals": 1},
        {"key": "多空_胜率", "type": "percent", "decimals": 1},
    ]
    
    source = "资料来源: Tushare" 
    
    _create_report_table(fig, ax, col_level0_labels, col_level0_spans, 
                         col_level1_labels, metrics_df, data_format_config, source, 
                         user_col_widths=col_widths_normalized)

    if save_path:
        plt.savefig(save_path, dpi=300) 
        print(f"因子收益表现图表已保存至：{save_path}")
    plt.close(fig)


def plot_factor_ic_table_report_style(metrics_df: pd.DataFrame, save_path: str = None):
    if metrics_df.empty:
        print("没有可供绘制的因子IC数据。")
        return

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_title("因子IC表现", fontsize=10, weight='bold', loc='left', y=0.95)

    col_level0_labels = ["", "IC表现", "rankIC表现"] 
    col_level0_spans = [1, 3, 4] 

    col_level1_labels = [
        "因子", 
        "IC均值", "IC标准差", "年化ICIR",                        
        "rankIC均值", "rankIC标准差", "年化rankICIR", "rankIC>0占比" 
    ]
    col_widths_relative_ic = [1.8, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2] 
    total_relative_ic = sum(col_widths_relative_ic)
    col_widths_normalized_ic = [w / total_relative_ic for w in col_widths_relative_ic]

    data_format_config = [
        {"key": "IC均值", "type": "percent", "decimals": 2},
        {"key": "IC标准差", "type": "float", "decimals": 2}, 
        {"key": "年化ICIR", "type": "float", "decimals": 2},
        {"key": "RankIC均值", "type": "percent", "decimals": 2, "text_color_rule": "red_if_excess_return"}, 
        {"key": "RankIC标准差", "type": "float", "decimals": 2}, 
        {"key": "年化RankICIR", "type": "float", "decimals": 2, "text_color_rule": "dark_yellow_if_rankicir"}, 
        {"key": "RankIC>0占比", "type": "percent", "decimals": 1},
    ]
    
    source = "资料来源: Tushare"

    _create_report_table(fig, ax, col_level0_labels, col_level0_spans, 
                         col_level1_labels, metrics_df, data_format_config, source, 
                         user_col_widths=col_widths_normalized_ic)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"因子IC表现图表已保存至：{save_path}")
    plt.close(fig)


def generate_report_style_plots():
    os.makedirs(RESULT_DIR, exist_ok=True)
    log_data_content = load_log_data()
    if not log_data_content:
        print("未能加载log数据，无法生成图表。")
        return

    factor_metrics_summary = get_factor_performance_metrics(log_data_content)
    if factor_metrics_summary.empty:
        if not any(factor_name in factor_metrics_summary.index for factor_name in REPORT_FACTOR_ORDER if factor_name in log_data_content.get("因子表现数据", {})):
             print("计算得到的因子指标DataFrame为空或不包含任何预期的因子，无法生成图表。")
             return
        print("警告: 计算得到的因子指标DataFrame为空，但已尝试处理因子。请检查指标计算逻辑或输入数据。")

    print("\n计算得到的因子表现汇总:")
    pd.set_option('display.width', 120) 
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.precision', 4) 
    print(factor_metrics_summary)

    returns_table_output_path = os.path.join(RESULT_DIR, "因子收益表现.png")
    plot_factor_returns_table_report_style(factor_metrics_summary, returns_table_output_path)

    ic_table_output_path = os.path.join(RESULT_DIR, "因子IC表现.png")
    plot_factor_ic_table_report_style(factor_metrics_summary, ic_table_output_path)
    
    print(f"\n所有图表已生成并尝试保存到：{RESULT_DIR}")

if __name__ == "__main__":
    print("开始生成图表...")
    generate_report_style_plots()
    print("图表生成流程结束。")

