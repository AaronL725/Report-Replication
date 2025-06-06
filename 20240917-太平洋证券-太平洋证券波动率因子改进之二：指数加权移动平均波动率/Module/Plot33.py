"""
绘图模块
功能：读取log.log文件数据，绘制分组净值图表，并保存到Result文件夹
- EWMAVOL因子在全A中回测的分组净值
- VOL_3M因子在全A中回测的分组净值
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
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

# --- 路径设置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(CURRENT_DIR, "log.log")
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if "Factor" in PROJECT_ROOT or "Plot" in PROJECT_ROOT:  # 修正项目根目录判断
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
RESULT_DIR = os.path.join(PROJECT_ROOT, "Result")

# 确保Result目录存在
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# 研报中的因子顺序
REPORT_FACTOR_ORDER = ['VOL_3M', 'RANKVOL', 'RVOL', 'GARCHVOL', 'EWMAVOL']

# --- 数据加载与处理函数 ---

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

def extract_factor_group_returns(log_data: Dict, factor_name: str) -> Dict:
    """从log数据中提取特定因子的分组收益率数据"""
    if not log_data or "因子表现数据" not in log_data:
        print("日志数据格式错误或为空")
        return {}
    
    # 从log数据中直接提取日期序列
    dates = log_data.get("回测日期序列", [])
    
    # 从log数据中直接提取特定因子的各分组收益率
    factor_data = log_data.get("因子表现数据", {}).get(factor_name, {})
    group_returns = factor_data.get("各分组收益率", {})
    
    # 获取多空组合的收益率序列
    long_short_returns = factor_data.get("多空组合收益率序列", [])
    
    return {
        "dates": dates,
        "group_returns": group_returns,
        "long_short_returns": long_short_returns
    }

def calculate_cumulative_returns(return_series: List[float]) -> List[float]:
    """计算累积收益率序列(净值序列)"""
    if not return_series:
        return []
    
    # 处理可能的前导NaN或None值
    first_valid_idx = 0
    for i, r in enumerate(return_series):
        if r is not None and not np.isnan(r):
            first_valid_idx = i
            break
    
    # 使用第一个有效值之前的所有值都设为0收益率
    # 计算累积净值序列 (初始值为1)
    cumulative_returns = [1.0] * len(return_series)
    for i in range(first_valid_idx + 1, len(return_series)):
        if return_series[i] is None or np.isnan(return_series[i]):
            # 如果当前收益率是NaN，使用前一个净值
            cumulative_returns[i] = cumulative_returns[i-1]
        else:
            # 否则正常累积计算
            cumulative_returns[i] = cumulative_returns[i-1] * (1 + return_series[i])
    
    return cumulative_returns

# --- 绘图函数 ---

def plot_group_net_value(factor_name: str, data: Dict, save_path: str = None):
    """绘制因子分组净值图表"""
    if not data:
        print(f"没有找到{factor_name}因子的数据")
        return
    
    # 提取日期和各分组收益率
    dates = data.get("dates", [])
    group_returns = data.get("group_returns", {})
    long_short_returns = data.get("long_short_returns", [])
    
    # 如果没有日期数据，无法绘图
    if not dates:
        print(f"{factor_name}因子的日期数据为空")
        return
    
    # 转换日期格式为datetime对象以方便绘图
    try:
        plot_dates = pd.to_datetime(dates)
    except Exception as e:
        print(f"日期转换错误: {e}")
        plot_dates = range(len(dates))
    
    # 设置新图表
    plt.figure(figsize=(11, 6))
    ax = plt.subplot(111)
    
    # 设置图表背景为浅绿色
    ax.set_facecolor('#e6f2e9')  # 浅绿色背景
    fig = plt.gcf()
    fig.patch.set_facecolor('#e6f2e9')
    
    # 绘制网格线
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, color='white')
    
    # 定义每个分组的颜色
    group_colors = {
        "第一组": "#c3dbd4",  # 最浅的颜色
        "第二组": "#8fbfd3",  # 较浅的颜色
        "第三组": "#5d9fc9",  # 中等颜色
        "第四组": "#2a7eb8",  # 较深的颜色
        "第五组": "#0a5c91",  # 最深的颜色
        "多空": "#e68a00"     # 橙色用于多空组合
    }
    
    # 计算并绘制各分组的净值曲线
    for group_name, returns in group_returns.items():
        if returns:  # 确保有数据
            # 如果returns是单个值的列表，需要复制它，使其长度与dates一致
            if len(returns) == 1 and len(dates) > 1:
                returns_full = returns * len(dates)
            else:
                returns_full = returns
                
            # 如果returns_full比dates短，用0补齐
            if len(returns_full) < len(dates):
                returns_full = returns_full + [0] * (len(dates) - len(returns_full))
                
            net_values = calculate_cumulative_returns(returns_full)
            ax.plot(plot_dates, net_values, label=group_name, linewidth=2, color=group_colors.get(group_name))
    
    # 如果有多空组合，绘制多空组合的净值曲线
    if long_short_returns:
        net_values = calculate_cumulative_returns(long_short_returns)
        ax.plot(plot_dates, net_values, label="多空", linewidth=2, color=group_colors.get("多空"))
    
    # 设置x轴标签旋转以避免重叠
    plt.xticks(rotation=45)
    
    # 设置图表标题和标签
    title = f"图表：{factor_name}因子在全A中回测的分组净值"
    plt.title(title, fontsize=12, fontweight='bold', loc='left')
    
    # 添加图例
    legend = ax.legend(loc='lower right')
    frame = legend.get_frame()
    frame.set_alpha(0.8)
    
    # 在图表底部添加数据源信息
    plt.figtext(0.1, 0.01, '资料来源：Tushare', fontsize=10, color='gray', style='italic')
    
    # 设置水平分隔线
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # 设置图表边框
    for spine in ax.spines.values():
        spine.set_color('#4682B4')  # 设置边框颜色为钢青色
        spine.set_linewidth(2)      # 设置边框宽度
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    # 显示图表
    plt.close()

# --- 主函数 ---

def generate_group_net_value_plots():
    """生成并保存所有图表"""
    # 确保Result目录存在
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        print(f"创建目录: {RESULT_DIR}")
    
    # 加载日志数据
    log_data = load_log_data()
    if not log_data:
        print("无法加载日志数据，退出绘图")
        return False
    
    # 绘制EWMAVOL因子分组净值图
    ewmavol_data = extract_factor_group_returns(log_data, "EWMAVOL")
    plot_group_net_value(
        "EWMAVOL", 
        ewmavol_data, 
        os.path.join(RESULT_DIR, "EWMAVOL因子在全A中回测的分组净值.png")
    )
    
    # 绘制VOL_3M因子分组净值图
    vol_3m_data = extract_factor_group_returns(log_data, "VOL_3M")
    plot_group_net_value(
        "VOL_3M", 
        vol_3m_data, 
        os.path.join(RESULT_DIR, "VOL_3M因子在全A中回测的分组净值.png")
    )
    
    print("所有图表已生成完成")
    return True

if __name__ == "__main__":
    generate_group_net_value_plots()