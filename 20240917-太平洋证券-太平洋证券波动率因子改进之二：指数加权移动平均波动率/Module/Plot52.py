# filepath: c:\Users\Aaron\Desktop\python期末设计要求\project - 23049009\Module\Plot52.py
"""
5.2 历史数据长度L测试模块
功能：对EWMAVOL因子的历史数据长度L进行测试，分析不同L值对因子表现的影响
测试范围：L ∈ [10, 20, 40, 60, 120]
输出：表格形式的性能对比图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
import sys
import pickle
import warnings
import traceback
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入必要模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_dir, 'Factor'))
sys.path.append(current_dir)

from EWMAVOL import calculate_ewmavol
from Backtesting import FactorBacktester

class HistoricalLengthTester:
    """历史数据长度L测试类"""
    
    def __init__(self, start_date='2008-12-31', end_date='2024-06-07'):
        """
        初始化L测试器
        
        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.L_values = [10, 20, 40, 60, 120]
        self.results = {}
        
        # 初始化回测器
        self.backtester = FactorBacktester(start_date=start_date, end_date=end_date)
        
        print(f"历史数据长度L测试器初始化完成")
        print(f"测试L值范围: {self.L_values}")
        
    def run_L_test(self):
        """运行不同L值的测试"""
        print("开始运行历史数据长度L值测试...")
        
        # 加载数据
        if not self.backtester.load_cached_data():
            print("错误：无法加载缓存数据")
            return None
        
        # 获取调仓日期
        self.backtester._get_month_end_dates()
        if len(self.backtester.rebalance_dates) < 2:
            print("错误：调仓日期不足")
            return None
        
        # 获取收盘价数据
        close_data = self.backtester.data['close']
        
        for i, L_val in enumerate(self.L_values):
            print(f"\n进度: {i+1}/{len(self.L_values)} - 测试L={L_val}")
            
            try:
                # 计算EWMAVOL因子（使用特定L值）
                print(f"计算EWMAVOL因子 (L={L_val})...")
                factor_data = calculate_ewmavol(
                    close_prices=close_data,
                    window=L_val,
                    lambda_decay=0.9  # 固定λ=0.9
                )
                
                # 处理因子数据
                print(f"处理因子数据...")
                processed_factor = self.backtester.process_factor(factor_data, f'EWMAVOL_L{L_val}')
                
                # 计算IC指标
                print(f"计算IC指标...")
                ic_series, rank_ic_series = self.backtester.calculate_factor_ic(processed_factor)
                
                # 计算组合收益
                print(f"计算组合收益...")
                portfolio_returns = self.backtester.calculate_portfolio_returns(processed_factor, n_groups=5)
                
                # 分析结果
                result = self._analyze_performance(
                    portfolio_returns, ic_series, rank_ic_series, L_val
                )
                
                self.results[L_val] = result
                print(f"L={L_val} 测试完成 - 年化收益: {result['long_annual_return']*100:.2f}%")
                
            except Exception as e:
                print(f"L={L_val} 测试失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nL值测试完成，成功测试 {len(self.results)} 个L值")
        return self.results

    def _analyze_performance(self, portfolio_returns, ic_series, rank_ic_series, L_val):
        """分析单个L值的性能表现"""
        result = {}

        print(f"DEBUG: portfolio_returns 列: {portfolio_returns.columns.tolist()}")
        print(f"DEBUG: portfolio_returns 形状: {portfolio_returns.shape}")
        
        # 获取多头组合（第5组，因子值最高的组合）和空头组合（第1组）
        # 处理中文列名
        if '第5组' in portfolio_returns.columns and '第1组' in portfolio_returns.columns:
            long_returns = portfolio_returns['第5组'].dropna()
            short_returns = portfolio_returns['第1组'].dropna()
            print(f"DEBUG: 找到分组数据，多头收益长度: {len(long_returns)}, 空头收益长度: {len(short_returns)}")
        else:
            # 如果没有数据，返回空结果
            print(f"警告：L={L_val} 没有找到分组收益数据")
            print(f"可用列: {portfolio_returns.columns.tolist()}")
            return {
                'long_annual_return': 0,
                'long_sharpe': 0,
                'long_short_annual_return': 0,
                'long_short_sharpe': 0,
                'ic_mean': 0,
                'ic_ir': 0,
                'rank_ic_mean': 0,
                'rank_ic_ir': 0
            }
        
        # 构建多空组合收益
        long_short_returns = (long_returns - short_returns).dropna()
        print(f"DEBUG: 多空组合收益长度: {len(long_short_returns)}")
        
        # 计算多头表现
        result['long_annual_return'] = self._calculate_annual_return(long_returns)
        result['long_sharpe'] = self._calculate_sharpe_ratio(long_returns)
        
        # 计算多空表现
        result['long_short_annual_return'] = self._calculate_annual_return(long_short_returns)
        result['long_short_sharpe'] = self._calculate_sharpe_ratio(long_short_returns)
        
        # 计算IC表现
        ic_mean = ic_series.mean() if len(ic_series) > 0 else 0
        ic_std = ic_series.std() if len(ic_series) > 1 else 1
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        
        result['ic_mean'] = ic_mean
        result['ic_ir'] = ic_ir * np.sqrt(12)  # 月度调仓，年化用sqrt(12)
        
        # 计算RankIC表现
        rank_ic_mean = rank_ic_series.mean() if len(rank_ic_series) > 0 else 0
        rank_ic_std = rank_ic_series.std() if len(rank_ic_series) > 1 else 1
        rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0
        
        result['rank_ic_mean'] = rank_ic_mean
        result['rank_ic_ir'] = rank_ic_ir * np.sqrt(12)  # 月度调仓，年化用sqrt(12)
        
        print(f"DEBUG: L={L_val} 计算结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        return result

    def _calculate_annual_return(self, returns):
        """计算年化收益率"""
        if len(returns) == 0:
            return 0
        total_return = (1 + returns).prod() - 1
        # 月度调仓，按月计算年化收益
        months = len(returns)
        if months <= 0:
            return 0
        annual_return = (1 + total_return) ** (12/months) - 1
        return annual_return
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.03):
        """计算夏普比率"""
        if len(returns) == 0:
            return 0
        # 月度调仓，风险调整为月度
        excess_returns = returns - risk_free_rate/12
        if excess_returns.std() == 0:
            return 0
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
        return sharpe

    def create_performance_table_chart(self, save_path=None):
        """创建性能对比表格图表"""
        if not self.results:
            print("错误：没有测试结果数据")
            return None
        
        # 准备数据
        table_data = []
        for L_val in self.L_values:
            if L_val in self.results:
                result = self.results[L_val]
                row = [
                    L_val,
                    f"{result['long_annual_return']*100:.2f}%",
                    f"{result['long_sharpe']:.2f}",
                    f"{result['long_short_annual_return']*100:.2f}%",
                    f"{result['long_short_sharpe']:.2f}",
                    f"{result['ic_mean']*100:.2f}%",
                    f"{result['ic_ir']:.2f}",
                    f"{result['rank_ic_mean']*100:.2f}%",
                    f"{result['rank_ic_ir']:.2f}"
                ]
                table_data.append(row)
        
        if not table_data:
            print("错误：没有有效的结果数据")
            return None
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # 表格标题
        fig.suptitle('5.2 历史数据长度L\n图表：历史数据长度L不同取值时的因子表现', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # 表格列名
        columns = [
            'L', '年化收益', '夏普比率', '年化收益', '夏普比率', 
            'IC均值', '年化ICIR', 'rankIC均值', '年化rankICIR'
        ]
        
        # 创建表格
        table = ax.table(cellText=table_data, colLabels=columns, 
                        cellLoc='center', loc='center',
                        bbox=[0.05, 0.1, 0.9, 0.75])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # 设置表头样式
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.1)
        
        # 手动添加分组标题
        ax.text(0.125, 0.93, 'L', ha='center', va='center', fontsize=12, fontweight='bold',
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        ax.text(0.275, 0.93, '多头表现', ha='center', va='center', fontsize=12, fontweight='bold',
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        ax.text(0.445, 0.93, '多空表现', ha='center', va='center', fontsize=12, fontweight='bold',
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        ax.text(0.615, 0.93, 'IC表现', ha='center', va='center', fontsize=12, fontweight='bold',
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
        ax.text(0.8, 0.93, 'RANKIC表现', ha='center', va='center', fontsize=12, fontweight='bold',
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
        # 添加分隔线
        line_positions = [0.19, 0.36, 0.53, 0.7]
        for pos in line_positions:
            ax.axvline(x=pos, ymin=0.12, ymax=0.85, color='gray', linewidth=2, linestyle='-')
        
        # 设置数据行样式
        for i in range(1, len(table_data) + 1):
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F8F9FA')
                else:
                    table[(i, j)].set_facecolor('white')
                table[(i, j)].set_height(0.08)
                
                # 高亮最优值（L=60的行，这是研报推荐的默认值）
                if table_data[i-1][0] == 60:  # L=60的行
                    table[(i, j)].set_facecolor('#FFE6E6')
                    if j > 0:  # 不包括L列
                        table[(i, j)].set_text_props(weight='bold', color='red')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            result_dir = os.path.join(os.path.dirname(current_dir), 'Result')
            os.makedirs(result_dir, exist_ok=True)
            save_path = os.path.join(result_dir, 'EWMAVOL因子历史数据长度L测试结果.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"图表已保存到: {save_path}")
        
        plt.show()  # 显示图表
        return fig

    def print_results_summary(self):
        """打印结果摘要"""
        if not self.results:
            print("没有测试结果")
            return
        
        print("\n" + "="*80)
        print("历史数据长度L测试结果摘要")
        print("="*80)
        
        headers = ['L', '多头年化收益', '多头夏普', '多空年化收益', '多空夏普', 
                  'IC均值', '年化ICIR', 'RankIC均值', '年化RankICIR']
        
        print(f"{'L':>6} {'多头年化收益':>12} {'多头夏普':>10} {'多空年化收益':>12} {'多空夏普':>10} "
              f"{'IC均值':>8} {'年化ICIR':>10} {'RankIC均值':>12} {'年化RankICIR':>14}")
        print("-" * 120)
        
        for L_val in self.L_values:
            if L_val in self.results:
                result = self.results[L_val]
                print(f"{L_val:>6d} "
                      f"{result['long_annual_return']*100:>11.2f}% "
                      f"{result['long_sharpe']:>9.2f} "
                      f"{result['long_short_annual_return']*100:>11.2f}% "
                      f"{result['long_short_sharpe']:>9.2f} "
                      f"{result['ic_mean']*100:>7.2f}% "
                      f"{result['ic_ir']:>9.2f} "
                      f"{result['rank_ic_mean']*100:>11.2f}% "
                      f"{result['rank_ic_ir']:>13.2f}")


def main():
    """主函数"""
    print("开始EWMAVOL因子历史数据长度L测试...")
    
    # 创建L测试器
    tester = HistoricalLengthTester(start_date='2008-12-31', end_date='2024-06-07')
    
    # 运行测试
    results = tester.run_L_test()
    
    if results:
        # 打印结果摘要
        tester.print_results_summary()
        
        # 创建并保存图表
        fig = tester.create_performance_table_chart()
        
        print("\n测试完成！")
    else:
        print("测试失败，请检查数据和配置")


if __name__ == "__main__":
    main()