import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

class TransformerVisualizationTools:
    """
    Transformer模型可视化和结果分析工具
    提供高级图表生成和深入分析功能
    """
    def __init__(self, result_dir='./analysis_results'):
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置美观的样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def create_training_curves(self, experiments, metrics=['train_loss', 'val_loss'], save_path=None):
        """
        创建训练曲线对比图
        
        Args:
            experiments (dict): 实验数据字典，键为实验名称，值为包含metrics的字典
            metrics (list): 要绘制的指标列表
            save_path (str): 保存路径
        """
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 5 * num_metrics), squeeze=False)
        
        for i, metric in enumerate(metrics):
            ax = axes[i, 0]
            
            for exp_name, exp_data in experiments.items():
                if metric in exp_data['metrics']:
                    values = exp_data['metrics'][metric]
                    ax.plot(values, label=exp_name, linewidth=2)
            
            # 设置轴标签和标题
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epochs / Steps', fontsize=12)
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
            
            # 添加图例和网格
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 设置坐标轴刻度
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存至 {save_path}")
        else:
            plt.show()
    
    def create_bleu_radar_chart(self, experiments, save_path=None):
        """
        创建BLEU分数雷达图
        
        Args:
            experiments (dict): 实验数据字典
            save_path (str): 保存路径
        """
        # 准备数据
        exp_names = list(experiments.keys())
        metrics = ['val_bleu', 'test_bleu']
        
        # 提取数据
        data = []
        for exp_name, exp_data in experiments.items():
            row = []
            for metric in metrics:
                if metric in exp_data['metrics']:
                    # 使用最大值
                    row.append(max(exp_data['metrics'][metric]) if isinstance(exp_data['metrics'][metric], list) else exp_data['metrics'][metric])
                else:
                    row.append(0)
            data.append(row)
        
        # 计算角度
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # 绘制每个实验的数据
        for i, exp_name in enumerate(exp_names):
            values = data[i] + values[:1]  # 闭合雷达图
            ax.plot(angles, values, linewidth=2, label=exp_name)
            ax.fill(angles, values, alpha=0.1)
        
        # 设置雷达图标签
        metric_labels = [m.replace("_", " ").title() for m in metrics]
        ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=12)
        
        # 设置y轴范围
        ax.set_ylim(0, min(100, max(max(d) for d in data) * 1.2))
        
        # 添加图例和标题
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('BLEU Score Comparison Across Experiments', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"雷达图已保存至 {save_path}")
        else:
            plt.show()
    
    def create_component_impact_heatmap(self, component_effects, save_path=None):
        """
        创建组件影响热力图
        
        Args:
            component_effects (dict): 组件影响字典，键为组件名，值为BLEU分数提升
            save_path (str): 保存路径
        """
        # 准备数据为DataFrame
        data = []
        components = []
        impacts = []
        
        for comp, impact in component_effects.items():
            components.append(comp)
            impacts.append(impact)
        
        # 创建二维数组用于热力图
        impact_matrix = np.array(impacts).reshape(1, -1)
        
        plt.figure(figsize=(12, 4))
        
        # 使用seaborn创建热力图
        sns.heatmap(impact_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                    xticklabels=components, yticklabels=['BLEU Impact'],
                    cbar_kws={'label': 'BLEU Score Improvement'})
        
        plt.title('Component Impact on BLEU Score', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"热力图已保存至 {save_path}")
        else:
            plt.show()
    
    def create_performance_comparison_bar(self, experiments, metric='val_bleu', save_path=None):
        """
        创建性能对比条形图
        
        Args:
            experiments (dict): 实验数据字典
            metric (str): 要比较的指标
            save_path (str): 保存路径
        """
        # 提取数据
        exp_names = []
        values = []
        
        for exp_name, exp_data in experiments.items():
            if metric in exp_data['metrics']:
                exp_names.append(exp_name)
                # 使用最大值
                val = exp_data['metrics'][metric]
                if isinstance(val, list):
                    values.append(max(val))
                else:
                    values.append(val)
        
        # 创建条形图
        plt.figure(figsize=(12, 6))
        
        # 按值排序
        sorted_pairs = sorted(zip(values, exp_names), reverse=True)
        values, exp_names = zip(*sorted_pairs)
        
        # 设置颜色渐变
        colors = sns.color_palette("viridis", len(values))
        
        bars = plt.barh(exp_names, values, color=colors)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                     f'{width:.2f}', va='center')
        
        plt.title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
        plt.xlabel(metric.replace("_", " ").title(), fontsize=12)
        plt.grid(True, axis='x', alpha=0.3)
        
        # 设置x轴范围
        plt.xlim(0, max(values) * 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能对比图已保存至 {save_path}")
        else:
            plt.show()
    
    def create_ablation_summary(self, experiments, save_path=None):
        """
        创建消融实验总结图表
        
        Args:
            experiments (dict): 实验数据字典
            save_path (str): 保存路径
        """
        # 准备数据
        rows = []
        
        for exp_name, exp_data in experiments.items():
            row = {
                'Experiment': exp_name,
                'Val BLEU': max(exp_data['metrics'].get('val_bleu', [0])) if 'val_bleu' in exp_data['metrics'] else 0,
                'Test BLEU': max(exp_data['metrics'].get('test_bleu', [0])) if 'test_bleu' in exp_data['metrics'] else 0,
                'Best Val Loss': min(exp_data['metrics'].get('val_loss', [float('inf')])) if 'val_loss' in exp_data['metrics'] else float('inf')
            }
            
            # 添加组件配置
            for key, value in exp_data['hyperparams'].items():
                if key.startswith('use_'):
                    row[key.replace('use_', '')] = '✓' if value else '✗'
            
            rows.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(rows)
        
        # 按Val BLEU排序
        df = df.sort_values('Val BLEU', ascending=False)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # 创建表格
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # 设置表头样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')
            cell.set_edgecolor('#dddddd')
        
        plt.title('Ablation Study Summary', fontsize=14, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"消融实验总结表已保存至 {save_path}")
            
            # 同时保存CSV格式
            csv_path = save_path.replace('.png', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"消融实验数据已保存至 {csv_path}")
        else:
            plt.show()
    
    def analyze_attention_heatmaps(self, model, src_sentence, tgt_sentence, layer=0, head=0, save_path=None):
        """
        分析并可视化注意力热图（需要模型支持注意力权重提取）
        
        Args:
            model: Transformer模型
            src_sentence: 源句子
            tgt_sentence: 目标句子
            layer: 要分析的层索引
            head: 要分析的注意力头索引
            save_path: 保存路径
        """
        try:
            # 注意：这部分需要模型支持提取注意力权重
            # 这里提供一个示例实现，实际使用时需要根据模型实现进行调整
            attention_weights = model.get_attention_weights(src_sentence, tgt_sentence)
            
            # 提取指定层和头的注意力权重
            attn_map = attention_weights[layer][head].cpu().detach().numpy()
            
            plt.figure(figsize=(10, 8))
            
            # 创建热力图
            sns.heatmap(attn_map, xticklabels=src_sentence, yticklabels=tgt_sentence,
                        cmap='viridis', cbar_kws={'label': 'Attention Weight'})
            
            plt.title(f'Attention Heatmap - Layer {layer+1}, Head {head+1}', fontsize=14, fontweight='bold')
            plt.xlabel('Source Tokens', fontsize=12)
            plt.ylabel('Target Tokens', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"注意力热图已保存至 {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"生成注意力热图时出错: {e}")
            print("请确保模型支持get_attention_weights方法")
    
    def create_training_speed_comparison(self, experiments, save_path=None):
        """
        创建训练速度对比图
        
        Args:
            experiments (dict): 实验数据字典
            save_path (str): 保存路径
        """
        # 假设每个实验都有训练时间记录
        exp_names = []
        times = []
        
        for exp_name, exp_data in experiments.items():
            if 'training_time' in exp_data['metrics']:
                exp_names.append(exp_name)
                times.append(exp_data['metrics']['training_time'])
        
        if not exp_names:
            print("未找到训练时间数据")
            return
        
        plt.figure(figsize=(10, 6))
        
        # 创建条形图
        colors = sns.color_palette("husl", len(times))
        bars = plt.bar(exp_names, times, color=colors)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Experiment', fontsize=12)
        plt.ylabel('Training Time (seconds)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练速度对比图已保存至 {save_path}")
        else:
            plt.show()
    
    def generate_full_analysis_report(self, experiments, report_dir=None):
        """
        生成完整的分析报告
        
        Args:
            experiments (dict): 实验数据字典
            report_dir (str): 报告保存目录
        """
        if report_dir is None:
            report_dir = self.result_dir
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. 生成消融实验总结表
        summary_path = os.path.join(report_dir, 'ablation_summary.png')
        self.create_ablation_summary(experiments, summary_path)
        
        # 2. 生成训练曲线
        curves_path = os.path.join(report_dir, 'training_curves.png')
        self.create_training_curves(experiments, ['train_loss', 'val_loss'], curves_path)
        
        # 3. 生成BLEU分数对比
        bleu_path = os.path.join(report_dir, 'bleu_comparison.png')
        self.create_performance_comparison_bar(experiments, 'val_bleu', bleu_path)
        
        # 4. 计算组件影响并生成热力图
        component_effects = self._calculate_component_effects(experiments)
        if component_effects:
            heatmap_path = os.path.join(report_dir, 'component_impact_heatmap.png')
            self.create_component_impact_heatmap(component_effects, heatmap_path)
        
        # 5. 生成训练速度对比（如果有数据）
        speed_path = os.path.join(report_dir, 'training_speed_comparison.png')
        self.create_training_speed_comparison(experiments, speed_path)
        
        print(f"\n完整分析报告已生成，保存至 {report_dir}")
        print("\n报告包含以下文件：")
        print(f"1. {summary_path} - 消融实验总结表")
        print(f"2. {curves_path} - 训练曲线对比")
        print(f"3. {bleu_path} - BLEU分数对比")
        if component_effects:
            print(f"4. {heatmap_path} - 组件影响热力图")
        print(f"5. {speed_path} - 训练速度对比")
    
    def _calculate_component_effects(self, experiments):
        """
        计算各组件对模型性能的影响
        """
        components = {
            'multihead': '多头注意力',
            'positional_encoding': '位置编码',
            'residual': '残差连接',
            'layernorm': 'LayerNorm',
            'decoder': '解码器'
        }
        
        component_effects = {}
        
        for comp_key, comp_name in components.items():
            with_comp = []
            without_comp = []
            
            for exp_name, exp_data in experiments.items():
                # 检查是否存在该组件的配置
                config_key = f'use_{comp_key}'
                if config_key in exp_data['hyperparams']:
                    metrics = exp_data['metrics']
                    val_bleu = max(metrics.get('val_bleu', [0])) if 'val_bleu' in metrics else 0
                    
                    if exp_data['hyperparams'][config_key]:
                        with_comp.append(val_bleu)
                    else:
                        without_comp.append(val_bleu)
            
            # 计算平均影响
            if with_comp and without_comp:
                avg_with = np.mean(with_comp)
                avg_without = np.mean(without_comp)
                effect = avg_with - avg_without
                component_effects[comp_name] = effect
        
        return component_effects

# 主函数，提供示例用法
def main():
    # 创建可视化工具实例
    viz_tools = TransformerVisualizationTools()
    
    # 生成示例实验数据
    sample_experiments = {
        'baseline': {
            'hyperparams': {
                'use_multihead': True,
                'use_positional_encoding': True,
                'use_residual': True,
                'use_layernorm': True,
                'use_decoder': True
            },
            'metrics': {
                'train_loss': [3.2, 2.5, 2.1, 1.8, 1.6, 1.4, 1.3, 1.2, 1.1, 1.0],
                'val_loss': [2.8, 2.3, 2.0, 1.9, 1.8, 1.7, 1.6, 1.6, 1.5, 1.5],
                'val_bleu': [25, 30, 35, 38, 41, 44, 46, 48, 49, 50],
                'training_time': 1200
            }
        },
        'no_multihead': {
            'hyperparams': {
                'use_multihead': False,
                'use_positional_encoding': True,
                'use_residual': True,
                'use_layernorm': True,
                'use_decoder': True
            },
            'metrics': {
                'train_loss': [3.5, 2.8, 2.4, 2.2, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5],
                'val_loss': [3.1, 2.6, 2.3, 2.1, 2.0, 1.9, 1.8, 1.8, 1.7, 1.7],
                'val_bleu': [20, 24, 28, 31, 34, 37, 39, 41, 43, 44],
                'training_time': 900
            }
        },
        'encoder_only': {
            'hyperparams': {
                'use_multihead': True,
                'use_positional_encoding': True,
                'use_residual': True,
                'use_layernorm': True,
                'use_decoder': False
            },
            'metrics': {
                'train_loss': [3.3, 2.6, 2.2, 1.9, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2],
                'val_loss': [2.9, 2.4, 2.1, 2.0, 1.9, 1.8, 1.7, 1.7, 1.6, 1.6],
                'val_bleu': [22, 27, 32, 36, 39, 42, 45, 47, 48, 49],
                'training_time': 700
            }
        }
    }
    
    print("示例分析：")
    # 生成完整分析报告
    viz_tools.generate_full_analysis_report(sample_experiments)

if __name__ == "__main__":
    main()