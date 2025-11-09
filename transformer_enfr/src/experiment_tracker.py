import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class ExperimentTracker:
    """
    实验跟踪器，用于记录、保存和可视化不同消融实验的结果
    """
    def __init__(self, log_dir='./experiment_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.experiments = {}
        
    def add_experiment(self, exp_name, hyperparams, metrics):
        """
        添加一个实验结果
        
        Args:
            exp_name (str): 实验名称
            hyperparams (dict): 实验超参数
            metrics (dict): 实验指标（训练损失、验证损失、BLEU分数等）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiments[exp_name] = {
            'timestamp': timestamp,
            'hyperparams': hyperparams,
            'metrics': metrics
        }
        self.save_experiment(exp_name)
        print(f"实验 {exp_name} 已添加并保存")
    
    def save_experiment(self, exp_name):
        """
        保存单个实验结果
        """
        if exp_name in self.experiments:
            filename = f"{exp_name}_{self.experiments[exp_name]['timestamp']}.json"
            filepath = os.path.join(self.log_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.experiments[exp_name], f, ensure_ascii=False, indent=2)
    
    def load_experiment(self, filepath):
        """
        加载实验结果
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            exp_data = json.load(f)
        
        # 从文件名中提取实验名称
        exp_name = os.path.basename(filepath).split('_')[0]
        self.experiments[exp_name] = exp_data
        return exp_name, exp_data
    
    def load_all_experiments(self):
        """
        加载所有已保存的实验结果
        """
        for filename in os.listdir(self.log_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.log_dir, filename)
                self.load_experiment(filepath)
        print(f"已加载 {len(self.experiments)} 个实验")
    
    def plot_loss_comparison(self, exp_names=None, save_path=None):
        """
        绘制不同实验的损失对比图
        """
        if exp_names is None:
            exp_names = list(self.experiments.keys())
        
        plt.figure(figsize=(12, 6))
        
        # 绘制训练损失
        for exp_name in exp_names:
            if exp_name in self.experiments and 'train_loss' in self.experiments[exp_name]['metrics']:
                losses = self.experiments[exp_name]['metrics']['train_loss']
                plt.plot(losses, label=f'{exp_name} - 训练')
        
        plt.title('不同实验的训练损失对比')
        plt.xlabel('训练步数')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"损失对比图已保存至 {save_path}")
        else:
            plt.show()
    
    def plot_bleu_comparison(self, exp_names=None, save_path=None):
        """
        绘制不同实验的BLEU分数对比图
        """
        if exp_names is None:
            exp_names = list(self.experiments.keys())
        
        plt.figure(figsize=(12, 6))
        
        # 绘制验证集BLEU分数
        for exp_name in exp_names:
            if exp_name in self.experiments and 'val_bleu' in self.experiments[exp_name]['metrics']:
                bleu_scores = self.experiments[exp_name]['metrics']['val_bleu']
                plt.plot(bleu_scores, label=f'{exp_name} - 验证BLEU')
        
        plt.title('不同实验的BLEU分数对比')
        plt.xlabel('评估轮次')
        plt.ylabel('BLEU分数')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"BLEU对比图已保存至 {save_path}")
        else:
            plt.show()
    
    def create_comparison_table(self, exp_names=None):
        """
        创建实验结果对比表格
        """
        if exp_names is None:
            exp_names = list(self.experiments.keys())
        
        print("===== 消融实验结果对比 =====")
        print("\n实验名称 | 验证BLEU | 最佳验证损失 | 参数量")
        print("-" * 70)
        
        for exp_name in exp_names:
            if exp_name in self.experiments:
                metrics = self.experiments[exp_name]['metrics']
                hyperparams = self.experiments[exp_name]['hyperparams']
                
                # 提取关键指标
                val_bleu = max(metrics.get('val_bleu', [0])) if 'val_bleu' in metrics else 0
                best_val_loss = min(metrics.get('val_loss', [float('inf')])) if 'val_loss' in metrics else float('inf')
                
                # 生成实验配置摘要
                config_str = []
                if 'use_multihead' in hyperparams:
                    config_str.append(f"多头注意力: {'✓' if hyperparams['use_multihead'] else '✗'}")
                if 'use_positional_encoding' in hyperparams:
                    config_str.append(f"位置编码: {'✓' if hyperparams['use_positional_encoding'] else '✗'}")
                if 'use_residual' in hyperparams:
                    config_str.append(f"残差连接: {'✓' if hyperparams['use_residual'] else '✗'}")
                if 'use_layernorm' in hyperparams:
                    config_str.append(f"LayerNorm: {'✓' if hyperparams['use_layernorm'] else '✗'}")
                if 'use_decoder' in hyperparams:
                    config_str.append(f"解码器: {'✓' if hyperparams['use_decoder'] else '✗'}")
                
                # 打印结果
                print(f"{exp_name:<10} | {val_bleu:.2f} | {best_val_loss:.4f} | {', '.join(config_str)}")
        print("-" * 70)

    def generate_ablation_analysis(self, save_dir=None):
        """
        生成消融实验分析报告
        """
        if save_dir is None:
            save_dir = './analysis_results'
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建对比表格
        self.create_comparison_table()
        
        # 保存损失对比图
        loss_fig_path = os.path.join(save_dir, 'loss_comparison.png')
        self.plot_loss_comparison(save_path=loss_fig_path)
        
        # 保存BLEU对比图
        bleu_fig_path = os.path.join(save_dir, 'bleu_comparison.png')
        self.plot_bleu_comparison(save_path=bleu_fig_path)
        
        # 创建消融组件效果分析
        self._analyze_ablation_effects(save_dir)
        
        print(f"\n消融实验分析报告已生成，保存至 {save_dir}")
    
    def _analyze_ablation_effects(self, save_dir):
        """
        分析各个组件对模型性能的影响
        """
        components = {
            'use_multihead': '多头注意力',
            'use_positional_encoding': '位置编码',
            'use_residual': '残差连接',
            'use_layernorm': 'LayerNorm',
            'use_decoder': '解码器'
        }
        
        # 计算每个组件的影响
        component_effects = {}
        
        for comp_key, comp_name in components.items():
            with_comp = []
            without_comp = []
            
            for exp_name, exp_data in self.experiments.items():
                if comp_key in exp_data['hyperparams']:
                    metrics = exp_data['metrics']
                    val_bleu = max(metrics.get('val_bleu', [0])) if 'val_bleu' in metrics else 0
                    
                    if exp_data['hyperparams'][comp_key]:
                        with_comp.append(val_bleu)
                    else:
                        without_comp.append(val_bleu)
            
            # 计算平均影响
            if with_comp and without_comp:
                avg_with = np.mean(with_comp)
                avg_without = np.mean(without_comp)
                effect = avg_with - avg_without
                component_effects[comp_name] = effect
        
        # 绘制组件影响条形图
        if component_effects:
            plt.figure(figsize=(10, 6))
            components = list(component_effects.keys())
            effects = list(component_effects.values())
            
            plt.bar(components, effects, color=['green' if e > 0 else 'red' for e in effects])
            plt.title('各组件对BLEU分数的影响')
            plt.ylabel('BLEU分数提升')
            plt.grid(True, axis='y')
            
            # 添加数值标签
            for i, v in enumerate(effects):
                plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
            
            save_path = os.path.join(save_dir, 'component_effects.png')
            plt.savefig(save_path)
            print(f"组件影响分析图已保存至 {save_path}")

def generate_sample_ablation_configs():
    """
    生成一组示例消融实验配置
    """
    base_config = {
        'batch_size': 64,
        'num_epochs': 20,
        'lr': 0.0001,
        'num_heads': 4,
        'd_model': 128,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
    }
    
    # 生成各种消融实验配置
    ablation_configs = {
        'baseline': {
            **base_config,
            'use_multihead': True,
            'use_positional_encoding': True,
            'use_residual': True,
            'use_layernorm': True,
            'use_decoder': True
        },
        'no_multihead': {
            **base_config,
            'use_multihead': False,
            'use_positional_encoding': True,
            'use_residual': True,
            'use_layernorm': True,
            'use_decoder': True
        },
        'no_position_encoding': {
            **base_config,
            'use_multihead': True,
            'use_positional_encoding': False,
            'use_residual': True,
            'use_layernorm': True,
            'use_decoder': True
        },
        'no_residual': {
            **base_config,
            'use_multihead': True,
            'use_positional_encoding': True,
            'use_residual': False,
            'use_layernorm': True,
            'use_decoder': True
        },
        'no_layernorm': {
            **base_config,
            'use_multihead': True,
            'use_positional_encoding': True,
            'use_residual': True,
            'use_layernorm': False,
            'use_decoder': True
        },
        'encoder_only': {
            **base_config,
            'use_multihead': True,
            'use_positional_encoding': True,
            'use_residual': True,
            'use_layernorm': True,
            'use_decoder': False
        }
    }
    
    return ablation_configs

if __name__ == "__main__":
    # 示例用法
    tracker = ExperimentTracker()
    configs = generate_sample_ablation_configs()
    
    # 保存示例配置
    with open('./ablation_configs.json', 'w') as f:
        json.dump(configs, f, indent=2)
    
    print("已生成示例消融实验配置并保存至 ablation_configs.json")
    print("可使用以下命令运行不同的消融实验：")
    for exp_name, config in configs.items():
        cmd = f"python train_ablation.py --exp_name={exp_name} "
        for key, value in config.items():
            if key.startswith('use_'):
                cmd += f"--{key}={value} "
        print(cmd)