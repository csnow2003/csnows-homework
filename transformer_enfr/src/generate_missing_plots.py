#!/usr/bin/env python3
"""
为现有的消融实验结果生成缺失的图表
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def generate_plots(run_dir):
    """为指定的运行目录生成损失曲线和BLEU分数图表"""
    # 检查metrics.npz文件是否存在
    metrics_path = os.path.join(run_dir, "metrics.npz")
    if not os.path.exists(metrics_path):
        print(f"错误: 在 {run_dir} 中找不到 metrics.npz 文件")
        return False
    
    # 加载metrics数据
    metrics = np.load(metrics_path)
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    val_bleus = metrics['val_bleus']
    
    # 生成图表
    plt.figure(figsize=(10, 6))
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制BLEU分数曲线
    plt.subplot(2, 1, 2)
    plt.plot(val_bleus, label="Validation BLEU", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("BLEU Score (%)")
    plt.title("Validation BLEU Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(run_dir, "loss_curve.png")
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    print(f"损失曲线图已生成并保存至: {curve_path}")
    
    # 清理内存
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description='为现有消融实验结果生成图表')
    parser.add_argument('--run_dir', type=str, help='实验运行目录路径')
    parser.add_argument('--results_dir', type=str, default='results_ablation', 
                        help='消融实验结果根目录')
    parser.add_argument('--all', action='store_true', help='为所有实验目录生成图表')
    
    args = parser.parse_args()
    
    if args.run_dir:
        # 为指定目录生成图表
        generate_plots(args.run_dir)
    elif args.all:
        # 为所有实验目录生成图表
        if not os.path.exists(args.results_dir):
            print(f"错误: 结果目录 {args.results_dir} 不存在")
            return
            
        for exp_dir in os.listdir(args.results_dir):
            run_dir = os.path.join(args.results_dir, exp_dir)
            if os.path.isdir(run_dir):
                print(f"处理目录: {run_dir}")
                generate_plots(run_dir)
    else:
        print("请指定运行目录或使用 --all 参数为所有目录生成图表")


if __name__ == "__main__":
    main()