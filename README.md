# Transformer (编码器-解码器) 机器翻译模型

本项目实现了一个**从零开始**（使用PyTorch）构建的Transformer（编码器-解码器）模型，用于英-法翻译任务，基于轻量级的`opus_books`数据集。

## 功能特点
- 纯PyTorch实现：多头注意力机制、前馈神经网络、残差连接+层归一化、正弦位置编码
- 编码器-解码器架构，包含掩码机制（填充掩码+后续掩码）
- 简单的空格分词器+动态构建词汇表（可配置大小）
- 训练循环：Adam优化器、梯度裁剪、余弦学习率调度
- 检查点保存/加载、损失曲线绘制、解码进行定性示例分析
- 完整的消融实验支持，可测试不同Transformer组件的贡献
- 可复现的随机种子+清晰的命令行接口

## 环境配置与依赖安装

### 系统要求
- **操作系统**：Linux、macOS或Windows
- **Python版本**：3.10或更高
- **硬件要求**：
  - **CPU模式**：至少8GB RAM，训练速度较慢
  - **GPU模式**（推荐）：支持CUDA的NVIDIA GPU，至少8GB GPU内存，训练速度显著提升

### 安装步骤

```bash
# 1) 创建并激活虚拟环境
conda create -n transformer python=3.10 -y
conda activate transformer

# 2) 安装依赖
pip install -r requirements.txt
```

### 依赖库说明

以下是项目使用的主要依赖库及其版本：

```
torch==2.2.2            # PyTorch深度学习框架
torchaudio==2.2.2       # PyTorch音频处理库
torchvision==0.17.2     # PyTorch计算机视觉库
datasets==4.4.1         # Hugging Face数据集加载库
matplotlib>=3.7         # 绘图库，用于损失曲线绘制
tqdm==4.67.1            # 进度条显示
nltk>=3.8.1             # 自然语言处理工具包，用于BLEU评分
numpy>=1.24             # 数值计算库
pandas>=2.0             # 数据处理库
transformers>=4.35      # Hugging Face Transformer库（用于数据集加载）
scikit-learn>=1.3       # 机器学习工具库
PyYAML>=6.0             # YAML配置文件处理
```

## 数据

我们使用Hugging Face的`datasets`库加载`opus_books`英-法翻译数据集，https://huggingface.co/datasets/Helsinki-NLP/opus_books，数据集已保存在data文件夹下。

```python
from datasets import load_dataset
dataset = load_dataset("opus_books", "en-fr")
```

数据集格式示例：`{'id': '0', 'translation': {'en': 'The Wanderer', 'fr': 'Le grand Meaulnes'}}`

## 训练指令

### 1. 初始实验（标准模型）

使用脚本运行（推荐）：
```bash
bash scripts/run.sh
```

或手动运行：
```bash
python -m src.train \
  --epochs 8 \
  --batch_size 64 \
  --lr 3e-4 \
  --d_model 256 \
  --nhead 4 \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --dim_feedforward 512 \
  --max_vocab 20000 \
  --max_len 128 \
  --seed 42 \
  --save_dir results
```

### 2. 对比实验（不同头数）

使用4头注意力：
```bash
python -m src.train_ablation \
  --nhead 4 \
  --experiment_name 4_heads \
  --seed 42
```

使用8头注意力：
```bash
python -m src.train_ablation \
  --nhead 8 \
  --experiment_name 8_heads \
  --seed 42
```

使用单头注意力：
```bash
python -m src.train_ablation \
  --nhead 1 \
  --experiment_name single_head \
  --seed 42
```

### 3. 消融实验

禁用多头注意力（使用单头）：
```bash
python -m src.train_ablation \
  --no_multihead \
  --experiment_name no_multihead \
  --seed 42
```

禁用位置编码：
```bash
python -m src.train_ablation \
  --no_positional_encoding \
  --experiment_name no_positional_encoding \
  --seed 42
```

禁用残差连接：
```bash
python -m src.train_ablation \
  --no_residual \
  --experiment_name no_residual \
  --seed 42
```

禁用层归一化：
```bash
python -m src.train_ablation \
  --no_layernorm \
  --experiment_name no_layernorm \
  --seed 42
```

## 项目结构

```
transformer_enfr/                 # 项目根目录
├── 4head.ipynb                  # 4头注意力模型实验
├── 8head.ipynb                  # 8头注意力模型实验
├── evn-pip.ipynb                # 环境配置和依赖安装
├── nomultihead.ipynb            # 无多头注意力模型实验
├── xiaorongshiyan.ipynb         # 消融实验综合分析
├── src/                         # 源代码目录
│   ├── __init__.py              # Python包初始化文件
│   ├── dataset.py               # 数据集处理模块
│   ├── model.py                 # 标准Transformer模型定义
│   ├── model_ablation.py        # 支持消融实验的Transformer模型
│   ├── train.py                 # 标准训练脚本
│   ├── train_ablation.py        # 支持消融实验的训练脚本
│   ├── utils.py                 # 工具函数（位置编码、掩码生成等）
│   ├── experiment_tracker.py    # 实验跟踪工具
│   ├── generate_missing_plots.py # 生成缺失的图表
│   └── visualization_tools.py   # 可视化工具
├── data/                        # 数据存储目录
│   ├── dataset_dict.json        # 数据集字典信息
│   └── train/                   # 训练数据
├── results/                     # 标准实验结果目录
│   ├── loss_curve.png           # 损失曲线图
│   ├── samples.txt              # 翻译样例
│   └── checkpoint.pt            # 模型检查点
├── results_ablation/            # 消融实验结果目录
│   ├── [experiment_name]_[timestamp]/ # 各消融实验子目录
├── scripts/                     # 脚本目录
│   └── run.sh                   # 运行脚本
├── requirements.txt             # 项目依赖
└── README.md                    # 项目说明文档
```

## 文件功能说明

### Jupyter笔记本文件

- **4head.ipynb**：用于4头注意力模型的实验训练分析
- **8head.ipynb**：用于8头注意力模型的实验训练分析
- **evn-pip.ipynb**：环境配置和依赖安装与执行脚本
- **nomultihead.ipynb**：无多头注意力模型的训练
- **xiaorongshiyan.ipynb**：消融实验训练

### 核心模块文件

- **src/dataset.py**：处理英-法翻译数据集，包括数据加载、预处理、分词和批处理
- **src/model.py**：实现标准Transformer编码器-解码器模型，包含多头注意力、前馈网络等核心组件
- **src/model_ablation.py**：支持消融实验的Transformer模型，可以禁用特定组件（多头注意力、位置编码等）
- **src/train.py**：标准训练流程，包括模型训练、验证、评估和结果保存
- **src/train_ablation.py**：支持消融实验的训练流程，可配置不同实验参数
- **src/utils.py**：提供各种工具函数，如位置编码生成、掩码创建、随机种子设置等

### 结果输出文件

#### 标准实验结果（results/目录）
- **loss_curve.png**：训练和验证损失曲线图
- **samples.txt**：模型生成的翻译样例
- **checkpoint.pt**：保存的模型权重、词汇表和训练参数

#### 消融实验结果（results_ablation/[experiment_name]_[timestamp]/目录）
- **best_model.pt**：验证集上表现最好的模型
- **config.txt**：实验配置参数记录
- **loss_curve.png**：损失曲线图
- **metrics.npz**：评估指标数据
- **samples.txt**：翻译样例
- **test_results.txt**：测试集评估结果

## 复现说明

为确保实验结果可复现，请使用以下设置：

1. **随机种子**：所有实验使用统一的随机种子`--seed 42`
2. **环境一致性**：确保使用requirements.txt中指定的库版本
3. **训练参数**：使用上述指令中的参数进行训练
4. **硬件影响**：在不同硬件上可能略有差异，但总体趋势应一致

## 训练时间估计

- **GPU模式**（推荐）：使用默认参数（batch_size=64，3层编码器/解码器），完整训练约需要1-2小时
- **CPU模式**：相同配置可能需要10-20小时，建议使用更小的batch_size（如16或32）

## 注意事项

- 本项目**不使用**Hugging Face的`transformers`库中的预构建模型，所有核心模块从0搭建
- 分词器基于简单的空格分词，为了简化实现。如有需要，可以替换为更高级的子词分词器
- 在资源受限的环境中，可以通过减小batch_size、模型维度（d_model）或层数来降低内存占用

