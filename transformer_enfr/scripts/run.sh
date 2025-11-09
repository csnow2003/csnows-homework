#!/usr/bin/env bash
set -e

# 确保Python可以找到src目录中的模块
cd "$(dirname "$0")/.."

python -m src.train   --epochs 8   --batch_size 64   --lr 3e-4   --d_model 256   --nhead 4   --num_encoder_layers 3   --num_decoder_layers 3   --dim_feedforward 512   --max_vocab 20000   --max_len 128   --seed 42
