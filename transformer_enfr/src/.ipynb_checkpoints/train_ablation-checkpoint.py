import os
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from .utils import set_seed, subsequent_mask, lengths_to_mask, SPECIAL_TOKENS
from .dataset import TranslationDataset, collate_pad
from .model_ablation import TransformerED

def init_nltk():
    try:
        # 尝试访问需要的数据，如果不存在会抛出LookupError
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("下载nltk必要数据...")
        nltk.download('punkt', quiet=True)

def get_args():
    ap = argparse.ArgumentParser()
    # 基本训练参数
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="results_ablation")
    
    # 模型结构参数
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_encoder_layers", type=int, default=3)
    ap.add_argument("--num_decoder_layers", type=int, default=3)
    ap.add_argument("--dim_feedforward", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    
    # 数据参数
    ap.add_argument("--max_vocab", type=int, default=20000)
    ap.add_argument("--max_len", type=int, default=128)
    
    # 消融实验参数
    ap.add_argument("--use_multihead", action="store_true", default=True, help="使用多头注意力")
    ap.add_argument("--no_multihead", action="store_false", dest="use_multihead", help="禁用多头注意力")
    
    ap.add_argument("--use_positional_encoding", action="store_true", default=True, help="使用位置编码")
    ap.add_argument("--no_positional_encoding", action="store_false", dest="use_positional_encoding", help="禁用位置编码")
    
    ap.add_argument("--use_residual", action="store_true", default=True, help="使用残差连接")
    ap.add_argument("--no_residual", action="store_false", dest="use_residual", help="禁用残差连接")
    
    ap.add_argument("--use_layernorm", action="store_true", default=True, help="使用LayerNorm")
    ap.add_argument("--no_layernorm", action="store_false", dest="use_layernorm", help="禁用LayerNorm")
    
    ap.add_argument("--use_decoder", action="store_true", default=True, help="使用解码器")
    ap.add_argument("--no_decoder", action="store_false", dest="use_decoder", help="禁用解码器，仅使用编码器")
    
    ap.add_argument("--experiment_name", type=str, default="", help="实验名称，用于区分不同消融实验")
    
    # 评估参数
    ap.add_argument("--eval_bleu_samples", type=int, default=200, help="计算BLEU时的样本数量")
    ap.add_argument("--best_model_criterion", type=str, default="bleu", choices=["loss", "bleu"], help="选择最佳模型的标准")
    
    return ap.parse_args()

def make_scheduler(optimizer, total_steps):
    return OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='linear'
    )

@torch.no_grad()
def greedy_decode(model, src_ids, src_len, tgt_vocab, max_len=50, device="cpu"):
    model.eval()
    bos_id = tgt_vocab.stoi[SPECIAL_TOKENS["bos"]]
    eos_id = tgt_vocab.stoi[SPECIAL_TOKENS["eos"]]

    B = src_ids.size(0)
    src_pad_mask = ~lengths_to_mask(src_len, max_len=src_ids.size(1))  # True where pad
    memory = model.encode(src_ids.to(device), src_pad_mask.to(device))

    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        tgt_mask = subsequent_mask(ys.size(1)).to(device)  # (1,T,T)
        tgt_pad_mask = torch.zeros((B, ys.size(1)), dtype=torch.bool, device=device)  # no pad inside
        dec = model.decode(ys, memory, tgt_mask, tgt_pad_mask, src_pad_mask.to(device))
        logits = model.generator(dec)  # (B,T,V)
        next_token = logits[:, -1, :].argmax(dim=-1)  # (B,)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_id)
        if finished.all():
            break
    return ys

@torch.no_grad()
def calculate_bleu(model, dataset, data_loader, device="cpu", max_samples=200):
    """
    计算模型在给定数据集上的BLEU分数
    max_samples: 限制计算BLEU的样本数量以加速计算
    """
    model.eval()
    smoothing = SmoothingFunction().method1
    total_bleu = 0.0
    count = 0
    
    for i, batch in enumerate(tqdm(data_loader, desc="Calculating BLEU")):
        src = batch["src_ids"].to(device)
        src_len = batch["src_len"].to(device)
        tgt_out = batch["tgt_out_ids"].to(device)
        
        # 生成预测
        out_ids = greedy_decode(model, src, src_len, dataset.tgt_vocab, max_len=50, device=device)
        
        # 处理每个样本
        for j in range(len(out_ids)):
            # 解码预测
            pred = dataset.tgt_vocab.decode(out_ids[j].tolist())[1:]  # 跳过BOS
            if SPECIAL_TOKENS["eos"] in pred:
                pred = pred[:pred.index(SPECIAL_TOKENS["eos"])]
            
            # 解码目标
            tgt = dataset.tgt_vocab.decode(tgt_out[j].tolist())
            if SPECIAL_TOKENS["eos"] in tgt:
                tgt = tgt[:tgt.index(SPECIAL_TOKENS["eos"])]
            
            # 过滤特殊标记
            pred = [token for token in pred if token not in SPECIAL_TOKENS.values()]
            tgt = [token for token in tgt if token not in SPECIAL_TOKENS.values()]
            
            # 计算BLEU分数
            if len(tgt) > 0 and len(pred) > 0:
                bleu_score = corpus_bleu([[tgt]], [pred], smoothing_function=smoothing)
                total_bleu += bleu_score
                count += 1
            
            # 限制样本数量
            if count >= max_samples:
                break
        
        if count >= max_samples:
            break
    
    avg_bleu = (total_bleu / count * 100) if count > 0 else 0
    return avg_bleu

def save_experiment_config(args, run_dir):
    """保存实验配置到文件"""
    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write("Experiment Configuration\n")
        f.write("========================\n\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

def main():
    args = get_args()
    set_seed(args.seed)
    
    # 初始化nltk数据
    init_nltk()
    
    # 创建实验名称
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        # 根据消融参数自动生成实验名称
        experiment_name = "full"
        if not args.use_multihead:
            experiment_name += "-nohead"
        if not args.use_positional_encoding:
            experiment_name += "-nope"
        if not args.use_residual:
            experiment_name += "-nores"
        if not args.use_layernorm:
            experiment_name += "-nonorm"
        if not args.use_decoder:
            experiment_name += "-noenc"
    
    # 根据当前时间创建子文件夹
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"训练结果将保存在: {run_dir}")
    
    # 保存实验配置
    save_experiment_config(args, run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 打印消融实验信息
    print("\n消融实验设置:")
    print(f"多头注意力: {'启用' if args.use_multihead else '禁用'}")
    print(f"位置编码: {'启用' if args.use_positional_encoding else '禁用'}")
    print(f"残差连接: {'启用' if args.use_residual else '禁用'}")
    print(f"LayerNorm: {'启用' if args.use_layernorm else '禁用'}")
    print(f"解码器: {'启用' if args.use_decoder else '禁用'}")

    # 构建数据集
    print("\n构建数据集...")
    train_ds = TranslationDataset(split="train", max_len=args.max_len, max_vocab=args.max_vocab, 
                                 cache_vocabs=None, seed=args.seed)
    val_ds = TranslationDataset(split="validation", max_len=args.max_len, max_vocab=args.max_vocab,
                               cache_vocabs={"src_vocab": train_ds.src_vocab, "tgt_vocab": train_ds.tgt_vocab}, 
                               seed=args.seed)
    test_ds = TranslationDataset(split="test", max_len=args.max_len, max_vocab=args.max_vocab,
                               cache_vocabs={"src_vocab": train_ds.src_vocab, "tgt_vocab": train_ds.tgt_vocab}, 
                               seed=args.seed)

    pad_id_src = train_ds.src_vocab.stoi[SPECIAL_TOKENS["pad"]]
    pad_id_tgt = val_ds.tgt_vocab.stoi[SPECIAL_TOKENS["pad"]]

    # 创建数据加载器
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_pad(b, pad_id_src, pad_id_tgt))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_pad(b, pad_id_src, pad_id_tgt))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_pad(b, pad_id_src, pad_id_tgt))

    # 创建模型（使用消融参数）
    print("\n创建模型...")
    model = TransformerED(
        train_ds.src_vocab, train_ds.tgt_vocab,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
        use_multihead=args.use_multihead,
        use_positional_encoding=args.use_positional_encoding,
        use_residual=args.use_residual,
        use_layernorm=args.use_layernorm,
        use_decoder=args.use_decoder
    ).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id_tgt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = make_scheduler(optimizer, total_steps)

    # 训练记录
    train_losses, val_losses, val_bleus = [], [], []
    best_val_loss = float('inf')
    best_val_bleu = 0.0
    best_model_path = os.path.join(run_dir, "best_model.pt")

    print("\n开始训练...")
    global_step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            src = batch["src_ids"].to(device)           # (B,S)
            tgt_in = batch["tgt_in_ids"].to(device)     # (B,T)
            tgt_out = batch["tgt_out_ids"].to(device)   # (B,T)
            src_len = batch["src_len"].to(device)
            tgt_len = batch["tgt_len"].to(device)

            # 掩码
            src_pad_mask = ~lengths_to_mask(src_len, max_len=src.size(1)).to(device)  # True where pad
            tgt_pad_mask = ~lengths_to_mask(tgt_len, max_len=tgt_in.size(1)).to(device)
            tgt_sub_mask = subsequent_mask(tgt_in.size(1)).to(device)  # (1,T,T)

            logits = model(src, tgt_in, src_key_padding_mask=src_pad_mask,
                           tgt_key_padding_mask=tgt_pad_mask, tgt_mask=tgt_sub_mask)
            # 计算损失
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
            global_step += 1

        train_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                src = batch["src_ids"].to(device)
                tgt_in = batch["tgt_in_ids"].to(device)
                tgt_out = batch["tgt_out_ids"].to(device)
                src_len = batch["src_len"].to(device)
                tgt_len = batch["tgt_len"].to(device)

                src_pad_mask = ~lengths_to_mask(src_len, max_len=src.size(1)).to(device)
                tgt_pad_mask = ~lengths_to_mask(tgt_len, max_len=tgt_in.size(1)).to(device)
                tgt_sub_mask = subsequent_mask(tgt_in.size(1)).to(device)

                logits = model(src, tgt_in, src_key_padding_mask=src_pad_mask,
                               tgt_key_padding_mask=tgt_pad_mask, tgt_mask=tgt_sub_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
                val_loss_total += loss.item()
        val_loss = val_loss_total / max(1, len(val_loader))
        val_losses.append(val_loss)

        # 计算验证集BLEU分数用于模型选择
        current_val_bleu = calculate_bleu(model, val_ds, val_loader, device=device, max_samples=args.eval_bleu_samples)
        val_bleus.append(current_val_bleu)

        # 每隔2轮输出一次结果，减少IOPub消息频率
        if epoch % 2 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}: train loss {train_loss:.4f} | val loss {val_loss:.4f} | val BLEU {current_val_bleu:.2f}%")
        else:
            # 只记录数据，不输出
            pass
        
        # 根据选择的标准保存最佳模型
        model_updated = False
        if args.best_model_criterion == 'bleu':
            # 使用BLEU分数作为标准，越大越好
            if current_val_bleu > best_val_bleu:
                best_val_bleu = current_val_bleu
                best_val_loss = val_loss  # 同时记录最佳损失
                model_updated = True
        else:
            # 使用损失作为标准，越小越好
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_bleu = current_val_bleu  # 同时记录对应的BLEU分数
                model_updated = True
        
        if model_updated:
            # 保存模型状态
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_val_bleu": best_val_bleu,
                "args": vars(args)
            }, best_model_path)
            print(f"  保存新的最佳模型: val {args.best_model_criterion} {'=' if args.best_model_criterion == 'loss' else '='} {best_val_bleu:.2f}%")

        # 保存一些样本预测结果
        if epoch == args.epochs:
            samples_path = os.path.join(run_dir, "samples.txt")
            with open(samples_path, "w", encoding="utf-8") as f:
                # 随机选择一些样本
                import random
                random_indices = random.sample(range(len(val_ds)), min(10, len(val_ds)))
                
                for i in random_indices:
                    batch = collate_pad([val_ds[i]], pad_id_src, pad_id_tgt)
                    out_ids = greedy_decode(model, batch["src_ids"], batch["src_len"],
                                            val_ds.tgt_vocab, max_len=40, device=device)
                    # decode (skip first bos)
                    pred = val_ds.tgt_vocab.decode(out_ids[0].tolist())[1:]
                    # cut at eos if present
                    if SPECIAL_TOKENS["eos"] in pred:
                        pred = pred[:pred.index(SPECIAL_TOKENS["eos"])]
                    src_text = " ".join(val_ds.src_vocab.decode(batch["src_ids"][0].tolist()))
                    tgt_text = " ".join(val_ds.tgt_vocab.decode(batch["tgt_out_ids"][0].tolist()))
                    # strip special tokens for printing
                    for sp in SPECIAL_TOKENS.values():
                        src_text = src_text.replace(sp, "").strip()
                        tgt_text = tgt_text.replace(sp, "").strip()
                    f.write(f"SRC: {src_text}\n")
                    f.write(f"PRED: {' '.join(pred)}\n")
                    f.write(f"TGT:  {tgt_text}\n")
                    f.write("-"*40 + "\n")

    # 保存损失曲线数据
    import numpy as np
    np.savez(os.path.join(run_dir, "metrics.npz"), 
             train_losses=train_losses,
             val_losses=val_losses,
             val_bleus=val_bleus)
    
    # 生成并保存损失曲线图
    import matplotlib.pyplot as plt
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
    print(f"损失曲线图已保存至: {curve_path}")

    # 加载最佳模型并在测试集上评估
    print(f"\nEvaluating best model on test set...")
    # 重新创建模型
    best_model = TransformerED(
        train_ds.src_vocab, train_ds.tgt_vocab,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
        use_multihead=args.use_multihead,
        use_positional_encoding=args.use_positional_encoding,
        use_residual=args.use_residual,
        use_layernorm=args.use_layernorm,
        use_decoder=args.use_decoder
    ).to(device)
    
    # 加载最佳模型权重
    best_ckpt = torch.load(best_model_path)
    best_model.load_state_dict(best_ckpt["model_state"])
    best_model.eval()
    
    # 在测试集上评估loss
    test_loss_total = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Loss"):
            src = batch["src_ids"].to(device)
            tgt_in = batch["tgt_in_ids"].to(device)
            tgt_out = batch["tgt_out_ids"].to(device)
            src_len = batch["src_len"].to(device)
            tgt_len = batch["tgt_len"].to(device)

            src_pad_mask = ~lengths_to_mask(src_len, max_len=src.size(1)).to(device)
            tgt_pad_mask = ~lengths_to_mask(tgt_len, max_len=tgt_in.size(1)).to(device)
            tgt_sub_mask = subsequent_mask(tgt_in.size(1)).to(device)

            logits = best_model(src, tgt_in, src_key_padding_mask=src_pad_mask,
                           tgt_key_padding_mask=tgt_pad_mask, tgt_mask=tgt_sub_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
            test_loss_total += loss.item()
    
    test_loss = test_loss_total / max(1, len(test_loader))
    print(f"Test loss: {test_loss:.4f}")
    
    # 计算测试集BLEU分数
    print("\nCalculating BLEU score on test set...")
    test_bleu = calculate_bleu(best_model, test_ds, test_loader, device=device, max_samples=args.eval_bleu_samples)
    print(f"Test BLEU score: {test_bleu:.2f}%")
    
    # 保存测试结果
    with open(os.path.join(run_dir, "test_results.txt"), "w") as f:
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Test BLEU score: {test_bleu:.2f}%\n")
        f.write(f"Best validation {args.best_model_criterion}: {best_val_bleu:.2f}%\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\n")

if __name__ == "__main__":
    main()