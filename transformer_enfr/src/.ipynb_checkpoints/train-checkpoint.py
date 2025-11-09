
import os
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import set_seed, subsequent_mask, lengths_to_mask, SPECIAL_TOKENS
from dataset import TranslationDataset, collate_pad
from model import TransformerED

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_encoder_layers", type=int, default=3)
    ap.add_argument("--num_decoder_layers", type=int, default=3)
    ap.add_argument("--dim_feedforward", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_vocab", type=int, default=20000)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="results")
    return ap.parse_args()

def make_scheduler(optimizer, total_steps):
    # Cosine schedule without warmup for simplicity
    def lr_lambda(step):
        if total_steps <= 0:
            return 1.0
        progress = step / total_steps
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def greedy_decode(model, src_ids, src_len, tgt_vocab, max_len=50, device="cpu"):
    model.eval()
    bos_id = tgt_vocab.stoi[SPECIAL_TOKENS["bos"]]
    eos_id = tgt_vocab.stoi[SPECIAL_TOKENS["eos"]]

    B = src_ids.size(0)
    src_pad_mask = ~lengths_to_mask(src_len, max_len=src_ids.size(1))  # True where pad
    memory = model.encode(src_ids.to(device), src_pad_mask.to(device))

    ys = torch.full((B,1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        tgt_mask = subsequent_mask(ys.size(1)).to(device)  # (1,T,T)
        tgt_pad_mask = torch.zeros((B, ys.size(1)), dtype=torch.bool, device=device)  # no pad inside
        dec = model.decode(ys, memory, tgt_mask, tgt_pad_mask, src_pad_mask.to(device))
        logits = model.generator(dec)  # (B,T,V)
        next_token = logits[:,-1,:].argmax(dim=-1)  # (B,)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_id)
        if finished.all():
            break
    return ys

def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build datasets / vocabs (train split builds vocabs)
    train_ds = TranslationDataset(split="train", max_len=args.max_len, max_vocab=args.max_vocab, cache_vocabs=None)
    val_ds = TranslationDataset(split="validation", max_len=args.max_len, max_vocab=args.max_vocab,
                                cache_vocabs={"src_vocab": train_ds.src_vocab, "tgt_vocab": train_ds.tgt_vocab})

    pad_id_src = train_ds.src_vocab.stoi[SPECIAL_TOKENS["pad"]]
    pad_id_tgt = val_ds.tgt_vocab.stoi[SPECIAL_TOKENS["pad"]]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_pad(b, pad_id_src, pad_id_tgt))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_pad(b, pad_id_src, pad_id_tgt))

    model = TransformerED(train_ds.src_vocab, train_ds.tgt_vocab,
                          d_model=args.d_model,
                          nhead=args.nhead,
                          num_encoder_layers=args.num_encoder_layers,
                          num_decoder_layers=args.num_decoder_layers,
                          dim_feedforward=args.dim_feedforward,
                          dropout=args.dropout,
                          max_len=args.max_len).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id_tgt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = make_scheduler(optimizer, total_steps)

    train_losses, val_losses = [], []

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

            # Masks
            src_pad_mask = ~lengths_to_mask(src_len, max_len=src.size(1)).to(device)  # True where pad
            tgt_pad_mask = ~lengths_to_mask(tgt_len, max_len=tgt_in.size(1)).to(device)
            tgt_sub_mask = subsequent_mask(tgt_in.size(1)).to(device)  # (1,T,T)

            logits = model(src, tgt_in, src_key_padding_mask=src_pad_mask,
                           tgt_key_padding_mask=tgt_pad_mask, tgt_mask=tgt_sub_mask)
            # shift: logits:(B,T,V), tgt_out:(B,T)
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

        # Validation
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

        print(f"Epoch {epoch}: train loss {train_loss:.4f} | val loss {val_loss:.4f}")

    # Save curves
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    os.makedirs(args.save_dir, exist_ok=True)
    curve_path = os.path.join(args.save_dir, "loss_curve.png")
    plt.savefig(curve_path, dpi=200, bbox_inches="tight")

    # Save a few greedy samples
    samples_path = os.path.join(args.save_dir, "samples.txt")
    with open(samples_path, "w", encoding="utf-8") as f:
        for i in range(min(8, len(val_ds))):
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

    # Save checkpoint (model state + vocabs + args)
    ckpt = {
        "model_state": model.state_dict(),
        "src_vocab": {"itos": train_ds.src_vocab.itos},
        "tgt_vocab": {"itos": train_ds.tgt_vocab.itos},
        "args": vars(args),
    }
    torch.save(ckpt, os.path.join(args.save_dir, "checkpoint.pt"))
    print(f"Saved: {curve_path}, {samples_path}, checkpoint.pt")

if __name__ == "__main__":
    main()
