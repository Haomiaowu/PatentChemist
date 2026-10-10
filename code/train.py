"""Fine-tune BART on the molecular pair data (stage 2).

Loads the pre-training checkpoint (strict load; raises on
hyper-parameter mismatch), fine-tunes with optional random-SMILES
augmentation, teacher-forced CE loss, bf16 AMP, OneCycleLR, and an
OOM skip-batch guard. Saves best.pt (raw state_dict) and best.ckpt
(loaded by generate.py).
"""
import argparse
import os
import pickle
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rdkit import RDLogger

from molbart.utils.tokenizers import ChemformerTokenizer
from model import BART
from data_utils import PairDataset, Collate
from train_utils import BucketBatchSampler, compute_lengths_par, validate

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(_HERE)  # repo root


def load_pretrained(model_path, vocab_size):
    ck = torch.load(model_path, map_location="cpu")
    sd = ck["state_dict"] if "state_dict" in ck else ck
    if any(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    hp = ck["hyper_parameters"]
    model = BART(
        vocabulary_size=hp.get("vocabulary_size", vocab_size),
        d_model=hp["d_model"], num_layers=hp["num_layers"],
        num_heads=hp["num_heads"], d_hidden=hp["d_hidden"],
        dropout=hp.get("dropout", 0.1), max_seq_len=hp.get("max_seq_len", 256),
        pad_token_idx=hp.get("pad_token_idx", 0),
    )
    model.load_state_dict(sd, strict=True)
    print(f"loaded pretrained checkpoint | d{hp['d_model']} L{hp['num_layers']}+{hp['num_layers']} "
          f"h{hp['num_heads']} SwiGLU{hp['d_hidden']} (strict)")
    return model, hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pkl", required=True, help="pair pkl: dict {{'input': [...], 'target': [...]}}")
    ap.add_argument("--val_pkl", required=True)
    ap.add_argument("--vocab", default=os.path.join(ROOT, "weights", "patentchemist", "vocab.json"))
    ap.add_argument("--pretrained", required=True, help="pre-trained ckpt (best.ckpt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--token_batch_size", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--aug_input", type=int, default=1)
    ap.add_argument("--aug_target", type=int, default=1)
    ap.add_argument("--aug_prob", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    print("loading tokenizer ...")
    tok = ChemformerTokenizer(filename=args.vocab)
    pad_idx = tok["pad"]
    print("vocab size:", len(tok), "pad idx:", pad_idx)

    print("loading data ...")
    with open(args.train_pkl, "rb") as f:
        dtr = pickle.load(f)
    with open(args.val_pkl, "rb") as f:
        dva = pickle.load(f)
    tr_in, tr_tg = dtr["input"], dtr["target"]
    va_in, va_tg = dva["input"], dva["target"]
    print("train:", len(tr_in), "val:", len(va_in))

    print("computing lengths (parallel cache) ...")
    li = compute_lengths_par(args.vocab, tr_in, workers=args.workers)
    lt = compute_lengths_par(args.vocab, tr_tg, workers=args.workers)
    tr_lens = (li + lt).astype(np.int64)

    train_ds = PairDataset(tr_in, tr_tg)
    val_ds = PairDataset(va_in, va_tg)
    collate = Collate(tok, args.max_seq_len, train=True, aug_input=bool(args.aug_input),
                      aug_target=bool(args.aug_target), aug_prob=args.aug_prob)
    collate_val = Collate(tok, args.max_seq_len, train=False, aug_input=False, aug_target=False)
    sampler = BucketBatchSampler(tr_lens, args.token_batch_size, shuffle=True, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.workers,
                              collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=True, num_workers=args.workers,
                            collate_fn=collate_val, pin_memory=True)

    model, hp = load_pretrained(args.pretrained, len(tok))
    model = model.to(device)
    model.train()
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    def save_ckpt(sd, path):
        torch.save({"state_dict": sd,
                    "hyper_parameters": {**hp, "dropout": 0.0}}, path)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    print("steps/epoch:", steps_per_epoch, "total steps:", total_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps,
                                                pct_start=0.05)

    best_val = float("inf")
    running, run_tok, t0 = 0.0, 0, time.time()
    step = 0
    oom_skips = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            try:
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch["encoder_input"], batch["encoder_pad_mask"],
                                   batch["decoder_input"], batch["decoder_pad_mask"])
                tgt = batch["target"]
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       tgt.reshape(-1), ignore_index=pad_idx)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                opt.step(); sched.step()
            except torch.OutOfMemoryError:
                oom_skips += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"[OOM] skipped batch at step {step} (total {oom_skips})", flush=True)
                if oom_skips >= 200:
                    raise RuntimeError("too many OOM skips, aborting")
                continue
            running += loss.item() * int((tgt != pad_idx).sum())
            run_tok += int((tgt != pad_idx).sum())
            step += 1
            if step % 200 == 0:
                el = time.time() - t0
                print(f"epoch {epoch} step {step}/{total_steps} loss {running/run_tok:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {el:.0f}s {step/el:.2f}it/s "
                      f"oom {oom_skips}", flush=True)
            if step % args.val_every == 0:
                vl = validate(model, val_loader, device, pad_idx)
                print(f"== val loss {vl:.4f} (best {best_val:.4f}) ==", flush=True)
                if vl < best_val:
                    best_val = vl
                    torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
                    save_ckpt(model.state_dict(), os.path.join(args.out, "best.ckpt"))
                    print("   saved best.pt", flush=True)
                model.train()
        torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
        print(f"-- epoch {epoch} done, saved last.pt --", flush=True)

    vl = validate(model, val_loader, device, pad_idx)
    print(f"final val loss {vl:.4f}")
    if vl < best_val:
        best_val = vl
        torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        save_ckpt(model.state_dict(), os.path.join(args.out, "best.ckpt"))
    print("DONE. best val:", best_val)


if __name__ == "__main__":
    main()
