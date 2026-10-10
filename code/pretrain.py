"""Pre-training of the PatentChemist encoder–decoder (model.BART).

BART denoising (mix masking, input SMILES augmentation, OneCycleLR lr 2.5e-4,
token-batch 12000, seed 42; data_utils semantics: n_val = max(2000, 1%),
raw-token-length filter <= max_seq_len) on a canonical patent-molecule corpus
(one SMILES per line, --corpus).

Checkpointing:
  - full-state resume.pt (model+opt+sched) every 2000 steps, atomic replace
  - OOM skip-batch guard (cap 200 -> abort)
  - token-lengths npy disk cache (per corpus file)
  - per-epoch last.pt + best.pt by val loss
  - train_log.txt mirror of console

Usage:
  python pretrain.py --corpus corpus.smi --out runs/pretrain
"""
import argparse
import gc
import json
import os
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

from data_utils import DenoiseCollate, PretrainDS
from train_utils import BucketBatchSampler, compute_lengths_par, validate, sample_validity

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(_HERE)  # repo root

ARCH = dict(d_model=768, num_layers=12, num_heads=12, d_hidden=2048)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="one SMILES per line")
    ap.add_argument("--vocab", default=os.path.join(ROOT, "weights", "patentchemist", "vocab.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--token_batch_size", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--mask_scheme", default="mix", choices=["token", "span", "mix"])
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    json.dump({**vars(args), **ARCH},
              open(f"{args.out}/args.json", "w"), indent=2)

    print("loading tokenizer ...")
    tok = ChemformerTokenizer(filename=args.vocab)
    pad_idx = tok["pad"]
    vocab_size = len(tok)
    print("vocab size:", vocab_size, "pad idx:", pad_idx, "mask idx:", tok["mask"])

    print("loading corpus ...")
    with open(args.corpus) as f:
        smiles_all = [ln.strip() for ln in f if ln.strip()]
    print("corpus size:", len(smiles_all))

    # lengths (raw token length) over the FULL corpus, disk-cached per corpus file
    tag = os.path.basename(args.corpus).replace(".smi", "")
    cache = os.path.join(args.out, f"len_{tag}_raw.npy")
    if os.path.exists(cache):
        lens = np.load(cache)
        assert len(lens) == len(smiles_all), f"length cache stale: {len(lens)} vs {len(smiles_all)}"
        print("loaded cached lengths:", cache)
    else:
        print("computing token lengths (parallel, raw semantics) ...")
        lens = compute_lengths_par(args.vocab, smiles_all, workers=args.workers)
        np.save(cache, lens)
        print("lengths cached:", cache)
        gc.collect()
    keep = lens <= args.max_seq_len
    smiles_all = [s for s, k in zip(smiles_all, keep) if k]
    lens = lens[keep]
    print(f"after len<={args.max_seq_len}: {len(smiles_all)}")

    # exact split semantics
    rng = random.Random(args.seed)
    idx = list(range(len(smiles_all)))
    rng.shuffle(idx)
    n_val = max(2000, int(len(smiles_all) * 0.01))
    val_idx = set(idx[:n_val])
    train_idx = [i for i in idx if i not in val_idx]
    train_smi = [smiles_all[i] for i in train_idx]
    val_smi = [smiles_all[i] for i in sorted(val_idx)]
    print("train:", len(train_smi), "val:", len(val_smi))

    tr_lens = lens[np.array(train_idx)]
    train_ds = PretrainDS(train_smi)
    val_ds = PretrainDS(val_smi)
    collate = DenoiseCollate(tok, args.max_seq_len, vocab_size, args.mask_prob,
                             aug=bool(args.aug), mask_scheme=args.mask_scheme)
    collate_val = DenoiseCollate(tok, args.max_seq_len, vocab_size, args.mask_prob,
                                 aug=True, mask_scheme=args.mask_scheme)
    sampler = BucketBatchSampler(tr_lens, args.token_batch_size, shuffle=True, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.workers,
                              collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=True, num_workers=4,
                            collate_fn=collate_val, pin_memory=True)

    hp = ARCH
    print(f"building BART {hp} (random init) ...")
    model = BART(vocabulary_size=vocab_size, dropout=0.1,
                       max_seq_len=args.max_seq_len, pad_token_idx=pad_idx, **hp)
    model = model.to(device)
    model.train()
    print("model params: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    print("steps/epoch:", steps_per_epoch, "total steps:", total_steps, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=total_steps, pct_start=0.05)

    best_val = float("inf")
    resume_pt = os.path.join(args.out, "resume.pt")

    def save_ckpt(sd, path):
        ck = {"state_dict": sd,
              "hyper_parameters": {"vocabulary_size": vocab_size,
                                   **hp, "dropout": 0.1, "max_seq_len": args.max_seq_len,
                                   "pad_token_idx": pad_idx}}
        torch.save(ck, path)

    if args.resume and os.path.exists(resume_pt) and args.resume == resume_pt:
        st = torch.load(resume_pt, map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        best_val = st["best_val"]
        print(f"full-state resume from step {st['step']} epoch {st['epoch']}", flush=True)
        model.to(device); model.train()
    elif args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)
        print("weight-only resume from", args.resume, flush=True)
        model.to(device); model.train()

    def save_resume(epoch, step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "epoch": epoch,
                    "best_val": best_val}, resume_pt + ".tmp")
        os.replace(resume_pt + ".tmp", resume_pt)

    log_path = f"{args.out}/train_log.txt"
    BREAKER = os.path.join(args.out, "CIRCUIT_BREAKER")
    if os.path.exists(BREAKER):
        print("circuit-breaker marker present — refusing to train; manual decision required", flush=True)
        sys.exit(42)
    running, run_tok, t0 = 0.0, 0, time.time()
    step_global = 0
    oom_skips = 0
    start_epoch = 0
    if args.resume and os.path.exists(resume_pt) and args.resume == resume_pt:
        st = torch.load(resume_pt, map_location="cpu")
        step_global = st["step"]; start_epoch = st["epoch"]

    bad_streak = 0  # divergence circuit-breaker: sustained val >> best after warmup -> stop

    training_done = False
    for epoch in range(start_epoch, args.epochs):
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
                opt.step()
                try:
                    sched.step()
                except ValueError as e:
                    # resume-replay makes LOCAL epoch count lag GLOBAL steps; the schedule
                    # exhausting at total_steps == the full intended budget is a CLEAN stop
                    # (otherwise repeated crashes can loop forever). Break -> final save.
                    if "Tried to step" in str(e):
                        print("scheduler exhausted at full global budget -> clean stop",
                              flush=True)
                        training_done = True
                        break
                    raise
            except torch.OutOfMemoryError:
                oom_skips += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"[OOM] skipped batch (total skips {oom_skips})", flush=True)
                if oom_skips >= 200:
                    raise RuntimeError("too many OOM skips, aborting for external restart")
                continue
            running += loss.item() * int((tgt != pad_idx).sum())
            run_tok += int((tgt != pad_idx).sum())
            step_global += 1
            if step_global % 200 == 0:
                el = time.time() - t0
                msg = (f"epoch {epoch} step {step_global}/{total_steps} loss {running/run_tok:.4f} "
                       f"lr {sched.get_last_lr()[0]:.2e} {el:.0f}s {step_global/el:.2f}it/s "
                       f"oom_skips {oom_skips}")
                print(msg, flush=True)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")
                running, run_tok = 0.0, 0
            if step_global % 2000 == 0:
                save_resume(epoch, step_global)
            if step_global % args.val_every == 0:
                vl = validate(model, val_loader, device, pad_idx)
                gv = sample_validity(model, tok, val_smi, device, args.max_seq_len, n=256)
                msg = (f"== val loss {vl:.4f} (best {best_val:.4f}) | greedy validity {gv:.3f} ==")
                print(msg, flush=True)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")
                # divergence circuit-breaker (post-warmup, meaningful best): 4 consecutive
                # vals above 3x best -> freeze training, demand manual decision
                if step_global > 30000 and best_val < 1.0:
                    if vl > 3.0 * best_val:
                        bad_streak += 1
                        print(f"   [circuit-breaker] val {vl:.4f} > 3x best {best_val:.4f} "
                              f"(streak {bad_streak}/4)", flush=True)
                        if bad_streak >= 4:
                            open(BREAKER, "w").write(
                                f"val {vl:.4f} vs best {best_val:.4f} at step {step_global}\n")
                            raise RuntimeError("DIVERGENCE CIRCUIT-BREAKER — stopping for manual decision")
                    else:
                        bad_streak = 0
                if vl < best_val:
                    best_val = vl
                    torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
                    save_ckpt(model.state_dict(), os.path.join(args.out, "best.ckpt"))
                    print("   saved best.pt", flush=True)
                model.train()
        torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
        save_ckpt(model.state_dict(), os.path.join(args.out, "last.ckpt"))
        save_resume(epoch + 1, step_global)
        print(f"-- epoch {epoch} done, saved last.pt --", flush=True)
        with open(log_path, "a") as f:
            f.write(f"-- epoch {epoch} done --\n")
        if training_done:
            print("breaking outer epoch loop (scheduler exhausted at full budget)", flush=True)
            break

    vl = validate(model, val_loader, device, pad_idx)
    gv = sample_validity(model, tok, val_smi, device, args.max_seq_len, n=512)
    print(f"final val loss {vl:.4f} | greedy validity {gv:.3f}")
    with open(log_path, "a") as f:
        f.write(f"final val {vl:.4f} validity {gv:.3f}\n")
    if vl < best_val:
        best_val = vl
        torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        save_ckpt(model.state_dict(), os.path.join(args.out, "best.ckpt"))
    print("DONE. best val:", best_val)
    with open(log_path, "a") as f:
        f.write(f"DONE best_val {best_val:.4f}\n")


if __name__ == "__main__":
    main()
