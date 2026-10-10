#!/usr/bin/env python3
"""PatentChemist: generate analogues from seed SMILES.

Given one or more seed SMILES, generate up to N unique, valid, canonical
analogues with the fine-tuned model.

Usage:
    python generate.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --n 1000
    python generate.py --smiles-file seeds.smi --n 1000 --out analogs.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
from rdkit import Chem, RDLogger
from pysmilesutils.augment import SMILESAugmenter
from molbart.utils.tokenizers import ChemformerTokenizer
from model import BART

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(_HERE)
START, END, PAD = 2, 3, 0  # ^, &, <PAD>


def canon(s: str):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def load_model(ckpt_path: str, vocab_path: str, device: str = "cuda"):
    """Load the fine-tuned checkpoint ({state_dict, hyper_parameters})."""
    tok = ChemformerTokenizer(filename=vocab_path)
    ck = torch.load(ckpt_path, map_location="cpu")
    sd, hp = ck["state_dict"], ck["hyper_parameters"]
    if any(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    model = BART(
        vocabulary_size=hp.get("vocabulary_size", len(tok)),
        d_model=hp.get("d_model", 768), num_layers=hp.get("num_layers", 12),
        num_heads=hp.get("num_heads", 12), d_hidden=hp.get("d_hidden", 2048),
        dropout=hp.get("dropout", 0.0),
        max_seq_len=hp.get("max_seq_len", 256),
        pad_token_idx=hp.get("pad_token_idx", PAD),
    )
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    return model, tok


def batch_encode(tokenizer, smiles_list, max_seq_len):
    tokens = tokenizer.tokenize(smiles_list)
    pad_tok = tokenizer.special_tokens["pad"]
    max_len = min(max(len(t) for t in tokens), max_seq_len)
    padded, mask = [], []
    for ts in tokens:
        ts = ts[:max_seq_len]
        n = len(ts)
        padded.append(ts + [pad_tok] * (max_len - n))
        mask.append([0] * n + [1] * (max_len - n))
    id_data = tokenizer.convert_tokens_to_ids(padded)
    id_tensor = torch.stack(id_data).transpose(0, 1)
    mask_tensor = torch.tensor(mask, dtype=torch.bool).transpose(0, 1)
    return id_tensor, mask_tensor


@torch.no_grad()
def _sample(model, tok, enc_ids, enc_mask, n_per, temperature, top_p, max_len, device):
    """Batched multinomial sampling. enc_ids: (T, B). Returns list of token-id lists."""
    enc_ids, enc_mask = enc_ids.to(device), enc_mask.to(device)
    mem = model.encode(enc_ids, enc_mask)
    mem = mem.repeat_interleave(n_per, dim=1)
    mem_pad = enc_mask.repeat_interleave(n_per, dim=1)
    BN = mem.size(1)
    ys = torch.full((1, BN), START, dtype=torch.long, device=device)
    ypad = torch.zeros((1, BN), dtype=torch.bool, device=device)
    finished = torch.zeros(BN, dtype=torch.bool, device=device)
    for _ in range(max_len):
        if finished.all():
            break
        logits = model.decode_step(ys, ypad, mem, mem_pad)[-1]
        logits = logits / max(temperature, 1e-5)
        if top_p and 0 < top_p < 1.0:
            srt_logits, srt_idx = torch.sort(logits, descending=True)
            srt_probs = torch.softmax(srt_logits, dim=-1)
            cum = srt_probs.cumsum(dim=-1)
            rm = cum > top_p
            rm[..., 1:] = rm[..., :-1].clone()
            rm[..., 0] = False
            srt_logits = srt_logits.masked_fill(rm, float("-inf"))
            logits = torch.zeros_like(logits).scatter(1, srt_idx, srt_logits)
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).squeeze(1)
        nxt = torch.where(finished, torch.full_like(nxt, PAD), nxt)
        ys = torch.cat([ys, nxt.unsqueeze(0)], dim=0)
        ypad = torch.cat([ypad, (nxt == PAD).unsqueeze(0)], dim=0)
        finished = finished | (nxt == END)
        if finished.all():
            break
    out = []
    arr = ys.cpu()
    for j in range(BN):
        ids = arr[:, j].tolist()
        try:
            ids = ids[1:ids.index(END)]
        except ValueError:
            ids = ids[1:]
        out.append(ids)
    return out


def _decode_smiles(tok, ids_list):
    smiles = []
    for ids in ids_list:
        toks = [tok.decoder_vocabulary[i] for i in ids if i >= 6]
        smiles.append("".join(toks))
    return smiles


def generate(model, tok, seed, n, device="cuda", n_aug=12, temperature=1.0,
             top_p=0.9, max_len=160, ladder=(3, 6, 12, 24), return_meta=False):
    """Return up to n distinct valid canonical SMILES generated from `seed`."""
    aug = SMILESAugmenter(active=True, augment_prob=1.0, restricted=True)
    canon_seed = canon(seed)
    if not canon_seed:
        raise ValueError(f"invalid seed SMILES: {seed}")
    inputs = [canon_seed] + [s for s in aug([canon_seed] * n_aug) if s]
    inputs = list(dict.fromkeys(inputs))[: (n_aug + 1)]
    enc_ids, enc_mask = batch_encode(tok, inputs, max_len)

    seen, results = set(), []
    n_raw = n_valid = 0
    for over in ladder:
        per_input = max(1, (n * over) // len(inputs))
        ids_list = _sample(model, tok, enc_ids, enc_mask, per_input,
                           temperature, top_p, max_len, device)
        raw = _decode_smiles(tok, ids_list)
        n_raw += len(raw)
        for c in (canon(s) for s in raw):
            if c is not None:
                n_valid += 1
            if c is None or c == canon_seed or c in seen:
                continue
            seen.add(c)
            results.append(c)
        if len(results) >= n:
            break
    if return_meta:
        return results, {"n_raw": n_raw, "n_valid": n_valid, "n_unique": len(results)}
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "weights", "patentchemist", "best.ckpt"))
    ap.add_argument("--vocab", default=os.path.join(ROOT, "weights", "patentchemist", "vocab.json"))
    ap.add_argument("--smiles", help="single seed SMILES")
    ap.add_argument("--smiles-file", help="file with one seed SMILES per line")
    ap.add_argument("--n", type=int, default=1000, help="target number of unique molecules per seed")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--aug", type=int, default=12, help="number of augmented input SMILES")
    ap.add_argument("--ladder", default="3,6,12,24", help="oversampling ladder (raw budget = n x over)")
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="", help="optional output CSV (seed, rank, smiles); default prints to stdout")
    args = ap.parse_args()

    if not args.smiles and not args.smiles_file:
        ap.error("provide --smiles or --smiles-file")
    seeds = [args.smiles] if args.smiles else [ln.strip() for ln in open(args.smiles_file) if ln.strip()]

    torch.manual_seed(args.seed)
    model, tok = load_model(args.ckpt, args.vocab, args.device)
    ladder = tuple(int(x) for x in args.ladder.split(","))

    rows = []
    for si, smi in enumerate(seeds):
        t0 = time.time()
        res, meta = generate(model, tok, smi, args.n, device=args.device,
                             n_aug=args.aug, temperature=args.temperature,
                             top_p=args.top_p, max_len=args.max_len,
                             ladder=ladder, return_meta=True)
        print(f"[{si + 1}/{len(seeds)}] {len(res)} unique molecules "
              f"(raw {meta['n_raw']}, valid {meta['n_valid']}, {time.time() - t0:.0f}s)", flush=True)
        rows.extend({"seed": smi, "rank": i + 1, "smiles": s} for i, s in enumerate(res))

    if args.out:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print("saved ->", args.out)
    else:
        print("\n".join(r["smiles"] for r in rows))


if __name__ == "__main__":
    main()
