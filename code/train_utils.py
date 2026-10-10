"""Shared training utilities for PatentChemist: length-bucketed batching,
parallel token-length computation, validation loss, and greedy-decode validity.
"""
import random
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from molbart.utils.tokenizers import ChemformerTokenizer
from pysmilesutils.augment import SMILESAugmenter

from data_utils import batch_encode, canon


class BucketBatchSampler:
    """Length-bucketed batch sampler: sorts by length, forms ~token-budget batches,
    shuffles batch order each epoch."""
    def __init__(self, lengths, token_batch_size, shuffle=True, seed=0):
        order = np.argsort(lengths)
        self.batches = []
        cur, cur_max = [], 0
        for idx in order:
            l = lengths[idx]
            new_max = max(cur_max, l)
            if cur and (len(cur) >= 512 or (len(cur) + 1) * new_max > token_batch_size):
                self.batches.append(cur)
                cur, cur_max = [], 0
                new_max = l
            cur.append(int(idx))
            cur_max = max(cur_max, l)
        if cur:
            self.batches.append(cur)
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            self.rng.shuffle(order)
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


_TOK = None
def _len_worker(init):
    global _TOK
    vocab, = init
    _TOK = ChemformerTokenizer(filename=vocab)
def _len_one(s):
    return len(_TOK.tokenize([s])[0])
def compute_lengths_par(vocab, smiles_list, workers=48):
    with Pool(workers, initializer=_len_worker, initargs=([vocab],)) as p:
        lens = p.map(_len_one, smiles_list, chunksize=4000)
    return np.array(lens, dtype=np.int64)


@torch.no_grad()
def validate(model, loader, device, pad_idx, max_batches=200):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["encoder_input"], batch["encoder_pad_mask"],
                           batch["decoder_input"], batch["decoder_pad_mask"])
        tgt = batch["target"]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), ignore_index=pad_idx, reduction="sum")
        tot_loss += loss.item()
        tot_tok += (tgt != pad_idx).sum().item()
    model.train()
    return tot_loss / max(tot_tok, 1)


@torch.no_grad()
def sample_validity(model, tok, val_smiles, device, max_seq_len, n=256):
    """Greedy decode from augmented input, measure % valid canonical SMILES (health check)."""
    model.eval()
    aug = SMILESAugmenter(active=True, augment_prob=1.0, restricted=True)
    smis = val_smiles[:n]
    inputs = aug(list(smis))
    enc_ids, enc_mask = batch_encode(tok, inputs, max_seq_len)
    enc_ids = enc_ids.to(device); enc_mask = enc_mask.to(device)
    start, end = tok["start"], tok["end"]
    T, B = enc_ids.shape
    mem = model.encode(enc_ids, enc_mask)
    ys = torch.full((1, B), start, dtype=torch.long, device=device)
    ypad = torch.zeros((1, B), dtype=torch.bool, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_seq_len):
        if finished.all():
            break
        logits = model.decode_step(ys, ypad, mem, enc_mask)[-1]
        nxt = logits.argmax(-1)
        nxt = torch.where(finished, torch.full_like(nxt, 0), nxt)
        ys = torch.cat([ys, nxt.unsqueeze(0)], dim=0)
        ypad = torch.cat([ypad, (nxt == 0).unsqueeze(0)], dim=0)
        finished = finished | (nxt == end)
    arr = ys.cpu()
    n_valid = 0
    for j in range(B):
        ids = arr[:, j].tolist()
        try:
            ids = ids[1:ids.index(end)]
        except ValueError:
            ids = ids[1:]
        toks = "".join(tok.decoder_vocabulary[i] for i in ids if i >= 6)
        if canon(toks) is not None:
            n_valid += 1
    model.train()
    return n_valid / max(B, 1)
