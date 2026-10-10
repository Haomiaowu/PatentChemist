"""Data pipeline for PatentChemist: tokenization, denoising pre-training and
pair fine-tuning datasets.

- batch_encode: tokenize/pad SMILES into (ids, pad_mask) tensors laid out (T, B)
- PretrainDS + DenoiseCollate: BART-style denoising (token/span/mix masking with
  input SMILES augmentation; target = canonical SMILES)
- PairDataset + Collate: directed input -> target pairs with optional random-SMILES
  augmentation on either side
"""
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem

from pysmilesutils.augment import SMILESAugmenter


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def batch_encode(tokenizer, smiles_list, max_seq_len):
    """Tokenize, pad, convert to id/mask tensors. Returns (id (T,B), pad_mask (T,B) bool)."""
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


class PretrainDS(Dataset):
    def __init__(self, smiles):
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles[i]


class DenoiseCollate:
    """Augment input (permutation noise) + corruption; target = canonical.

    mask_scheme:
      "token" : per-token masking (BERT/BART style, 80/10/10)
      "span"  : contiguous-span masking (BART span corruption, length-preserving)
      "mix"   : 50% token, 50% span (per sample)
    """
    def __init__(self, tokenizer, max_seq_len, vocab_size, mask_prob=0.15,
                 aug=True, mask_scheme="token", span_p=0.20, mean_span=3):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.aug = aug
        self.mask_scheme = mask_scheme
        self.span_p = span_p
        self.mean_span = mean_span
        self.mask_id = tokenizer["mask"]
        self.augmenter = SMILESAugmenter(active=True, augment_prob=1.0, restricted=True)

    def _apply_span(self, ids1d):
        """Length-preserving contiguous-span masking on a single 1D id tensor."""
        T = ids1d.shape[0]
        nonspecial_bool = ids1d >= 6
        nonspecial = nonspecial_bool.nonzero(as_tuple=True)[0]
        n = len(nonspecial)
        if n == 0:
            return ids1d
        ns_set = set(int(x) for x in nonspecial)
        target = int(round(n * self.span_p))
        domask = torch.zeros(T, dtype=torch.bool)
        placed = 0
        guard = 0
        while placed < target and guard < n * 4:
            guard += 1
            sl = max(1, int(np.random.geometric(1.0 / self.mean_span)))
            start = int(nonspecial[int(np.random.randint(n))])
            hi = start
            while sl > 1 and (hi + 1) in ns_set:
                hi += 1; sl -= 1
            p = start
            while p <= hi and not domask[p]:
                domask[p] = True
                placed += 1
                p += 1
                if placed >= target:
                    break
        r2 = torch.rand(T)
        randid = torch.randint(6, self.vocab_size, (T,))
        out = ids1d.clone()
        m = domask & nonspecial_bool
        out = torch.where(m & (r2 < 0.8), torch.full_like(out, self.mask_id), out)
        ridx = (m & (r2 >= 0.8) & (r2 < 0.9)).nonzero(as_tuple=True)[0]
        out[ridx] = randid[ridx]
        return out

    def __call__(self, batch):
        targets = [b for b in batch]                       # canonical
        if self.aug:
            inputs = self.augmenter(list(targets))         # non-canonical noise
        else:
            inputs = list(targets)
        enc_ids, enc_mask = batch_encode(self.tok, inputs, self.max_seq_len)
        dec_ids, dec_mask = batch_encode(self.tok, targets, self.max_seq_len)
        scheme = self.mask_scheme
        if scheme == "token":
            enc_ids = self._token_mask(enc_ids)
        elif scheme == "span":
            T, B = enc_ids.shape
            cols = [self._apply_span(enc_ids[:, b]) for b in range(B)]
            enc_ids = torch.stack(cols, dim=1)
        elif scheme == "mix":
            T, B = enc_ids.shape
            half = B // 2
            if half > 0:
                enc_ids[:, :half] = self._token_mask(enc_ids[:, :half])
            for b in range(half, B):
                enc_ids[:, b] = self._apply_span(enc_ids[:, b])
        return {
            "encoder_input": enc_ids,
            "encoder_pad_mask": enc_mask,
            "decoder_input": dec_ids[:-1, :],
            "decoder_pad_mask": dec_mask[:-1, :],
            "target": dec_ids[1:, :],
            "target_mask": dec_mask[1:, :],
        }

    def _token_mask(self, enc_ids):
        rand = torch.rand(enc_ids.shape)
        nonsep = enc_ids >= 6
        m = (rand < self.mask_prob) & nonsep
        r2 = torch.rand(enc_ids.shape)
        enc_ids = torch.where(m & (r2 < 0.8), torch.full_like(enc_ids, self.mask_id), enc_ids)
        randid = torch.randint(6, self.vocab_size, enc_ids.shape)
        enc_ids = torch.where(m & (r2 >= 0.8) & (r2 < 0.9), randid, enc_ids)
        return enc_ids


class PairDataset(Dataset):
    def __init__(self, inputs: List[str], targets: List[str]):
        assert len(inputs) == len(targets)
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]


class Collate:
    def __init__(self, tokenizer, max_seq_len, train: bool, aug_input: bool, aug_target: bool,
                 aug_prob: float = 1.0):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.train = train
        self.aug_input = aug_input and train
        self.aug_target = aug_target and train
        self.aug = SMILESAugmenter(active=True, augment_prob=aug_prob, restricted=True)

    def __call__(self, batch):
        inputs = [b[0] for b in batch]
        targets = [b[1] for b in batch]
        if self.aug_input:
            inputs = self.aug(inputs)
        if self.aug_target:
            targets = self.aug(targets)
        enc_ids, enc_mask = batch_encode(self.tokenizer, inputs, self.max_seq_len)
        dec_ids, dec_mask = batch_encode(self.tokenizer, targets, self.max_seq_len)
        return {
            "encoder_input": enc_ids,
            "encoder_pad_mask": enc_mask,
            "decoder_input": dec_ids[:-1, :],
            "decoder_pad_mask": dec_mask[:-1, :],
            "target": dec_ids[1:, :],
            "target_mask": dec_mask[1:, :],
            "target_smiles": targets,
        }
