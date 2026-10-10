"""The PatentChemist encoder–decoder.

BART-style seq2seq over SMILES (encode / decode_step / forward interface):
  - RoPE on Q,K in self-attention (enc + dec); cross-attention stays
    position-free (standard enc-dec practice)
  - RMSNorm (pre-norm, no bias, eps 1e-6)
  - SwiGLU feed-forward (d_hidden)
No positional embedding table: RoPE buffers (cos/sin) instead.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dt)


def build_rope(head_dim, max_seq_len, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv)                      # (T, hd/2)
    return freqs.cos(), freqs.sin()                  # each (T, hd/2)


def apply_rope(x, cos, sin):
    """x: (T, B, H, hd) -> rotated. cos/sin: (T, hd/2)."""
    T = x.size(0)
    c = cos[:T].to(x.dtype)                          # (T, hd/2)
    s = sin[:T].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]              # (T,B,H,hd/2)
    o1 = x1 * c.unsqueeze(1).unsqueeze(1) - x2 * s.unsqueeze(1).unsqueeze(1)
    o2 = x1 * s.unsqueeze(1).unsqueeze(1) + x2 * c.unsqueeze(1).unsqueeze(1)
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


class RopeMHA(nn.Module):
    """Multi-head attention with optional RoPE (self-attn) and manual masks.

    Mirrors nn.MultiheadLayer semantics: attn_mask float (T,S) additive,
    key_padding_mask bool (B,S) True=PAD.
    """

    def __init__(self, d_model, num_heads, dropout, use_rope, max_seq_len):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.hd = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.use_rope = use_rope
        if use_rope:
            cos, sin = build_rope(self.hd, max_seq_len)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        for lin in (self.q, self.k, self.v, self.o):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, rope=False):
        # query (T,B,D); key/value (S,B,D)
        T, B, D = query.shape
        S = key.size(0)
        q = self.q(query).view(T, B, self.h, self.hd).permute(1, 2, 0, 3)   # (B,H,T,hd)
        k = self.k(key).view(S, B, self.h, self.hd).permute(1, 2, 0, 3)
        v = self.v(value).view(S, B, self.h, self.hd).permute(1, 2, 0, 3)
        if self.use_rope and rope:
            q = apply_rope(q.permute(2, 0, 1, 3), self.rope_cos, self.rope_sin).permute(1, 2, 0, 3)
            k = apply_rope(k.permute(2, 0, 1, 3), self.rope_cos, self.rope_sin).permute(1, 2, 0, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)  # (B,H,T,S)
        if attn_mask is not None:
            scores = scores + attn_mask.to(scores.dtype)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)                                         # (B,H,T,hd)
        out = out.permute(2, 0, 1, 3).reshape(T, B, D)
        return self.o(out), None


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.w3 = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EncoderLayer(nn.Module):
    def __init__(self, d, h, d_hidden, dropout, max_seq_len):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = RopeMHA(d, h, dropout, use_rope=True, max_seq_len=max_seq_len)
        self.drop1 = nn.Dropout(dropout)
        self.n2 = RMSNorm(d)
        self.ffn = SwiGLU(d, d_hidden)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None, **kw):
        a = self.attn(self.n1(src), self.n1(src), self.n1(src),
                      key_padding_mask=src_key_padding_mask, rope=True)[0]
        x = src + self.drop1(a)
        x = x + self.drop2(self.ffn(self.n2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d, h, d_hidden, dropout, max_seq_len):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.sattn = RopeMHA(d, h, dropout, use_rope=True, max_seq_len=max_seq_len)
        self.drop1 = nn.Dropout(dropout)
        self.n2 = RMSNorm(d)
        self.xattn = RopeMHA(d, h, dropout, use_rope=False, max_seq_len=max_seq_len)
        self.drop2 = nn.Dropout(dropout)
        self.n3 = RMSNorm(d)
        self.ffn = SwiGLU(d, d_hidden)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None,
                tgt_key_padding_mask=None, **kw):
        n = self.n1(tgt)
        a = self.sattn(n, n, n, attn_mask=tgt_mask,
                       key_padding_mask=tgt_key_padding_mask, rope=True)[0]
        x = tgt + self.drop1(a)
        n2 = self.n2(x)
        a = self.xattn(n2, memory, memory, key_padding_mask=memory_key_padding_mask)[0]
        x = x + self.drop2(a)
        x = x + self.drop3(self.ffn(self.n3(x)))
        return x


class BART(nn.Module):
    def __init__(self, vocabulary_size, d_model=512, num_layers=6, num_heads=8,
                 d_hidden=1408, dropout=0.1, max_seq_len=256, pad_token_idx=0, **_ignored):
        super().__init__()
        self.pad_token_idx = pad_token_idx
        self.vocabulary_size = vocabulary_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(vocabulary_size, d_model, padding_idx=pad_token_idx)
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[pad_token_idx].zero_()
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            EncoderLayer(d_model, num_heads, d_hidden, dropout, max_seq_len)
            for _ in range(num_layers))
        self.enc_norm = RMSNorm(d_model)
        self.decoder = nn.ModuleList(
            DecoderLayer(d_model, num_heads, d_hidden, dropout, max_seq_len)
            for _ in range(num_layers))
        self.dec_norm = RMSNorm(d_model)
        self.token_fc = nn.Linear(d_model, vocabulary_size)
        nn.init.zeros_(self.token_fc.bias)

    @staticmethod
    def _causal_mask(sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)

    def _construct_input(self, token_ids):
        return self.dropout(self.emb(token_ids) * math.sqrt(self.d_model))

    def encode(self, encoder_input, encoder_pad_mask):
        x = self._construct_input(encoder_input)
        m = encoder_pad_mask.transpose(0, 1)          # (B,T) True=PAD
        for layer in self.encoder:
            x = layer(x, src_key_padding_mask=m)
        return self.enc_norm(x)

    def decode_step(self, decoder_input, decoder_pad_mask, memory, memory_pad_mask):
        x = self._construct_input(decoder_input)
        dmask = decoder_pad_mask.transpose(0, 1)
        mmask = memory_pad_mask.transpose(0, 1)
        tgt_mask = self._causal_mask(x.size(0), device=x.device)
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask=tgt_mask,
                      memory_key_padding_mask=mmask, tgt_key_padding_mask=dmask)
        h = self.dec_norm(x)
        return self.token_fc(h)

    def forward(self, encoder_input, encoder_pad_mask, decoder_input, decoder_pad_mask):
        memory = self.encode(encoder_input, encoder_pad_mask)
        return self.decode_step(decoder_input, decoder_pad_mask, memory, encoder_pad_mask)
