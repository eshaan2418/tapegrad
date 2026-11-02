"""PyTorch reference GPT with parameter names identical to :class:`tapegrad.nn.GPT`.

Used only for verification (identical-initialisation training comparison) and the speed
baseline. Requires ``torch`` (a dev/eval extra).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .nn import GPTConfig


class _Attn(nn.Module):
    def __init__(self, c: GPTConfig) -> None:
        super().__init__()
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.h = c.n_head
        self.register_buffer("mask", torch.triu(torch.ones(c.block_size, c.block_size,
                                                           dtype=torch.bool), 1),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        hd = d // self.h
        q, k, v = self.qkv(x).reshape(b, t, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        att = att.masked_fill(self.mask[:t, :t], float("-inf")).softmax(-1)
        return self.proj((att @ v).transpose(1, 2).reshape(b, t, d))


class _MLP(nn.Module):
    def __init__(self, c: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.out = nn.Linear(4 * c.n_embd, c.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(nn.functional.gelu(self.fc(x), approximate="tanh"))


class _Block(nn.Module):
    def __init__(self, c: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = _Attn(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = _MLP(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TorchGPT(nn.Module):
    def __init__(self, c: GPTConfig) -> None:
        super().__init__()
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.ModuleList([_Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):  # type: ignore[no-untyped-def]
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        return logits, nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                   targets.reshape(-1))


def state_from_tapegrad(model) -> dict[str, torch.Tensor]:  # type: ignore[no-untyped-def]
    return {k: torch.tensor(np.array(v.data)) for k, v in model.named_parameters()}
