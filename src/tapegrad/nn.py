"""Modules: Linear, Embedding, LayerNorm, causal self-attention, MLP, and a GPT-style model.

Layer layouts mirror ``torch.nn`` (``Linear.weight`` is (out, in)) so weights can be copied
between the two libraries one-to-one; :func:`load_torch_state` does exactly that.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from . import functional as F
from .tensor import Tensor


class Module:
    training = True

    def parameters(self) -> list[Tensor]:
        return [p for _, p in self.named_parameters()]

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        for k, v in vars(self).items():
            name = f"{prefix}{k}"
            if isinstance(v, Tensor) and v.requires_grad:
                yield name, v
            elif isinstance(v, Module):
                yield from v.named_parameters(name + ".")
            elif isinstance(v, list):
                for i, m in enumerate(v):
                    if isinstance(m, Module):
                        yield from m.named_parameters(f"{name}.{i}.")

    def modules(self) -> Iterator[Module]:
        yield self
        for v in vars(self).values():
            if isinstance(v, Module):
                yield from v.modules()
            elif isinstance(v, list):
                for m in v:
                    if isinstance(m, Module):
                        yield from m.modules()

    def train(self, mode: bool = True) -> Module:
        for m in self.modules():
            m.training = mode
        return self

    def eval(self) -> Module:
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def __call__(self, *a, **k):  # type: ignore[no-untyped-def]
        return self.forward(*a, **k)

    def forward(self, *a, **k):  # type: ignore[no-untyped-def]  # pragma: no cover
        raise NotImplementedError


def _param(a: np.ndarray) -> Tensor:
    return Tensor(a.astype(np.float32), requires_grad=True)


class Linear(Module):
    def __init__(self, n_in: int, n_out: int, bias: bool = True,
                 rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng(0)
        k = 1 / math.sqrt(n_in)
        self.weight = _param(rng.uniform(-k, k, (n_out, n_in)))
        self.bias = _param(rng.uniform(-k, k, (n_out,))) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y


class Embedding(Module):
    def __init__(self, n: int, d: int, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng(0)
        self.weight = _param(rng.normal(0, 1, (n, d)))

    def forward(self, idx: np.ndarray) -> Tensor:
        return self.weight[idx]


class LayerNorm(Module):
    def __init__(self, d: int, eps: float = 1e-5, fused: bool = True) -> None:
        self.weight = _param(np.ones(d))
        self.bias = _param(np.zeros(d))
        self.eps = eps
        self.fused = fused

    def forward(self, x: Tensor) -> Tensor:
        fn = F.layer_norm if self.fused else F.layer_norm_composite
        return fn(x, self.weight, self.bias, self.eps)


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    fused: bool = True  # fused softmax / layernorm / cross-entropy vs composite primitives


class CausalSelfAttention(Module):
    def __init__(self, c: GPTConfig, rng: np.random.Generator) -> None:
        self.qkv = Linear(c.n_embd, 3 * c.n_embd, rng=rng)
        self.proj = Linear(c.n_embd, c.n_embd, rng=rng)
        self.h = c.n_head
        self.fused = c.fused
        self.mask = np.triu(np.ones((c.block_size, c.block_size), dtype=bool), k=1)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        hd = d // self.h
        qkv = self.qkv(x).reshape(b, t, 3, self.h, hd).transpose(2, 0, 3, 1, 4)  # 3,b,h,t,hd
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.swapaxes(-1, -2)) * (1.0 / math.sqrt(hd))
        att = att.masked_fill(self.mask[:t, :t], -np.inf)
        att = att.softmax(-1) if self.fused else F.softmax_composite(att, -1)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(b, t, d)
        return self.proj(y)


class MLP(Module):
    def __init__(self, c: GPTConfig, rng: np.random.Generator) -> None:
        self.fc = Linear(c.n_embd, 4 * c.n_embd, rng=rng)
        self.out = Linear(4 * c.n_embd, c.n_embd, rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.fc(x).gelu())


class Block(Module):
    def __init__(self, c: GPTConfig, rng: np.random.Generator) -> None:
        self.ln1 = LayerNorm(c.n_embd, fused=c.fused)
        self.attn = CausalSelfAttention(c, rng)
        self.ln2 = LayerNorm(c.n_embd, fused=c.fused)
        self.mlp = MLP(c, rng)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(Module):
    def __init__(self, c: GPTConfig, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.config = c
        self.tok = Embedding(c.vocab_size, c.n_embd, rng)
        self.pos = Embedding(c.block_size, c.n_embd, rng)
        self.blocks = [Block(c, rng) for _ in range(c.n_layer)]
        self.ln_f = LayerNorm(c.n_embd, fused=c.fused)
        self.head = Linear(c.n_embd, c.vocab_size, bias=False, rng=rng)

    def forward(self, idx: np.ndarray, targets: np.ndarray | None = None
                ) -> tuple[Tensor, Tensor | None]:
        b, t = idx.shape
        x = self.tok(idx) + self.pos(np.arange(t))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        ce = F.cross_entropy if self.config.fused else F.cross_entropy_composite
        return logits, ce(logits, targets)

    def num_params(self) -> int:
        return sum(p.data.size for p in self.parameters())


def load_torch_state(model: Module, state: dict[str, np.ndarray]) -> None:
    """Copy arrays (by matching parameter names) into ``model``; shapes must match exactly."""
    params = dict(model.named_parameters())
    missing = set(params) - set(state)
    extra = set(state) - set(params)
    if missing or extra:
        raise KeyError(f"missing={sorted(missing)} unexpected={sorted(extra)}")
    for k, p in params.items():
        arr = np.asarray(state[k], dtype=p.data.dtype)
        if arr.shape != p.shape:
            raise ValueError(f"{k}: {arr.shape} != {p.shape}")
        p.data = arr.copy()
