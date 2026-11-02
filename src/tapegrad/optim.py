"""Optimisers matching PyTorch's update rules (SGD with momentum/Nesterov, Adam, AdamW)."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Optimizer:
    def __init__(self, params: list[Tensor], lr: float) -> None:
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:  # pragma: no cover
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float, momentum: float = 0.0,
                 nesterov: bool = False, weight_decay: float = 0.0) -> None:
        super().__init__(params, lr)
        self.mu, self.nesterov, self.wd = momentum, nesterov, weight_decay
        self.buf: list[np.ndarray | None] = [None] * len(self.params)

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data if self.wd else p.grad
            if self.mu:
                b = self.buf[i]
                b = g.copy() if b is None else self.mu * b + g
                self.buf[i] = b
                g = g + self.mu * b if self.nesterov else b
            p.data = p.data - self.lr * g


class Adam(Optimizer):
    """Adam / AdamW. ``decoupled=True`` gives AdamW (weight decay applied to the weights)."""

    def __init__(self, params: list[Tensor], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, decoupled: bool = False) -> None:
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps, self.wd, self.decoupled = eps, weight_decay, decoupled
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd and self.decoupled:
                p.data = p.data * (1 - self.lr * self.wd)
            elif self.wd:
                g = g + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            denom = np.sqrt(self.v[i]) / np.sqrt(bc2) + self.eps
            p.data = (p.data - (self.lr / bc1) * self.m[i] / denom).astype(p.data.dtype)


def AdamW(params: list[Tensor], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
          eps: float = 1e-8, weight_decay: float = 1e-2) -> Adam:
    return Adam(params, lr, betas, eps, weight_decay, decoupled=True)


def clip_grad_norm(params: list[Tensor], max_norm: float) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    total = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads)))
    coef = max_norm / (total + 1e-6)
    if coef < 1:
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * p.grad.dtype.type(coef)
    return total
