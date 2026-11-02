"""Fused, numerically stable ops with hand-written backward passes, and their composite
equivalents (built from primitive ops) for the fusion ablation."""

from __future__ import annotations

import numpy as np

from .tensor import Context, Function, Tensor, _raw


class CrossEntropy(Function):
    """Mean cross-entropy over rows of ``logits`` (..., C) with integer ``targets`` (...).

    Saves only the softmax probabilities; backward is ``(softmax - onehot) / N``.
    """

    @staticmethod
    def forward(ctx: Context, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        x = logits.reshape(-1, logits.shape[-1])
        t = targets.reshape(-1)
        s = x - x.max(axis=1, keepdims=True)
        lse = np.log(np.exp(s).sum(axis=1, keepdims=True))
        logp = s - lse
        n = x.shape[0]
        loss = -logp[np.arange(n), t].mean()
        ctx.save(np.exp(logp), t)
        ctx.meta["shape"] = logits.shape
        return np.asarray(loss, dtype=logits.dtype)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, None]:
        p, t = ctx.saved
        n = p.shape[0]
        d = p.copy()
        d[np.arange(n), t] -= 1
        d *= g / n
        return d.reshape(ctx.meta["shape"]), None


def cross_entropy(logits: Tensor, targets: Tensor | np.ndarray) -> Tensor:
    return CrossEntropy.apply(logits, _raw(targets).astype(np.int64))


def cross_entropy_composite(logits: Tensor, targets: Tensor | np.ndarray) -> Tensor:
    t = _raw(targets).astype(np.int64).reshape(-1)
    lp = logits.reshape(-1, logits.shape[-1]).log_softmax(-1)
    return -(lp[np.arange(len(t)), t]).mean()


class LayerNormFn(Function):
    """LayerNorm over the last axis with affine weight and bias (fused forward/backward)."""

    @staticmethod
    def forward(ctx: Context, x: np.ndarray, w: np.ndarray, b: np.ndarray,
                eps: float) -> np.ndarray:
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + x.dtype.type(eps))
        xhat = xc * rstd
        ctx.save(xhat, rstd, w)
        return xhat * w + b

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xhat, rstd, w = ctx.saved
        lead = tuple(range(g.ndim - 1))
        gw = (g * xhat).sum(axis=lead)
        gb = g.sum(axis=lead)
        gx_hat = g * w
        n = xhat.shape[-1]
        gx = rstd / n * (n * gx_hat - gx_hat.sum(-1, keepdims=True)
                         - xhat * (gx_hat * xhat).sum(-1, keepdims=True))
        return gx, gw, gb


def layer_norm(x: Tensor, w: Tensor, b: Tensor, eps: float = 1e-5) -> Tensor:
    return LayerNormFn.apply(x, w, b, eps=eps)


def layer_norm_composite(x: Tensor, w: Tensor, b: Tensor, eps: float = 1e-5) -> Tensor:
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    return xc * ((var + eps) ** -0.5) * w + b


def softmax_composite(x: Tensor, axis: int = -1) -> Tensor:
    e = (x - Tensor(x.data.max(axis=axis, keepdims=True))).exp()
    return e / e.sum(axis, keepdims=True)


def dropout(x: Tensor, p: float, training: bool, rng: np.random.Generator) -> Tensor:
    if not training or p == 0:
        return x
    keep = (rng.random(x.shape) >= p).astype(x.dtype) / x.dtype.type(1 - p)
    return x * Tensor(keep)
