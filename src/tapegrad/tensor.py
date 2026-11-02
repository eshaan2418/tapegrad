"""Tensor with reverse-mode automatic differentiation over NumPy arrays.

Every differentiable operation is a :class:`Function` with a ``forward`` on raw arrays and a
``backward`` that maps the output gradient to one gradient per input. Calling an op on tensors
records a node (the function, its saved context, its parent tensors) when any input requires a
gradient. ``Tensor.backward`` topologically sorts the recorded graph iteratively (no recursion
limit), accumulates gradients, and frees each node's saved arrays as soon as it has been used,
so peak memory during backward does not hold the whole tape twice.

Broadcasting follows NumPy. Gradients flowing into a broadcast input are summed back to the
input's shape (:func:`unbroadcast`).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

ArrayLike = Any
_grad_enabled = True
DEFAULT_DTYPE = np.float32


@contextlib.contextmanager
def no_grad() -> Iterator[None]:
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


def is_grad_enabled() -> bool:
    return _grad_enabled


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape`` (inverse of NumPy broadcasting)."""
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, (g, s) in enumerate(zip(grad.shape, shape, strict=True))
                 if s == 1 and g != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


class Context:
    __slots__ = ("saved", "meta")

    def __init__(self) -> None:
        self.saved: tuple[Any, ...] = ()
        self.meta: dict[str, Any] = {}

    def save(self, *xs: Any) -> None:
        self.saved = xs


class Function:
    @staticmethod
    def forward(ctx: Context, *args: Any, **kw: Any) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple[np.ndarray | None, ...]:
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def apply(cls, *inputs: Any, **kw: Any) -> Tensor:
        tensors = [x if isinstance(x, Tensor) else None for x in inputs]
        raw = [x.data if isinstance(x, Tensor) else x for x in inputs]
        ctx = Context()
        out = np.asarray(cls.forward(ctx, *raw, **kw))  # 0-d results come back as scalars
        needs = _grad_enabled and any(t is not None and t.requires_grad for t in tensors)
        res = Tensor(out, requires_grad=needs)
        if needs:
            res._node = (cls, ctx, tensors)
        return res


class Tensor:
    __array_priority__ = 1000  # make ndarray + Tensor dispatch to Tensor.__radd__

    def __init__(self, data: ArrayLike, requires_grad: bool = False, dtype: Any = None) -> None:
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data, dtype=dtype)
        if dtype is None and arr.dtype == np.float64 and not isinstance(
            data, (np.ndarray, np.generic)
        ):
            arr = arr.astype(DEFAULT_DTYPE)
        self.data: np.ndarray = arr
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._node: tuple[type[Function], Context, list[Tensor | None]] | None = None

    # ------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data)

    def detach(self) -> Tensor:
        return Tensor(self.data)

    def __repr__(self) -> str:
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------ autograd
    def backward(self, grad: ArrayLike | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("backward() on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            g = np.ones_like(self.data)
        else:
            g = np.asarray(grad, dtype=self.data.dtype).reshape(self.shape)
        order = self._topo()
        grads: dict[int, np.ndarray] = {id(self): g}
        for t in order:
            gt = grads.pop(id(t), None)
            if gt is None:
                continue
            if t._node is None:  # leaf
                t.grad = gt if t.grad is None else t.grad + gt
                continue
            fn, ctx, parents = t._node
            in_grads = fn.backward(ctx, gt)
            t._node = None  # free the tape entry (saved arrays) as soon as it is consumed
            for p, pg in zip(parents, in_grads, strict=True):
                if p is None or pg is None or not p.requires_grad:
                    continue
                if pg.shape != p.shape:
                    pg = unbroadcast(pg, p.shape)
                if pg.dtype != p.data.dtype:
                    pg = pg.astype(p.data.dtype)
                prev = grads.get(id(p))
                grads[id(p)] = pg if prev is None else prev + pg

    def _topo(self) -> list[Tensor]:
        """Reverse topological order of the recorded graph (iterative DFS)."""
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            t, done = stack.pop()
            if done:
                order.append(t)
                continue
            if id(t) in seen:
                continue
            seen.add(id(t))
            stack.append((t, True))
            if t._node is not None:
                for p in t._node[2]:
                    if p is not None and p.requires_grad and id(p) not in seen:
                        stack.append((p, False))
        return order[::-1]

    def zero_grad(self) -> None:
        self.grad = None

    # ------------------------------------------------------------ operators
    def __add__(self, o: ArrayLike) -> Tensor:
        return Add.apply(self, _t(o, self))

    def __radd__(self, o: ArrayLike) -> Tensor:
        return Add.apply(_t(o, self), self)

    def __sub__(self, o: ArrayLike) -> Tensor:
        return Sub.apply(self, _t(o, self))

    def __rsub__(self, o: ArrayLike) -> Tensor:
        return Sub.apply(_t(o, self), self)

    def __mul__(self, o: ArrayLike) -> Tensor:
        return Mul.apply(self, _t(o, self))

    def __rmul__(self, o: ArrayLike) -> Tensor:
        return Mul.apply(_t(o, self), self)

    def __truediv__(self, o: ArrayLike) -> Tensor:
        return Div.apply(self, _t(o, self))

    def __rtruediv__(self, o: ArrayLike) -> Tensor:
        return Div.apply(_t(o, self), self)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __pow__(self, p: float) -> Tensor:
        if isinstance(p, Tensor):
            raise TypeError("tensor exponents are not supported; use exp(log(x) * y)")
        return Pow.apply(self, p=float(p))

    def __matmul__(self, o: Tensor) -> Tensor:
        return MatMul.apply(self, _t(o, self))

    def __getitem__(self, idx: Any) -> Tensor:
        if isinstance(idx, Tensor):
            idx = idx.data
        elif isinstance(idx, tuple):
            idx = tuple(i.data if isinstance(i, Tensor) else i for i in idx)
        return GetItem.apply(self, idx=idx)

    # comparisons produce non-differentiable arrays
    def __gt__(self, o: ArrayLike) -> Tensor:
        return Tensor(self.data > _raw(o))

    def __lt__(self, o: ArrayLike) -> Tensor:
        return Tensor(self.data < _raw(o))

    # ------------------------------------------------------------ methods
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        n = self.data.size if axis is None else int(np.prod([self.shape[a] for a in _axes(axis)]))
        return self.sum(axis, keepdims) * (1.0 / n)

    def var(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Population variance (ddof=0), as used by LayerNorm."""
        mu = self.mean(axis, keepdims=True)
        return ((self - mu) ** 2).mean(axis, keepdims)

    def max(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        return Max.apply(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape: Any) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Reshape.apply(self, shape=shape)

    def transpose(self, *axes: int) -> Tensor:
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return Transpose.apply(self, axes=axes or None)

    def swapaxes(self, a: int, b: int) -> Tensor:
        perm = list(range(self.ndim))
        perm[a], perm[b] = perm[b], perm[a]
        return self.transpose(*perm)

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def exp(self) -> Tensor:
        return Exp.apply(self)

    def log(self) -> Tensor:
        return Log.apply(self)

    def tanh(self) -> Tensor:
        return Tanh.apply(self)

    def relu(self) -> Tensor:
        return Relu.apply(self)

    def sigmoid(self) -> Tensor:
        return Sigmoid.apply(self)

    def sqrt(self) -> Tensor:
        return self ** 0.5

    def gelu(self) -> Tensor:
        return Gelu.apply(self)

    def softmax(self, axis: int = -1) -> Tensor:
        return Softmax.apply(self, axis=axis)

    def log_softmax(self, axis: int = -1) -> Tensor:
        return LogSoftmax.apply(self, axis=axis)

    def masked_fill(self, mask: ArrayLike, value: float) -> Tensor:
        return MaskedFill.apply(self, mask=_raw(mask).astype(bool), value=value)


def _raw(x: ArrayLike) -> np.ndarray:
    return x.data if isinstance(x, Tensor) else np.asarray(x)


def _t(x: ArrayLike, like: Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(np.asarray(x, dtype=like.data.dtype))


def _axes(axis: int | tuple[int, ...]) -> tuple[int, ...]:
    return (axis,) if isinstance(axis, int) else tuple(axis)


def tensor(data: ArrayLike, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    return Tensor(data, requires_grad, dtype)


# ================================================================ elementwise


class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return g, g


class Sub(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return g, -g


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved
        return g * b, g * a


class Div(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save(a, b)
        return a / b

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved
        return g / b, -g * a / (b * b)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        return -a

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        return (-g,)


class Pow(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, p: float) -> np.ndarray:
        ctx.save(a)
        ctx.meta["p"] = p
        return a ** np.asarray(p, dtype=a.dtype)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        (a,) = ctx.saved
        p = ctx.meta["p"]
        return (g * p * a ** np.asarray(p - 1, dtype=a.dtype),)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        out = np.exp(a)
        ctx.save(out)
        return out

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        return (g * ctx.saved[0],)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save(a)
        return np.log(a)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        return (g / ctx.saved[0],)


class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        out = np.tanh(a)
        ctx.save(out)
        return out

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        (y,) = ctx.saved
        return (g * (1 - y * y),)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        out = np.where(a >= 0, 1 / (1 + np.exp(-np.abs(a))), np.exp(-np.abs(a))
                       / (1 + np.exp(-np.abs(a)))).astype(a.dtype)
        ctx.save(out)
        return out

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        (y,) = ctx.saved
        return (g * y * (1 - y),)


class Relu(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save(a > 0)
        return np.maximum(a, 0)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        return (g * ctx.saved[0],)


_GELU_C = np.sqrt(2.0 / np.pi)


class Gelu(Function):
    """tanh approximation (matches ``torch.nn.functional.gelu(approximate='tanh')``)."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        c = a.dtype.type(_GELU_C)
        inner = c * (a + a.dtype.type(0.044715) * a ** 3)
        t = np.tanh(inner)
        ctx.save(a, t)
        return a.dtype.type(0.5) * a * (1 + t)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        a, t = ctx.saved
        c = a.dtype.type(_GELU_C)
        dinner = c * (1 + a.dtype.type(3 * 0.044715) * a * a)
        d = a.dtype.type(0.5) * (1 + t) + a.dtype.type(0.5) * a * (1 - t * t) * dinner
        return (g * d,)


class MaskedFill(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, mask: np.ndarray, value: float) -> np.ndarray:
        ctx.save(mask)
        return np.where(mask, a.dtype.type(value), a)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        return (np.where(ctx.saved[0], 0, g).astype(g.dtype),)


# ================================================================ reductions / shape


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, axis: Any, keepdims: bool) -> np.ndarray:
        ctx.meta.update(shape=a.shape, axis=axis, keepdims=keepdims)
        return np.asarray(a.sum(axis=axis, keepdims=keepdims))

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        shape, axis, keep = ctx.meta["shape"], ctx.meta["axis"], ctx.meta["keepdims"]
        if axis is not None and not keep:
            g = np.expand_dims(g, tuple(a % len(shape) for a in _axes(axis)))
        return (np.broadcast_to(g, shape).copy(),)


class Max(Function):
    """Gradient is split evenly between tied maxima (matches ``torch.amax``)."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, axis: int | None, keepdims: bool) -> np.ndarray:
        out = a.max(axis=axis, keepdims=True)
        mask = a == out
        ctx.save(mask)
        ctx.meta.update(axis=axis, keepdims=keepdims)
        return out if keepdims else (out.reshape(()) if axis is None else out.squeeze(axis))

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        (mask,) = ctx.saved
        axis, keep = ctx.meta["axis"], ctx.meta["keepdims"]
        if not keep:
            g = g.reshape((1,) * mask.ndim) if axis is None else np.expand_dims(g, axis)
        count = mask.sum(axis=axis, keepdims=True)
        return ((mask * g / count).astype(g.dtype),)


class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        ctx.meta["shape"] = a.shape
        return a.reshape(shape)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        return (g.reshape(ctx.meta["shape"]),)


class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, axes: tuple[int, ...] | None) -> np.ndarray:
        axes = tuple(range(a.ndim))[::-1] if axes is None else tuple(x % a.ndim for x in axes)
        ctx.meta["axes"] = axes
        return np.transpose(a, axes)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        return (np.transpose(g, np.argsort(ctx.meta["axes"])),)


class GetItem(Function):
    """Basic and advanced indexing. Backward scatters with ``np.add.at`` so repeated indices
    (e.g. an embedding lookup of the same token twice) accumulate correctly."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, idx: Any) -> np.ndarray:
        ctx.meta.update(shape=a.shape, idx=idx, dtype=a.dtype)
        return np.asarray(a[idx])

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        out = np.zeros(ctx.meta["shape"], dtype=ctx.meta["dtype"])
        np.add.at(out, ctx.meta["idx"], g)
        return (out,)


class Concat(Function):
    @staticmethod
    def forward(ctx: Context, *arrays: np.ndarray, axis: int = 0) -> np.ndarray:
        ctx.meta.update(sizes=[a.shape[axis] for a in arrays], axis=axis)
        return np.concatenate(arrays, axis=axis)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, ...]:
        cuts = np.cumsum(ctx.meta["sizes"])[:-1]
        return tuple(np.split(g, cuts, axis=ctx.meta["axis"]))


def concat(ts: Sequence[Tensor], axis: int = 0) -> Tensor:
    return Concat.apply(*ts, axis=axis)


def stack(ts: Sequence[Tensor], axis: int = 0) -> Tensor:
    return concat([t.reshape(t.shape[:axis] + (1,) + t.shape[axis:]) for t in ts], axis=axis)


def where(cond: ArrayLike, a: Tensor, b: Tensor) -> Tensor:
    c = _raw(cond).astype(bool)
    return a.masked_fill(~c, 0.0) + b.masked_fill(c, 0.0)


# ================================================================ matmul


class MatMul(Function):
    """Batched matmul with broadcasting over leading dims (1-D operands handled as in NumPy)."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save(a, b)
        return a @ b

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved
        a2 = a[None, :] if a.ndim == 1 else a
        b2 = b[:, None] if b.ndim == 1 else b
        g2 = g
        if a.ndim == 1:
            g2 = np.expand_dims(g2, -2)
        if b.ndim == 1:
            g2 = np.expand_dims(g2, -1)
        ga = g2 @ np.swapaxes(b2, -1, -2)
        gb = np.swapaxes(a2, -1, -2) @ g2
        ga = unbroadcast(ga, a2.shape)
        gb = unbroadcast(gb, b2.shape)
        return ga.reshape(a.shape), gb.reshape(b.shape)


# ================================================================ fused numerically stable ops


class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, axis: int) -> np.ndarray:
        e = np.exp(a - a.max(axis=axis, keepdims=True))
        y = e / e.sum(axis=axis, keepdims=True)
        ctx.save(y)
        ctx.meta["axis"] = axis
        return y

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        (y,) = ctx.saved
        ax = ctx.meta["axis"]
        return (y * (g - (g * y).sum(axis=ax, keepdims=True)),)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, axis: int) -> np.ndarray:
        s = a - a.max(axis=axis, keepdims=True)
        out = s - np.log(np.exp(s).sum(axis=axis, keepdims=True))
        ctx.save(out)
        ctx.meta["axis"] = axis
        return out

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray]:
        (out,) = ctx.saved
        ax = ctx.meta["axis"]
        return (g - np.exp(out) * g.sum(axis=ax, keepdims=True),)
