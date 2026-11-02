"""Every op: forward and gradients equal PyTorch autograd (float64) on random shapes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import tapegrad.functional as F
from tapegrad.tensor import Tensor, concat, no_grad, stack, unbroadcast, where

TOL = dict(rtol=1e-7, atol=1e-9)


def check(fn_tg, fn_torch, *arrays, upstream_seed=0, tol=TOL):
    tg_in = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    th_in = [torch.tensor(a.copy(), requires_grad=True) for a in arrays]
    out_tg = fn_tg(*tg_in)
    out_th = fn_torch(*th_in)
    np.testing.assert_allclose(out_tg.data, out_th.detach().numpy(), **tol)
    g = np.random.default_rng(upstream_seed).normal(size=out_th.shape)
    out_tg.backward(g)
    out_th.backward(torch.tensor(g))
    for a, b in zip(tg_in, th_in, strict=True):
        np.testing.assert_allclose(a.grad, b.grad.numpy(), **tol)


def arr(shape, seed=0, lo=-2.0, hi=2.0):
    return np.random.default_rng(seed).uniform(lo, hi, shape)


shapes = hnp.array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=4)
bshapes = hnp.mutually_broadcastable_shapes(num_shapes=2, min_dims=1, max_dims=4, max_side=4)


@settings(max_examples=60, deadline=None)
@given(bshapes, st.sampled_from(["add", "sub", "mul", "div"]), st.integers(0, 10**6))
def test_binary_broadcast(shp, op, seed):
    a, b = arr(shp.input_shapes[0], seed), arr(shp.input_shapes[1], seed + 1)
    if op == "div":
        b = np.sign(b) * (np.abs(b) + 0.5)
    f = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y,
         "mul": lambda x, y: x * y, "div": lambda x, y: x / y}[op]
    check(f, f, a, b)


UNARY = {
    "exp": (lambda x: x.exp(), torch.exp, (-2, 2)),
    "log": (lambda x: x.log(), torch.log, (0.1, 3)),
    "tanh": (lambda x: x.tanh(), torch.tanh, (-3, 3)),
    "sigmoid": (lambda x: x.sigmoid(), torch.sigmoid, (-6, 6)),
    "relu": (lambda x: x.relu(), torch.relu, (-2, 2)),
    "gelu": (lambda x: x.gelu(), lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
             (-3, 3)),
    "neg": (lambda x: -x, lambda x: -x, (-2, 2)),
    "pow3": (lambda x: x ** 3, lambda x: x ** 3, (-2, 2)),
    "sqrt": (lambda x: x.sqrt(), torch.sqrt, (0.1, 4)),
    "rsub": (lambda x: 1.5 - x, lambda x: 1.5 - x, (-2, 2)),
    "rdiv": (lambda x: 2.0 / x, lambda x: 2.0 / x, (0.5, 3)),
}


@settings(max_examples=30, deadline=None)
@given(shapes, st.sampled_from(sorted(UNARY)), st.integers(0, 10**6))
def test_unary(shape, name, seed):
    ftg, fth, (lo, hi) = UNARY[name]
    a = arr(shape, seed, lo, hi)
    if name == "relu":
        a[np.abs(a) < 1e-3] = 0.5  # avoid the kink
    check(ftg, fth, a)


@settings(max_examples=40, deadline=None)
@given(shapes, st.data())
def test_reductions(shape, data):
    nd = len(shape)
    axis = data.draw(st.one_of(st.none(), st.integers(-nd, nd - 1)))
    keep = data.draw(st.booleans())
    a = arr(shape, data.draw(st.integers(0, 10**6)))
    check(lambda x: x.sum(axis, keep), lambda x: x.sum(axis, keep) if axis is not None
          else (x.sum().reshape([1] * nd) if keep else x.sum()), a)
    check(lambda x: x.mean(axis, keep), lambda x: x.mean(axis, keep) if axis is not None
          else (x.mean().reshape([1] * nd) if keep else x.mean()), a)
    if axis is not None:
        check(lambda x: x.var(axis, keep), lambda x: x.var(axis, keepdim=keep, unbiased=False), a)
        check(lambda x: x.max(axis, keep), lambda x: torch.amax(x, axis, keepdim=keep), a)
        check(lambda x: x.softmax(axis), lambda x: torch.softmax(x, axis), a)
        check(lambda x: x.log_softmax(axis), lambda x: torch.log_softmax(x, axis), a)


def test_max_ties_split_like_torch():
    a = np.array([[1.0, 3.0, 3.0], [2.0, 2.0, 2.0]])
    check(lambda x: x.max(1), lambda x: torch.amax(x, 1), a)
    check(lambda x: x.max(), lambda x: torch.amax(x), a)


@settings(max_examples=40, deadline=None)
@given(st.integers(1, 3), st.integers(1, 4), st.integers(1, 4), st.integers(1, 4),
       st.sampled_from(["mm", "bmm", "bcast", "vec_l", "vec_r"]), st.integers(0, 10**6))
def test_matmul(b, m, k, n, kind, seed):
    shapes_ = {"mm": ((m, k), (k, n)), "bmm": ((b, m, k), (b, k, n)),
               "bcast": ((b, 1, m, k), (3, k, n)), "vec_l": ((k,), (b, k, n)),
               "vec_r": ((b, m, k), (k,))}[kind]
    check(lambda x, y: x @ y, lambda x, y: x @ y, arr(shapes_[0], seed), arr(shapes_[1], seed + 7))


def test_shape_ops():
    a = arr((2, 3, 4))
    check(lambda x: x.reshape(4, 6), lambda x: x.reshape(4, 6), a)
    check(lambda x: x.transpose(2, 0, 1), lambda x: x.permute(2, 0, 1), a)
    check(lambda x: x.T, lambda x: x.permute(2, 1, 0), a)
    check(lambda x: x.swapaxes(0, 2), lambda x: x.transpose(0, 2), a)
    check(lambda x: x[1, :, ::2], lambda x: x[1, :, ::2], a)
    idx = np.array([0, 2, 2, 1, 0])  # repeated indices must accumulate
    check(lambda x: x[:, idx], lambda x: x[:, torch.tensor(idx)], a)
    check(lambda x, y: concat([x, y], axis=1), lambda x, y: torch.cat([x, y], 1), a, arr((2, 5, 4)))
    check(lambda x, y: stack([x, y], axis=0), lambda x, y: torch.stack([x, y], 0), a, arr((2, 3, 4)))
    mask = arr((2, 3, 4), 3) > 0
    check(lambda x: x.masked_fill(mask, 0.0), lambda x: x.masked_fill(torch.tensor(mask), 0.0), a)
    check(lambda x, y: where(mask, x, y), lambda x, y: torch.where(torch.tensor(mask), x, y),
          a, arr((2, 3, 4), 5))


@pytest.mark.parametrize("fused", [True, False])
def test_cross_entropy(fused):
    logits = arr((3, 5, 7), 1, -4, 4)
    t = np.random.default_rng(2).integers(0, 7, (3, 5))
    ce = F.cross_entropy if fused else F.cross_entropy_composite
    check(lambda x: ce(x, t),
          lambda x: torch.nn.functional.cross_entropy(x.reshape(-1, 7), torch.tensor(t).reshape(-1)),
          logits)


@pytest.mark.parametrize("fused", [True, False])
def test_layer_norm(fused):
    ln = F.layer_norm if fused else F.layer_norm_composite
    check(lambda x, w, b: ln(x, w, b),
          lambda x, w, b: torch.nn.functional.layer_norm(x, (6,), w, b),
          arr((2, 4, 6)), arr((6,), 1), arr((6,), 2))


def test_fan_out_accumulates_and_graph_is_freed():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    y = x * x + x * 3.0 + x  # x used three times
    y.sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 4)
    assert y._node is None  # tape freed after backward


def test_deep_graph_no_recursion_limit():
    x = Tensor(np.ones(3), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 1e-3
    y.sum().backward()
    np.testing.assert_allclose(x.grad, np.ones(3))


def test_no_grad_records_nothing():
    x = Tensor(np.ones(3), requires_grad=True)
    with no_grad():
        y = x * 2
    assert not y.requires_grad and y._node is None


@settings(max_examples=60, deadline=None)
@given(bshapes)
def test_unbroadcast_inverts_broadcast(shp):
    a_shape, b_shape = shp.input_shapes
    out = np.broadcast_shapes(a_shape, b_shape)
    g = np.ones(out)
    assert unbroadcast(g, a_shape).shape == a_shape
    assert unbroadcast(g, a_shape).sum() == g.sum()


def test_finite_differences_on_composite_function():
    """Independent of torch: central differences on a small MLP-like expression."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 3))
    x = rng.normal(size=(5, 4))

    def f(warr):
        W = Tensor(warr, requires_grad=True)
        out = ((Tensor(x) @ W).tanh() ** 2).mean()
        return out, W

    out, W = f(w)
    out.backward()
    num = np.zeros_like(w)
    eps = 1e-6
    for i in np.ndindex(w.shape):
        wp, wm = w.copy(), w.copy()
        wp[i] += eps
        wm[i] -= eps
        num[i] = (f(wp)[0].item() - f(wm)[0].item()) / (2 * eps)
    np.testing.assert_allclose(W.grad, num, rtol=1e-6, atol=1e-9)
