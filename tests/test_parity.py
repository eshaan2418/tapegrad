"""tapegrad and PyTorch, same weights and batches: identical losses, grads and AdamW trajectory."""

import numpy as np
import pytest
import torch

from tapegrad.nn import GPT, GPTConfig
from tapegrad.optim import SGD, AdamW, clip_grad_norm
from tapegrad.torch_ref import TorchGPT, state_from_tapegrad

CFG = GPTConfig(vocab_size=17, block_size=8, n_layer=2, n_head=2, n_embd=16)


def _pair(fused=True, dtype=np.float64):
    c = GPTConfig(**{**CFG.__dict__, "fused": fused})
    tg = GPT(c, seed=0)
    for p in tg.parameters():
        p.data = p.data.astype(dtype)
    th = TorchGPT(c).to(torch.float64 if dtype == np.float64 else torch.float32)
    th.load_state_dict(state_from_tapegrad(tg), strict=True)
    return tg, th


def _batch(seed, b=3, t=8):
    r = np.random.default_rng(seed)
    x = r.integers(0, CFG.vocab_size, (b, t))
    return x, (x * 3 + 1) % CFG.vocab_size


@pytest.mark.parametrize("fused", [True, False])
def test_forward_backward_match(fused):
    tg, th = _pair(fused)
    x, y = _batch(0)
    _, lt = tg(x, y)
    _, lh = th(torch.tensor(x), torch.tensor(y))
    assert abs(lt.item() - lh.item()) < 1e-12
    lt.backward()
    lh.backward()
    ref = dict(th.named_parameters())
    for name, p in tg.named_parameters():
        np.testing.assert_allclose(p.grad, ref[name].grad.numpy(), rtol=1e-8, atol=1e-12,
                                   err_msg=name)


def test_adamw_training_trajectory_matches():
    tg, th = _pair()
    opt_t = AdamW(tg.parameters(), lr=3e-3, weight_decay=0.1)
    opt_h = torch.optim.AdamW(th.parameters(), lr=3e-3, weight_decay=0.1)
    for step in range(30):
        x, y = _batch(step)
        _, lt = tg(x, y)
        opt_t.zero_grad()
        lt.backward()
        nt = clip_grad_norm(tg.parameters(), 1.0)
        opt_t.step()
        _, lh = th(torch.tensor(x), torch.tensor(y))
        opt_h.zero_grad()
        lh.backward()
        nh = float(torch.nn.utils.clip_grad_norm_(th.parameters(), 1.0))
        opt_h.step()
        assert abs(lt.item() - lh.item()) < 1e-9, step
        assert abs(nt - nh) < 1e-8
    ref = dict(th.named_parameters())
    for name, p in tg.named_parameters():
        np.testing.assert_allclose(p.data, ref[name].detach().numpy(), rtol=1e-7, atol=1e-10)


def test_sgd_nesterov_matches():
    tg, th = _pair()
    o1 = SGD(tg.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-3)
    o2 = torch.optim.SGD(th.parameters(), lr=0.05, momentum=0.9, nesterov=True,
                         weight_decay=1e-3)
    for step in range(10):
        x, y = _batch(step)
        _, lt = tg(x, y)
        o1.zero_grad()
        lt.backward()
        o1.step()
        _, lh = th(torch.tensor(x), torch.tensor(y))
        o2.zero_grad()
        lh.backward()
        o2.step()
        assert abs(lt.item() - lh.item()) < 1e-10


def test_float32_close():
    tg, th = _pair(dtype=np.float32)
    x, y = _batch(1)
    _, lt = tg(x, y)
    _, lh = th(torch.tensor(x), torch.tensor(y))
    assert abs(lt.item() - lh.item()) < 1e-5
