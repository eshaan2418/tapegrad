"""tapegrad CLI: sample text from a trained character-level GPT.

    tapegrad sample --weights runs/tapegrad_s0.npz --prompt "ROMEO:" --tokens 300
"""

from __future__ import annotations

import argparse

import numpy as np

from .data import load
from .nn import GPT, GPTConfig, load_torch_state
from .tensor import no_grad


def sample(model: GPT, ids: list[int], n: int, temperature: float, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    bs = model.config.block_size
    out = list(ids)
    with no_grad():
        for _ in range(n):
            ctx = np.array([out[-bs:]])
            logits, _ = model(ctx)
            z = logits.data[0, -1].astype(np.float64) / temperature
            p = np.exp(z - z.max())
            out.append(int(rng.choice(len(p), p=p / p.sum())))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tapegrad")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--weights", default="runs/tapegrad_s0.npz")
    s.add_argument("--prompt", default="ROMEO:")
    s.add_argument("--tokens", type=int, default=300)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    _, _, chars = load()
    stoi = {c: i for i, c in enumerate(chars)}
    model = GPT(GPTConfig(vocab_size=len(chars), block_size=64, n_layer=4, n_head=4,
                          n_embd=128))
    w = np.load(a.weights)
    load_torch_state(model, {k: w[k] for k in w.files})
    out = sample(model, [stoi[c] for c in a.prompt], a.tokens, a.temperature, a.seed)
    print("".join(chars[i] for i in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
