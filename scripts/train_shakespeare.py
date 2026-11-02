"""Train the char-level GPT with tapegrad or PyTorch (CPU) from the same initial weights.

    python scripts/train_shakespeare.py --impl tapegrad --seed 0 [--composite] [--steps 2000]

Both implementations see identical batches and use AdamW + cosine schedule + grad clipping
with identical hyperparameters. Writes eval/results/train/<run>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import time
from pathlib import Path

import numpy as np

from tapegrad.data import batches, load
from tapegrad.nn import GPT, GPTConfig
from tapegrad.optim import AdamW, clip_grad_norm
from tapegrad.tensor import no_grad


def lr_at(step: int, total: int, base: float, warmup: int = 100) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["tapegrad", "torch"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--composite", action="store_true", help="unfused ops (tapegrad only)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=200)
    a = ap.parse_args()
    train, val, chars = load()
    cfg = GPTConfig(vocab_size=len(chars), block_size=64, n_layer=4, n_head=4, n_embd=128,
                    fused=not a.composite)
    tg = GPT(cfg, seed=a.seed)  # canonical initialisation for both implementations
    lr, wd, bs = 1e-3, 0.1, 32
    val_batches = list(zip(range(20), batches(val, 64, 64, seed=12345), strict=False))
    if a.impl == "torch":
        import torch

        from tapegrad.torch_ref import TorchGPT, state_from_tapegrad

        torch.set_num_threads(a.threads)
        model = TorchGPT(cfg)
        model.load_state_dict(state_from_tapegrad(tg))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        def step_fn(x: np.ndarray, y: np.ndarray, lr_now: float) -> float:
            for g in opt.param_groups:
                g["lr"] = lr_now
            _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            return float(loss)

        def val_fn() -> float:
            with torch.no_grad():
                return float(np.mean([float(model(torch.from_numpy(x),
                                                    torch.from_numpy(y))[1])
                                      for _, (x, y) in val_batches]))
    else:
        topt = AdamW(tg.parameters(), lr=lr, weight_decay=wd)

        def step_fn(x: np.ndarray, y: np.ndarray, lr_now: float) -> float:
            topt.lr = lr_now
            _, loss = tg(x, y)
            topt.zero_grad()
            loss.backward()
            clip_grad_norm(tg.parameters(), 1.0)
            topt.step()
            return loss.item()

        def val_fn() -> float:
            with no_grad():
                return float(np.mean([tg(x, y)[1].item() for _, (x, y) in val_batches]))

    curve, val_curve = [], []
    stream = batches(train, 64, bs, seed=a.seed)
    t0 = time.perf_counter()
    for step in range(a.steps):
        x, y = next(stream)
        curve.append(step_fn(x, y, lr_at(step, a.steps, lr)))
        if (step + 1) % a.eval_every == 0 or step == 0:
            val_curve.append((step + 1, val_fn()))
            print(json.dumps({"step": step + 1, "train": curve[-1], "val": val_curve[-1][1],
                              "elapsed": time.perf_counter() - t0}), flush=True)
    wall = time.perf_counter() - t0
    name = f"{a.impl}{'_composite' if a.composite else ''}_s{a.seed}"
    out = {
        "run": name, "impl": a.impl, "fused": not a.composite, "seed": a.seed, "steps": a.steps,
        "config": cfg.__dict__, "lr": lr, "weight_decay": wd, "batch": bs,
        "train_loss": curve, "val_loss": val_curve, "wall_s": wall,
        "steps_per_s": a.steps / wall, "threads": a.threads,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "params": tg.num_params(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
        "env_threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS",
                                                        "VECLIB_MAXIMUM_THREADS")},
    }
    if a.impl == "tapegrad":
        Path("runs").mkdir(exist_ok=True)
        np.savez(f"runs/{name}.npz", **{k: v.data for k, v in tg.named_parameters()})
    d = Path("eval/results/train")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
