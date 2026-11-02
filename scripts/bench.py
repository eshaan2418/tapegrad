"""Speed and memory benchmark (run on an otherwise idle machine).

* steps/s for tapegrad (fused, composite) and PyTorch CPU at 1 and 4 BLAS threads;
* peak traced allocation during one forward+backward for tapegrad fused vs composite
  (tracemalloc sees NumPy's allocations), and with vs without freeing the tape during backward.

Writes eval/results/bench/bench.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np


def speed(impl: str, fused: bool, threads: int, steps: int = 30) -> float:
    code = f"""
import os, time, numpy as np
from tapegrad.nn import GPT, GPTConfig
from tapegrad.optim import AdamW
c = GPTConfig(vocab_size=65, block_size=64, n_layer=4, n_head=4, n_embd=128, fused={fused})
tg = GPT(c)
r = np.random.default_rng(0); x = r.integers(0, 65, (32, 64)); y = r.integers(0, 65, (32, 64))
if "{impl}" == "torch":
    import torch
    torch.set_num_threads({threads})
    from tapegrad.torch_ref import TorchGPT, state_from_tapegrad
    m = TorchGPT(c); m.load_state_dict(state_from_tapegrad(tg))
    o = torch.optim.AdamW(m.parameters(), 1e-3)
    X, Y = torch.from_numpy(x), torch.from_numpy(y)
    def step():
        o.zero_grad(); m(X, Y)[1].backward(); o.step()
else:
    o = AdamW(tg.parameters(), 1e-3)
    def step():
        o.zero_grad(); tg(x, y)[1].backward(); o.step()
for _ in range(3): step()
t = time.perf_counter()
for _ in range({steps}): step()
print({steps} / (time.perf_counter() - t))
"""
    env = {**os.environ, "OPENBLAS_NUM_THREADS": str(threads),
           "VECLIB_MAXIMUM_THREADS": str(threads), "OMP_NUM_THREADS": str(threads)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         check=True)
    return float(out.stdout.strip().splitlines()[-1])


def memory(fused: bool, free_tape: bool) -> dict:
    import tapegrad.tensor as T
    from tapegrad.nn import GPT, GPTConfig

    c = GPTConfig(vocab_size=65, block_size=64, n_layer=4, n_head=4, n_embd=128, fused=fused)
    m = GPT(c)
    r = np.random.default_rng(0)
    x, y = r.integers(0, 65, (32, 64)), r.integers(0, 65, (32, 64))
    keep: list = []
    orig = T.Tensor.backward
    if not free_tape:
        # ablation: keep every node's saved arrays alive until backward finishes
        def backward_keep(self, grad=None):  # type: ignore[no-untyped-def]
            for t in self._topo():
                if t._node is not None:
                    keep.append(t._node)
            orig(self, grad)
        T.Tensor.backward = backward_keep  # type: ignore[method-assign]
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    _, loss = m(x, y)
    after_fwd = tracemalloc.get_traced_memory()[0] - base
    loss.backward()
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    T.Tensor.backward = orig  # type: ignore[method-assign]
    return {"fused": fused, "free_tape": free_tape, "tape_mb_after_forward": after_fwd / 2**20,
            "peak_mb_forward_backward": peak / 2**20}


def main() -> None:
    res: dict = {"speed": [], "memory": []}
    for threads in (1, 4):
        for impl, fused in (("tapegrad", True), ("tapegrad", False), ("torch", True)):
            sps = [speed(impl, fused, threads) for _ in range(3)]
            row = {"impl": impl, "fused": fused, "threads": threads, "steps_per_s": sps,
                   "mean": float(np.mean(sps))}
            res["speed"].append(row)
            print(json.dumps(row), flush=True)
    for fused in (True, False):
        for free in (True, False):
            res["memory"].append(memory(fused, free))
            print(json.dumps(res["memory"][-1]), flush=True)
    res["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip()
    res["thermal"] = subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                                    text=True).stdout.strip()
    d = Path("eval/results/bench")
    d.mkdir(parents=True, exist_ok=True)
    (d / "bench.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"wall {time.perf_counter() - t0:.0f}s")
