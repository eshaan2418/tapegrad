# Design

## Graph and tape
- An op is a `Function` subclass with `forward(ctx, *arrays)` and `backward(ctx, grad)`.
  `Function.apply` unwraps tensors, runs `forward` on raw NumPy arrays, and records a node
  `(function, context, parents)` on the result only if some input requires grad and grad mode
  is on. `no_grad()` therefore costs nothing per op beyond the check.
- `Tensor.backward` computes a reverse topological order with an iterative DFS, because
  recursion fails at about 1,000 nodes in CPython (there is a 5,000-deep test). It
  accumulates gradients in a dict keyed by tensor id, and sets `node = None` as soon as a
  node's backward has run. That drops the node's saved arrays while backward continues, so
  activations are freed progressively instead of all at the end. `scripts/bench.py` measures
  the peak-memory effect against an ablation that keeps every node alive.
- Gradients of broadcast inputs are reduced with `unbroadcast` (sum over added leading axes
  and over axes where the input had size 1). Hypothesis tests every mutually-broadcastable
  shape pair for add/sub/mul/div against PyTorch.

## Fused vs composite ops
Softmax, log-softmax, cross-entropy and LayerNorm each exist twice: as a single `Function`
with a hand-derived backward, and as a composition of primitives (`*_composite`). Both are
checked against PyTorch. The fused forms save fewer intermediates (cross-entropy saves only
the probabilities; LayerNorm saves `xhat` and `1/std`) and run fewer Python-level nodes. The
`GPTConfig.fused` flag switches the whole model so the effect can be measured end to end.

## Making it fast enough to matter
Profiling a training step found two hot spots, both fixed:
1. `a ** 3` in GELU goes through float `np.power`. Writing `a * a * a` made GELU forward
   cheaper (13.2 ms -> 2.9 ms on a 32x64x512 float32 array, 4.6x, measured).
2. Indexing backward used `np.add.at`, which is slow. Basic indices (ints and slices, as in
   `qkv[0]`) cannot repeat targets and now use `out[idx] += g`. Integer lookups along axis 0
   (embeddings) use a sort plus `np.add.reduceat` segment sum. Negative indices are
   normalised first, because `-1` and `n-1` address the same row (a test caught this).
   Together, steps/s went from 10.9 to 18.1.

## Parity with PyTorch
`nn.GPT` uses the same parameter names and layouts as `torch_ref.TorchGPT` (`Linear.weight`
is (out, in)), so weights copy one-to-one. In float64, losses, all parameter gradients, and
30 steps of AdamW with gradient clipping match PyTorch to below 1e-9 (`tests/test_parity.py`).
In float32 over 2,000 steps the trajectories drift apart only through floating-point
reassociation. The README reports the measured gap.
