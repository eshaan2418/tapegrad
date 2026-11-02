# Deep dive: questions and answers

### 1. What exactly does "matches PyTorch" mean here, and why should anyone believe it?
There are three levels of evidence. (a) Op-level: every primitive and fused op is compared with PyTorch autograd in float64, on Hypothesis-generated shapes including all broadcastable shape pairs, at rtol 1e-7 (`tests/test_grad.py`). (b) Model-level: a 2-layer GPT with identical weights gives the same loss to 1e-12, the same gradient for every parameter to 1e-8 relative, and the same parameters after 30 AdamW steps with clipping to 1e-7 relative (`tests/test_parity.py`). (c) Training-level: 2,000 float32 steps on Tiny Shakespeare from identical weights and batches stay within 9.5e-7 of PyTorch's loss at every step, with 41-44% of steps bit-identical. Each level would catch a different class of bug. Op tests miss composition errors, the model test misses optimiser drift, and the training run misses nothing large but cannot localise a bug.

### 2. Why does the float32 gap not grow over 2,000 steps? Isn't training chaotic?
Small perturbations can grow in principle. Here they are re-injected at the 1-ULP level each step rather than compounded, because both libraries compute the same function and differ only in summation order. The measured gap is largest at steps 1-3, when the loss is ~4.2 and one ULP is 4.8e-7. After that it sits at 1-4 ULPs. With a larger learning rate or a longer run I would expect eventual divergence. The parity claim is for this regime.

### 3. Why reduce broadcast gradients in the engine rather than in each op?
An op's backward naturally produces a gradient at the broadcast output shape. Reducing it to each input's shape is the same operation for every elementwise op: sum over added leading axes, then over axes where the input had size 1. Doing it once in `Tensor.backward` (`unbroadcast`) means one tested implementation instead of a copy per op, and a property test checks it for arbitrary shape pairs.

### 4. How does `backward` handle a tensor used several times (fan-out)?
Gradients are accumulated in a dictionary keyed by tensor id, and a node's backward runs only once all its consumers have contributed. That is guaranteed by processing nodes in reverse topological order. `x*x + 3x + x` gives `2x + 4` (test). Leaves accumulate into `.grad` across backward calls, as in PyTorch.

### 5. Why an iterative topological sort?
A recursive DFS hits CPython's recursion limit (~1,000 frames) on deep graphs. Long unrolled sequences or many layers exceed that easily. The iterative version uses an explicit stack with "children done" markers. A 5,000-op chain is tested.

### 6. What do the fused ops buy?
Memory: the tape after forward is 173 MB fused vs 217 MB composite (−20%), and peak RSS during training is 361 vs 515 MB. Cross-entropy fused saves only the softmax probabilities, where composite saves log-softmax outputs plus indexing metadata. LayerNorm fused saves `x̂` and `1/σ` instead of every intermediate. Graph size: 127 vs 234 nodes per step. Speed: within measurement noise on this machine, because the matmuls dominate.

### 7. Freeing the tape during backward barely changed peak memory (197 vs 201 MB). Is the feature useless?
For the peak, nearly. The peak occurs at the start of backward, when almost all activations are still needed: the last layers' backward needs their saved inputs, and the first layers' activations are still waiting. Freeing helps steady state, since memory comes back as backward proceeds, which matters if something else allocates during backward (e.g. optimiser state on first use). To actually lower the peak you need activation checkpointing (recomputing forward segments during backward) or smaller saved tensors (e.g. saving the ReLU mask as bits). Neither is implemented.

### 8. Where did the speedup from 10.9 to 18.1 steps/s come from?
cProfile, not guesswork. GELU's `a ** 3` takes NumPy's general float power path. `a*a*a` is 4.6x faster on a 32×64×512 array (13.2 → 2.9 ms). Indexing backward used `np.add.at`, which is unbuffered and slow. Basic indices (ints and slices, e.g. `qkv[0]`) cannot hit the same element twice, so `out[idx] += g` is correct and fast. Integer lookups along axis 0 (embeddings) use a stable sort plus `np.add.reduceat` segment sum.

### 9. What bug did the segment-sum optimisation introduce, and how was it caught?
Negative indices. `x[[-1, 6]]` on a 7-row tensor addresses row 6 twice. The segment sum treated -1 and 6 as different keys, and the final assignment `out[uniq] = sums` let one overwrite the other. A dedicated test with repeated and negative indices failed. The fix normalises indices with `% n` first.

### 10. Why is tapegrad 0.89x PyTorch single-threaded but 0.56x with 4 threads?
At 1 thread both are dominated by the same BLAS matmuls, so the gap is mostly Python overhead per node. With 4 threads, BLAS speeds up in both, but NumPy's elementwise ops (GELU, softmax, LayerNorm arithmetic, Adam updates) stay single-threaded, while PyTorch's CPU kernels parallelise them. Caveat: these benchmarks ran on a loaded machine (the min-max ranges are wide), so only the coarse ratios are reliable.

### 11. How were the PyTorch comparisons made fair?
Same initial weights (copied by name), same batch sequence (one NumPy RNG stream drives both), same optimiser hyperparameters and update rule (AdamW with decoupled decay, bias correction as in `torch.optim.AdamW`), same clipping, same dtype, and the same thread count for BLAS and `torch.set_num_threads`.

### 12. What isn't supported, and what would be hardest to add?
Higher-order gradients (backward is not recorded as a graph), `retain_graph`, GPU execution, in-place ops with version checks, and mixed precision. Higher-order gradients would be the deepest change: every backward must be written in terms of Tensor ops rather than raw arrays, so the backward pass itself builds a graph.

### 13. How would this scale with 100x the compute or model size?
Python overhead per node is constant, so larger tensors amortise it and the gap to PyTorch shrinks toward BLAS parity at 1 thread. The limits are memory (no checkpointing) and the single-threaded elementwise kernels. The next steps would be a lazy graph that fuses elementwise chains into one NumPy expression or a generated kernel, a buffer pool to cut allocation churn, and a Metal or MLX backend for the matmuls.

### 14. Why Tiny Shakespeare and not a harder benchmark?
The goal is to test the library, not language modelling. A character-level corpus with a 65-symbol vocabulary gives a real transformer training signal in minutes on a CPU, and it is the standard sanity task for small GPT implementations. That makes it easy for others to compare.

### 15. What would convince you the library is wrong despite these tests?
A divergence that grows with step count in float64. That would indicate a real difference in the maths, since float64 removes rounding as an explanation. The float64 trajectory test covers 30 steps on a small model. A longer float64 run on the full model would be the next check.
