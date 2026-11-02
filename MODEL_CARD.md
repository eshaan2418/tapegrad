# Model card

- **Model:** 4-layer, 4-head, 128-d character-level GPT (818,176 parameters, context 64), trained
  for 2,000 AdamW steps on Tiny Shakespeare with tapegrad and, for reference, PyTorch from the
  same initial weights. Weights are produced by `make train` and not committed.
- **Intended use:** demonstrating and testing the autodiff library (correctness, speed and memory
  relative to PyTorch). It is not a language model for any downstream purpose.
- **Limitations:** tiny model, tiny corpus, 2,000 steps. Samples are pastiche at best and can
  reproduce fragments of the training text.
