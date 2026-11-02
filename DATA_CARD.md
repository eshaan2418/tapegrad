# Data card: Tiny Shakespeare

- **Source:** `tinyshakespeare/input.txt` from Andrej Karpathy's char-rnn repository
  (1,115,394 characters, SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`,
  checked by `tapegrad.data.load`).
- **Licence:** the text is a concatenation of works by William Shakespeare (public domain). The
  file is downloaded at run time, not committed.
- **Vocabulary:** 65 characters. Split: first 90% of characters for training, last 10% for
  validation (contiguous, so no window overlaps the boundary except by at most one block).
- **Use here:** a standard small benchmark for checking that an autodiff library can train a
  transformer correctly. The quality of the generated text is not a goal.
