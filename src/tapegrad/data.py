"""Tiny Shakespeare (character level): download with checksum, 90/10 split by position."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def load(path: str | Path = "data/tinyshakespeare.txt") -> tuple[np.ndarray, np.ndarray, str]:
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, p)
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise SystemExit("tinyshakespeare checksum mismatch")
    text = raw.decode("utf-8")
    chars = "".join(sorted(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], chars


def batches(ids: np.ndarray, block: int, batch: int, seed: int):  # type: ignore[no-untyped-def]
    """Infinite stream of (x, y) windows; identical for every implementation given a seed."""
    rng = np.random.default_rng(seed)
    while True:
        ix = rng.integers(0, len(ids) - block - 1, batch)
        x = np.stack([ids[i : i + block] for i in ix])
        y = np.stack([ids[i + 1 : i + block + 1] for i in ix])
        yield x, y
