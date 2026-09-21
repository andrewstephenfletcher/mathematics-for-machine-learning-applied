"""Device selection, seeding and timing helpers.

Everything in this repo must run on CPU, so :func:`get_device` is a convenience for the times an
accelerator is available, never a requirement.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["get_device", "set_seed", "timer", "Timer"]


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device: CUDA, then MPS, then CPU.

    Args:
        prefer: force a specific device string (``"cpu"``, ``"cuda"``, ``"mps"``). If the
            requested backend is unavailable, fall back to the usual order rather than raising.
            ``MML_DEVICE`` in the environment acts as a default for this argument.

    Returns:
        The selected :class:`torch.device`.
    """
    prefer = prefer or os.environ.get("MML_DEVICE")
    if prefer is not None:
        name = prefer.split(":")[0]
        if name == "cpu":
            return torch.device(prefer)
        if name == "cuda" and torch.cuda.is_available():
            return torch.device(prefer)
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 0, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch (all devices) from a single integer.

    Args:
        seed: the seed to apply.
        deterministic: also ask cuDNN for deterministic kernels. Slower, and some ops have no
            deterministic implementation, so this is off by default.

    Returns:
        The seed, so it can be recorded alongside results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


@dataclass
class Timer:
    """Result of a :func:`timer` block. ``elapsed`` is filled in when the block exits."""

    label: str = ""
    elapsed: float = field(default=float("nan"))

    def __str__(self) -> str:
        prefix = f"{self.label}: " if self.label else ""
        return f"{prefix}{self.elapsed:.4f}s"


@contextmanager
def timer(label: str = "", *, verbose: bool = True) -> Iterator[Timer]:
    """Time a block of code with a monotonic clock.

    >>> with timer("svd") as t:  # doctest: +SKIP
    ...     u, s, vh = torch.linalg.svd(a)
    >>> t.elapsed  # doctest: +SKIP
    0.0123

    Args:
        label: name printed with the elapsed time.
        verbose: print on exit. Set ``False`` to collect timings silently.

    Yields:
        A :class:`Timer` whose ``elapsed`` attribute is set once the block finishes.
    """
    result = Timer(label=label)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed = time.perf_counter() - start
        if verbose:
            print(result)
