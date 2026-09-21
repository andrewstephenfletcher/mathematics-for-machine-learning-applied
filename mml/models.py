"""Loading small models and caching their activations.

Deliberately thin at setup — the real helpers arrive with the chapters that need them:

* **Ch 2–3** — ``cache_residual_stream(model, prompts, layer)`` returning
  ``Float[Tensor, "batch d_model"]``, with caching to ``data/`` so a notebook restart is cheap.
* **Ch 4** — ``ov_matrix(model, layer, head)`` and ``qk_matrix(...)``, the full composed circuits
  rather than the raw weight blocks.
* **Ch 8–12** — a probe-training helper with proper train/val/test splits.

Everything must run on CPU: stick to GPT-2 small, Pythia-70m and toy models. The heavy
dependencies live in the optional ``interp`` extra (``uv sync --extra interp``), so this module
must remain importable without them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    pass

__all__ = ["DEFAULT_MODEL", "load_model", "require_interp"]

#: Small enough to run on CPU, large enough to have interesting circuits.
DEFAULT_MODEL = "gpt2-small"


def require_interp() -> Any:
    """Import :mod:`transformer_lens`, with an actionable error if the extra is not installed.

    Returns:
        The imported :mod:`transformer_lens` module.

    Raises:
        ImportError: if the ``interp`` extra is missing.
    """
    try:
        import transformer_lens
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise ImportError(
            "transformer-lens is not installed. Run `uv sync --extra interp` to install the "
            "interpretability dependencies."
        ) from exc
    return transformer_lens


def load_model(name: str = DEFAULT_MODEL, device: torch.device | str | None = None) -> Any:
    """Load a small HookedTransformer for the interpretability exercises.

    Args:
        name: a transformer-lens model name, e.g. ``"gpt2-small"`` or ``"pythia-70m"``.
        device: where to put the model; defaults to :func:`mml.utils.get_device`.

    Returns:
        A ``transformer_lens.HookedTransformer``.
    """
    from mml.utils import get_device

    tl = require_interp()
    return tl.HookedTransformer.from_pretrained(name, device=str(device or get_device()))
