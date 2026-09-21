"""Shared library for the Mathematics for Machine Learning applied repo.

This package grows as chapters are completed. When a function written in a chapter's ``src/``
proves generally useful, it is promoted here and given tests in the top-level ``tests/``.

At setup only :mod:`mml.utils` and :mod:`mml.checks` have implementations;
:mod:`mml.plotting` and :mod:`mml.models` are thin and fill in as the chapters need them.
"""

from mml.checks import assert_close, check_gradient, check_shape
from mml.utils import get_device, set_seed, timer

__all__ = [
    "assert_close",
    "check_gradient",
    "check_shape",
    "get_device",
    "set_seed",
    "timer",
]
