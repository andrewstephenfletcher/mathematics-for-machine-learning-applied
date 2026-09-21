"""Checking helpers, so every exercise can be verified rather than eyeballed.

The three things worth checking, in the order CLAUDE.md asks for them:

* :func:`check_shape` — the answer has the shape the docstring promised.
* :func:`assert_close` — the answer matches an independent reference.
* :func:`check_gradient` — an analytic derivative matches central finite differences.

Property checks (orthogonality, idempotence, symmetry) live in the chapters that introduce them
and are promoted here once they are reused.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

__all__ = [
    "assert_close",
    "check_shape",
    "numerical_jacobian",
    "check_gradient",
]

ArrayLike = Tensor | float | int | Sequence


def _as_tensor(x: ArrayLike, *, dtype: torch.dtype | None = None) -> Tensor:
    t = x if isinstance(x, Tensor) else torch.as_tensor(x)
    return t if dtype is None else t.to(dtype)


def assert_close(
    actual: ArrayLike,
    expected: ArrayLike,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    msg: str = "",
) -> None:
    """Assert two tensors agree, with a message that says *how far* apart they were.

    Wraps :func:`torch.testing.assert_close` but compares on the CPU in a common dtype, so a
    float32 implementation can be checked against a float64 reference without a spurious failure.

    Args:
        actual: the value under test.
        expected: the reference value.
        rtol: relative tolerance. The float32 default is deliberately loose; tighten it when
            checking a float64 computation.
        atol: absolute tolerance, which dominates near zero.
        msg: extra context prepended to the failure message.

    Raises:
        AssertionError: if the shapes differ or the values are not within tolerance.
    """
    a = _as_tensor(actual).detach().cpu()
    b = _as_tensor(expected).detach().cpu()

    prefix = f"{msg}: " if msg else ""
    if a.shape != b.shape:
        raise AssertionError(f"{prefix}shape {tuple(a.shape)} != {tuple(b.shape)}")

    dtype = torch.promote_types(a.dtype, b.dtype)
    if not dtype.is_floating_point and not dtype.is_complex:
        dtype = torch.float64
    a, b = a.to(dtype), b.to(dtype)

    diff = (a - b).abs()
    scale = b.abs().max().clamp(min=1.0)
    detail = f"max abs diff {diff.max().item():.3e} (scale {scale.item():.3e})"
    torch.testing.assert_close(
        a, b, rtol=rtol, atol=atol, msg=lambda default: f"{prefix}{detail}\n{default}"
    )


def check_shape(
    tensor: Tensor, expected: tuple[int | None, ...], *, name: str = "tensor"
) -> Tensor:
    """Assert a tensor's shape, allowing ``None`` as a wildcard for any one axis.

    >>> import torch
    >>> _ = check_shape(torch.zeros(8, 768), (None, 768), name="activations")

    Args:
        tensor: the tensor to check.
        expected: the expected shape; ``None`` in a position means "any size".
        name: name used in the error message.

    Returns:
        The tensor unchanged, so this can be used inline.

    Raises:
        AssertionError: if the rank or any fixed axis does not match.
    """
    actual = tuple(tensor.shape)
    if len(actual) != len(expected):
        raise AssertionError(
            f"{name}: expected rank {len(expected)} with shape {expected}, got shape {actual}"
        )
    for axis, (got, want) in enumerate(zip(actual, expected, strict=True)):
        if want is not None and got != want:
            raise AssertionError(
                f"{name}: axis {axis} is {got}, expected {want} (full shape {actual} vs {expected})"
            )
    return tensor


def numerical_jacobian(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Central-difference Jacobian of ``f`` at ``x``, in numerator layout.

    Numerator layout (the book's convention, §5.1) means row ``i`` holds the gradient of output
    ``i``, so the result has shape ``(*f(x).shape, *x.shape)``. For a scalar ``f`` of a vector
    ``x`` this is a row vector of shape ``(n,)`` — note that ``torch.autograd.grad`` instead
    returns the same shape as ``x``, which for a 1-D input coincides.

    The input is promoted to float64: central differences in float32 lose roughly half the
    available precision and will fail any tolerance worth setting.

    Args:
        f: function of a single tensor, returning a tensor.
        x: point at which to differentiate.
        eps: step size. ``1e-6`` is near optimal for float64; too small and rounding dominates.

    Returns:
        Jacobian of shape ``(*f(x).shape, *x.shape)``.
    """
    x64 = x.detach().to(torch.float64)
    out_shape = tuple(f(x64).shape)
    jac = torch.zeros((*out_shape, *tuple(x64.shape)), dtype=torch.float64)

    flat = x64.reshape(-1)
    for i in range(flat.numel()):
        step = torch.zeros_like(flat)
        step[i] = eps
        plus = f((flat + step).reshape(x64.shape)).to(torch.float64)
        minus = f((flat - step).reshape(x64.shape)).to(torch.float64)
        jac.reshape(*out_shape, -1)[..., i] = (plus - minus) / (2 * eps)
    return jac


def check_gradient(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    grad: Tensor | Callable[[Tensor], Tensor] | None = None,
    *,
    eps: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> Tensor:
    """Check an analytic gradient of a **scalar** function against finite differences.

    Args:
        f: function of a single tensor returning a scalar (a loss).
        x: point at which to check.
        grad: the analytic gradient — a tensor, or a callable evaluated at ``x``. If ``None``,
            ``torch.autograd`` supplies it, which turns this into a check of ``f`` itself.
        eps: finite-difference step, passed to :func:`numerical_jacobian`.
        rtol: relative tolerance for the comparison.
        atol: absolute tolerance for the comparison.

    Returns:
        The numerical gradient, shaped like ``x``.

    Raises:
        AssertionError: if ``f`` is not scalar-valued, or the gradients disagree.
    """
    probe = f(x.detach().to(torch.float64))
    if probe.numel() != 1:
        raise AssertionError(
            f"check_gradient expects a scalar-valued f, got output shape {tuple(probe.shape)}; "
            "use numerical_jacobian for vector-valued functions"
        )

    numerical = numerical_jacobian(f, x, eps=eps).reshape(x.shape)

    if grad is None:
        xa = x.detach().to(torch.float64).requires_grad_(True)
        (analytic,) = torch.autograd.grad(f(xa).reshape(()), xa)
    elif callable(grad):
        analytic = grad(x)
    else:
        analytic = grad

    assert_close(
        analytic.reshape(x.shape),
        numerical,
        rtol=rtol,
        atol=atol,
        msg="analytic gradient disagrees with central differences",
    )
    return numerical
