"""Tests for the shared library.

These are the template for chapter tests: check against an independent reference
(``torch.autograd``, ``torch.linalg``) or a property that must hold, never against a value copied
from a previous run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mml.checks import assert_close, check_gradient, check_shape, numerical_jacobian
from mml.utils import Timer, get_device, set_seed, timer

# --- utils -------------------------------------------------------------------------------------


def test_get_device_returns_available_device():
    device = get_device()
    assert device.type in {"cpu", "cuda", "mps"}
    # Whatever it picked must actually work.
    torch.zeros(2, device=device)


def test_get_device_honours_cpu_preference():
    assert get_device("cpu").type == "cpu"


def test_get_device_falls_back_when_backend_unavailable():
    device = get_device("cuda" if not torch.cuda.is_available() else "definitely-not-a-backend")
    assert device.type in {"cpu", "cuda", "mps"}


def test_set_seed_makes_all_three_rngs_reproducible():
    import random

    set_seed(1234)
    first = (random.random(), np.random.rand(), torch.rand(3))
    set_seed(1234)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_set_seed_returns_the_seed():
    assert set_seed(7) == 7


def test_different_seeds_give_different_draws():
    set_seed(0)
    a = torch.rand(8)
    set_seed(1)
    b = torch.rand(8)
    assert not torch.equal(a, b)


def test_timer_records_elapsed_time():
    with timer("noop", verbose=False) as t:
        sum(range(10_000))
    assert isinstance(t, Timer)
    assert t.elapsed > 0
    assert "noop" in str(t)


def test_timer_records_elapsed_even_if_the_block_raises():
    t_seen = None
    with pytest.raises(ValueError), timer(verbose=False) as t:
        t_seen = t
        raise ValueError("boom")
    assert t_seen is not None
    assert t_seen.elapsed > 0


# --- checks: assert_close -----------------------------------------------------------------------


def test_assert_close_passes_across_dtypes():
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert_close(a, b)


def test_assert_close_accepts_python_sequences():
    assert_close([1.0, 2.0], torch.tensor([1.0, 2.0]))


def test_assert_close_rejects_a_real_difference():
    with pytest.raises(AssertionError):
        assert_close(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.5]))


def test_assert_close_reports_a_shape_mismatch_as_such():
    with pytest.raises(AssertionError, match="shape"):
        assert_close(torch.zeros(2, 3), torch.zeros(3, 2))


def test_assert_close_message_includes_the_context():
    with pytest.raises(AssertionError, match="projector"):
        assert_close(torch.zeros(2), torch.ones(2), msg="projector")


# --- checks: check_shape ------------------------------------------------------------------------


def test_check_shape_accepts_a_matching_shape_and_returns_the_tensor():
    x = torch.zeros(4, 768)
    assert check_shape(x, (4, 768)) is x


def test_check_shape_treats_none_as_a_wildcard():
    check_shape(torch.zeros(9, 768), (None, 768))


def test_check_shape_rejects_a_wrong_axis():
    with pytest.raises(AssertionError, match="axis 1"):
        check_shape(torch.zeros(4, 768), (4, 512))


def test_check_shape_rejects_a_wrong_rank():
    with pytest.raises(AssertionError, match="rank"):
        check_shape(torch.zeros(4, 768), (4, 768, 1))


# --- checks: numerical differentiation ----------------------------------------------------------


def test_numerical_jacobian_matches_a_known_linear_map():
    # f(x) = A x has Jacobian A everywhere (numerator layout: rows index outputs).
    set_seed(0)
    a = torch.randn(3, 4, dtype=torch.float64)
    jac = numerical_jacobian(lambda x: a @ x, torch.randn(4, dtype=torch.float64))
    assert jac.shape == (3, 4)
    assert_close(jac, a, rtol=1e-6, atol=1e-6)


def test_numerical_jacobian_matches_autograd_on_a_nonlinear_map():
    set_seed(0)
    x = torch.randn(4, dtype=torch.float64)

    def f(v: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.sin(v).sum(), (v**3).sum()])

    numerical = numerical_jacobian(f, x)
    analytic = torch.autograd.functional.jacobian(f, x)
    assert_close(numerical, analytic, rtol=1e-6, atol=1e-6)


def test_check_gradient_accepts_a_correct_analytic_gradient():
    # f(x) = x' A x with symmetric A has gradient 2 A x  (MML eq. 5.72 territory).
    set_seed(0)
    a = torch.randn(5, 5, dtype=torch.float64)
    a = a + a.T

    def f(x: torch.Tensor) -> torch.Tensor:
        return x @ a @ x

    x = torch.randn(5, dtype=torch.float64)
    check_gradient(f, x, grad=lambda v: 2 * a @ v, rtol=1e-6, atol=1e-6)


def test_check_gradient_catches_a_wrong_analytic_gradient():
    set_seed(0)
    a = torch.randn(5, 5, dtype=torch.float64)
    a = a + a.T
    x = torch.randn(5, dtype=torch.float64)

    with pytest.raises(AssertionError):
        # The classic slip: forgetting the factor of 2 from the symmetric quadratic form.
        check_gradient(lambda v: v @ a @ v, x, grad=lambda v: a @ v, rtol=1e-6, atol=1e-6)


def test_check_gradient_defaults_to_autograd():
    set_seed(0)
    x = torch.randn(3, dtype=torch.float64)
    grad = check_gradient(lambda v: torch.log(torch.exp(v).sum()), x, rtol=1e-6, atol=1e-6)
    # log-sum-exp has the softmax as its gradient, which sums to one.
    assert_close(grad.sum(), torch.tensor(1.0, dtype=torch.float64), rtol=1e-6, atol=1e-6)


def test_check_gradient_rejects_a_vector_valued_function():
    with pytest.raises(AssertionError, match="scalar"):
        check_gradient(lambda v: v * 2, torch.randn(3, dtype=torch.float64))
