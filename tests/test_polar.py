"""
Tests for ssvm_glint.geometry.polar.

CUDA tests are skipped when no GPU is available.
"""

import math

import pytest
import torch

from ssvm_glint.geometry.polar import (
    _polar2x2,
    polar2x2,
    polar2x2_a,
    polar2x2_R,
    polar2x2_S,
)


def random_glp2(*shape, dtype=torch.float64, device="cpu", seed=0):
    """Random matrices in GL^+(2) with shape [*shape,2,2]."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(*shape, 2, 2, generator=g, dtype=dtype)
    # flip the first column where det < 0
    sign = torch.linalg.det(A).sign()[..., None]
    A = torch.stack([A[..., :, 0] * sign, A[..., :, 1]], -1)
    return A.to(device)


def rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


def eye_like(A):
    return torch.eye(2, dtype=A.dtype, device=A.device).expand_as(A)


def svd_polar(A):
    """Reference: A = U diag(sigma) V^T gives R = U V^T and P = V diag(sigma) V^T."""
    U, sigma, Vh = torch.linalg.svd(A)
    R = U @ Vh
    P = Vh.mT @ torch.diag_embed(sigma) @ Vh
    a = torch.linalg.det(A).sqrt()
    return a, R, P / a[..., None, None]


# ---------------------------------------------------------------------------
# decomposition properties
# ---------------------------------------------------------------------------


def test_reconstruction():
    A = random_glp2(64)
    a, R, S = polar2x2(A)
    torch.testing.assert_close(a[..., None, None] * R @ S, A)


def test_a_is_sqrt_det():
    A = random_glp2(64)
    a, _, _ = polar2x2(A)
    torch.testing.assert_close(a, torch.linalg.det(A).sqrt())


def test_R_in_SO2():
    A = random_glp2(64)
    _, R, _ = polar2x2(A)
    torch.testing.assert_close(R.mT @ R, eye_like(R))
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(64, dtype=R.dtype))


def test_S_unimodular_spd():
    A = random_glp2(64)
    _, _, S = polar2x2(A)
    torch.testing.assert_close(S, S.mT)
    torch.testing.assert_close(torch.linalg.det(S), torch.ones(64, dtype=S.dtype))
    assert (torch.linalg.eigvalsh(S) > 0).all()


def test_matches_svd():
    A = random_glp2(64)
    for got, ref in zip(polar2x2(A), svd_polar(A)):
        torch.testing.assert_close(got, ref)


# ---------------------------------------------------------------------------
# known cases
# ---------------------------------------------------------------------------


def test_identity():
    A = torch.eye(2, dtype=torch.float64)
    a, R, S = polar2x2(A)
    torch.testing.assert_close(a, torch.tensor(1.0, dtype=A.dtype))
    torch.testing.assert_close(R, A)
    torch.testing.assert_close(S, A)


@pytest.mark.parametrize("theta", [0.3, -2.0, math.pi / 2, math.pi - 1e-6, math.pi])
def test_pure_rotation(theta):
    A = rotation(theta)
    a, R, S = polar2x2(A)
    torch.testing.assert_close(a, torch.tensor(1.0, dtype=A.dtype))
    torch.testing.assert_close(R, A)
    torch.testing.assert_close(S, torch.eye(2, dtype=A.dtype))


def test_isotropic_scaling():
    A = 3.5 * torch.eye(2, dtype=torch.float64)
    a, R, S = polar2x2(A)
    torch.testing.assert_close(a, torch.tensor(3.5, dtype=A.dtype))
    torch.testing.assert_close(R, torch.eye(2, dtype=A.dtype))
    torch.testing.assert_close(S, torch.eye(2, dtype=A.dtype))


def test_spd_has_trivial_rotation():
    A = torch.tensor([[2.0, 0.7], [0.7, 1.5]], dtype=torch.float64)
    a, R, S = polar2x2(A)
    torch.testing.assert_close(R, torch.eye(2, dtype=A.dtype))
    torch.testing.assert_close(a[..., None, None] * S, A)


def test_ill_conditioned():
    """Strong anisotropy, condition number 1e6."""
    P = torch.diag(torch.tensor([1e3, 1e-3], dtype=torch.float64))
    Q = rotation(0.4)
    A = rotation(1.1) @ Q @ P @ Q.mT
    for got, ref in zip(polar2x2(A), svd_polar(A)):
        torch.testing.assert_close(got, ref)


# ---------------------------------------------------------------------------
# equivariance
# ---------------------------------------------------------------------------


def test_left_rotation_equivariance():
    """polar(Q A) = (a, Q R, S)"""
    A = random_glp2(64)
    Q = rotation(0.8)
    a, R, S = polar2x2(A)
    a2, R2, S2 = polar2x2(Q @ A)
    torch.testing.assert_close(a2, a)
    torch.testing.assert_close(R2, Q @ R)
    torch.testing.assert_close(S2, S)


def test_right_rotation_equivariance():
    """polar(A Q) = (a, R Q, Q^T S Q)"""
    A = random_glp2(64)
    Q = rotation(-1.3)
    a, R, S = polar2x2(A)
    a2, R2, S2 = polar2x2(A @ Q)
    torch.testing.assert_close(a2, a)
    torch.testing.assert_close(R2, R @ Q)
    torch.testing.assert_close(S2, Q.mT @ S @ Q)


def test_scale_equivariance():
    """polar(l A) = (l a, R, S) for l > 0"""
    A = random_glp2(64)
    a, R, S = polar2x2(A)
    a2, R2, S2 = polar2x2(2.5 * A)
    torch.testing.assert_close(a2, 2.5 * a)
    torch.testing.assert_close(R2, R)
    torch.testing.assert_close(S2, S)


# ---------------------------------------------------------------------------
# entry points, shapes, dtypes, devices
# ---------------------------------------------------------------------------


def test_entry_points_agree():
    A = random_glp2(64)
    a, R, S = _polar2x2(A)
    ac, Rc, Sc = polar2x2(A)
    torch.testing.assert_close(ac, a)
    torch.testing.assert_close(Rc, R)
    torch.testing.assert_close(Sc, S)
    torch.testing.assert_close(polar2x2_a(A), a)
    torch.testing.assert_close(polar2x2_R(A), R)
    torch.testing.assert_close(polar2x2_S(A), S)


@pytest.mark.parametrize("shape", [(), (7,), (2, 3, 4)])
def test_batch_shapes(shape):
    A = random_glp2(*shape)
    a, R, S = polar2x2(A)
    assert a.shape == shape
    assert R.shape == S.shape == (*shape, 2, 2)
    torch.testing.assert_close(a[..., None, None] * R @ S, A)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype(dtype):
    A = random_glp2(64, dtype=dtype)
    a, R, S = polar2x2(A)
    assert a.dtype == R.dtype == S.dtype == dtype
    torch.testing.assert_close(a[..., None, None] * R @ S, A)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu():
    A = random_glp2(64)
    for got, ref in zip(polar2x2(A.cuda()), polar2x2(A)):
        assert got.device.type == "cuda"
        torch.testing.assert_close(got.cpu(), ref)


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------


def test_gradcheck_eager():
    A = random_glp2(8).requires_grad_()
    assert torch.autograd.gradcheck(_polar2x2, (A,))


def test_gradcheck_compiled(monkeypatch):
    # gradcheck retains the graph, which compiled backwards with donated buffers do not allow
    monkeypatch.setattr(torch._functorch.config, "donated_buffer", False)
    A = random_glp2(8).requires_grad_()
    assert torch.autograd.gradcheck(polar2x2, (A,))
    for f in (polar2x2_a, polar2x2_R, polar2x2_S):
        assert torch.autograd.gradcheck(f, (A,))
