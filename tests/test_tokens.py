"""
Tests for ssvm_glint.geometry.tokens.
"""

import dataclasses

import pytest
import torch

from ssvm_glint.geometry.exp import exp_so2, exp_sym2
from ssvm_glint.geometry.tokens import AffPlus2Tokens

B, N, C = 2, 5, 3


def make_inputs(dtype=torch.float64, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g, dtype=dtype).to(device)

    return dict(
        c0=rn(B, N, C),
        c1=rn(B, N, C),
        c2=rn(B, N, C),
        r=rn(B, N),
        s=rn(B, N, 3),
        t=rn(B, N, 2),
    )


def lie_A(r, s):
    """Reference A = expm(so2(r)) @ expm(sym2(s)) via torch.linalg.matrix_exp."""
    z = torch.zeros_like(r)
    X = torch.stack([torch.stack([z, -r], -1), torch.stack([r, z], -1)], -2)
    s1, s2, s3 = s[..., 0], s[..., 1], s[..., 2]
    Y = torch.stack([torch.stack([s1, s3], -1), torch.stack([s3, s2], -1)], -2)
    return torch.linalg.matrix_exp(X) @ torch.linalg.matrix_exp(Y)


# ---------------------------------------------------------------------------
# construction and shapes
# ---------------------------------------------------------------------------


def test_shapes_and_batch_size():
    tok = AffPlus2Tokens.fromg3splatter(**make_inputs())
    assert tok.g.shape == (B, N, 2, 3)
    assert tok.u.shape == (B, N, 3 * C)
    assert tok.batch_size == torch.Size([B, N])
    assert tok.A.shape == (B, N, 2, 2)
    assert tok.x.shape == (B, N, 2)
    assert tok.a.shape == (B, N)


def test_semantic_layout():
    inp = make_inputs()
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    torch.testing.assert_close(tok.u[..., :C], inp["c0"], rtol=0, atol=0)
    torch.testing.assert_close(tok.u[..., C : 2 * C], inp["c1"], rtol=0, atol=0)
    torch.testing.assert_close(tok.u[..., 2 * C :], inp["c2"], rtol=0, atol=0)


def test_translation():
    inp = make_inputs()
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    torch.testing.assert_close(tok.x, inp["t"], rtol=0, atol=0)


@pytest.mark.parametrize(
    "key, bad",
    [
        ("c1", lambda x: x[..., :-1]),  # coefficient shapes differ
        ("r", lambda x: x[:, :-1]),  # r does not match [B,N]
        ("s", lambda x: x[..., :2]),  # s last dim != 3
        ("t", lambda x: x[..., :1]),  # t last dim != 2
    ],
)
def test_shape_asserts(key, bad):
    inp = make_inputs()
    inp[key] = bad(inp[key])
    with pytest.raises(AssertionError):
        AffPlus2Tokens.fromg3splatter(**inp)


@pytest.mark.parametrize("shape", [(), (7,), (2, 3, 4)])
def test_batch_size_inferred(shape):
    tok = AffPlus2Tokens(g=torch.randn(*shape, 2, 3), u=torch.randn(*shape, 5))
    assert tok.batch_size == torch.Size(shape)


def test_batch_size_inferred_when_D_equals_2():
    """u of shape [...,2] must not be mistaken for an extra batch dimension of g."""
    tok = AffPlus2Tokens(g=torch.randn(B, N, 2, 3), u=torch.randn(B, N, 2))
    assert tok.batch_size == torch.Size([B, N])


@pytest.mark.parametrize(
    "g_shape",
    [
        (B, N, 3, 3),  # full homogeneous matrix
        (B, N, 2, 2),  # linear part only
        (B, N, 6),  # flattened
        (3,),  # too few dimensions
    ],
)
def test_bad_g_shape(g_shape):
    with pytest.raises(ValueError, match="g must have shape"):
        AffPlus2Tokens(g=torch.randn(*g_shape), u=torch.randn(B, N, 5))


@pytest.mark.parametrize(
    "u_shape",
    [
        (B, N + 1, 5),  # leading dimensions differ
        (B, 5),  # missing a batch dimension
        (B, N, 4, 5),  # extra dimension
    ],
)
def test_bad_u_shape(u_shape):
    with pytest.raises(ValueError, match="u must have shape"):
        AffPlus2Tokens(g=torch.randn(B, N, 2, 3), u=torch.randn(*u_shape))


def test_batch_size_after_operations():
    tok = AffPlus2Tokens.fromg3splatter(**make_inputs())
    assert tok.replace(u=torch.randn(B, N, 7, dtype=tok.u.dtype)).batch_size == (B, N)
    assert tok[:, 1:].batch_size == (B, N - 1)
    assert tok.clone().batch_size == (B, N)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_A_matches_matrix_exp():
    inp = make_inputs()
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    torch.testing.assert_close(tok.A, lie_A(inp["r"], inp["s"]))


def test_identity():
    inp = make_inputs()
    inp["r"] = torch.zeros_like(inp["r"])
    inp["s"] = torch.zeros_like(inp["s"])
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    torch.testing.assert_close(tok.A, torch.eye(2, dtype=tok.A.dtype).expand_as(tok.A))


def test_determinant_positive():
    inp = make_inputs()
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    s = inp["s"]
    det = torch.linalg.det(tok.A)
    torch.testing.assert_close(det, torch.exp(s[..., 0] + s[..., 1]))
    assert (det > 0).all()


def test_scale_factor():
    inp = make_inputs()
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    s = inp["s"]
    torch.testing.assert_close(tok.a, torch.exp((s[..., 0] + s[..., 1]) / 2))


def test_polar_decomposition():
    """A^T A = S^2 since the rotation cancels, with S symmetric positive definite."""
    inp = make_inputs()
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    S = exp_sym2(inp["s"])
    torch.testing.assert_close(tok.A.mT @ tok.A, S @ S)
    torch.testing.assert_close(S, S.mT)
    assert (torch.linalg.eigvalsh(S) > 0).all()


def test_consistent_with_rasterizer_inverse():
    """g3splatter.rasterize uses Ainv = exp_sym2(-s) @ exp_so2(-r)."""
    inp = make_inputs()
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    Ainv = exp_sym2(-inp["s"]) @ exp_so2(-inp["r"])
    eye = torch.eye(2, dtype=tok.A.dtype).expand_as(tok.A)
    torch.testing.assert_close(tok.A @ Ainv, eye)


def test_t_property():
    """t = A^-1 x, i.e. the translation mapped back through the inverse rasterizer transform."""
    inp = make_inputs()
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    assert tok.t.shape == (B, N, 2)
    expected = torch.linalg.solve(tok.A, tok.x)
    torch.testing.assert_close(tok.t, expected)
    Ainv = exp_sym2(-inp["s"]) @ exp_so2(-inp["r"])
    torch.testing.assert_close(tok.t, (Ainv @ inp["t"][..., None])[..., 0])
    torch.testing.assert_close((tok.A @ tok.t[..., None])[..., 0], tok.x)


def test_t_property_identity():
    inp = make_inputs()
    inp["r"] = torch.zeros_like(inp["r"])
    inp["s"] = torch.zeros_like(inp["s"])
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    torch.testing.assert_close(tok.t, inp["t"])


def test_t_property_gradients():
    inp = {k: v.requires_grad_() for k, v in make_inputs().items()}
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    tok.t.sum().backward()
    for k in ("r", "s", "t"):
        assert inp[k].grad is not None, k
        assert inp[k].grad.isfinite().all(), k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_t_property_cuda():
    inp = make_inputs()
    cpu = AffPlus2Tokens.fromg3splatter(**inp)
    gpu = AffPlus2Tokens.fromg3splatter(**{k: v.cuda() for k, v in inp.items()})
    torch.testing.assert_close(gpu.t.cpu(), cpu.t)


def test_R_S_factorization():
    tok = AffPlus2Tokens.fromg3splatter(**make_inputs())
    R, S = tok.R, tok.S
    assert R.shape == S.shape == (B, N, 2, 2)
    torch.testing.assert_close(tok.a[..., None, None] * R @ S, tok.A)


def test_polar_matches_properties():
    inp = {k: v.requires_grad_() for k, v in make_inputs().items()}
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    a, R, S = tok.polar()
    torch.testing.assert_close(a, tok.a)
    torch.testing.assert_close(R, tok.R)
    torch.testing.assert_close(S, tok.S)
    (a.sum() + R.sum() + S.sum()).backward()
    for k in ("r", "s"):
        assert inp[k].grad is not None, k
        assert inp[k].grad.isfinite().all(), k


def test_R_is_rotation():
    tok = AffPlus2Tokens.fromg3splatter(**make_inputs())
    R = tok.R
    eye = torch.eye(2, dtype=R.dtype).expand_as(R)
    torch.testing.assert_close(R.mT @ R, eye)
    torch.testing.assert_close(torch.linalg.det(R), torch.ones_like(tok.a))


def test_S_is_unimodular_spd():
    tok = AffPlus2Tokens.fromg3splatter(**make_inputs())
    S = tok.S
    torch.testing.assert_close(S, S.mT)
    torch.testing.assert_close(torch.linalg.det(S), torch.ones_like(tok.a))
    assert (torch.linalg.eigvalsh(S) > 0).all()


def test_R_S_match_g3splatter_params():
    """R = exp_so2(r) (angles mod 2pi) and S = exp_sym2(s) / exp((s1+s2)/2)."""
    inp = make_inputs()
    inp["r"] = 10 * inp["r"]  # exercise angles well outside (-pi, pi]
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    s = inp["s"]
    torch.testing.assert_close(tok.R, exp_so2(inp["r"]))
    S = exp_sym2(s) / torch.exp((s[..., 0] + s[..., 1]) / 2)[..., None, None]
    torch.testing.assert_close(tok.S, S)


def test_R_S_gradients():
    inp = {k: v.requires_grad_() for k, v in make_inputs().items()}
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    (tok.a.sum() + tok.R.sum() + tok.S.sum()).backward()
    for k in ("r", "s"):
        assert inp[k].grad is not None, k
        assert inp[k].grad.isfinite().all(), k


def test_near_isotropic_scaling():
    """s1 ~ s2 and s3 ~ 0 hits the eps / sinhc Taylor branch of exp_sym2."""
    inp = make_inputs()
    base = inp["s"][..., :1]
    inp["s"] = torch.cat([base, base + 1e-9, torch.full_like(base, 1e-9)], dim=-1)
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    assert tok.A.isfinite().all()
    torch.testing.assert_close(tok.A, lie_A(inp["r"], inp["s"]))


# ---------------------------------------------------------------------------
# tensor behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_follows_inputs(dtype):
    tok = AffPlus2Tokens.fromg3splatter(**make_inputs(dtype=dtype))
    assert tok.g.dtype == dtype
    assert tok.u.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_follows_inputs():
    tok = AffPlus2Tokens.fromg3splatter(**make_inputs(device="cuda"))
    assert tok.g.device.type == "cuda"
    assert tok.u.device.type == "cuda"


def test_gradients_reach_all_inputs():
    inp = {k: v.requires_grad_() for k, v in make_inputs().items()}
    tok = AffPlus2Tokens.fromg3splatter(**inp)
    (tok.g.sum() + tok.u.sum()).backward()
    for k, v in inp.items():
        assert v.grad is not None, k
        assert v.grad.isfinite().all(), k


def test_gradcheck_geometry(monkeypatch):
    # gradcheck retains the graph, which torch.compile'd backwards with donated
    # buffers (exp_so2, exp_sym2) do not allow
    monkeypatch.setattr(torch._functorch.config, "donated_buffer", False)
    inp = make_inputs()
    r = inp["r"].requires_grad_()
    s = inp["s"].requires_grad_()
    t = inp["t"].requires_grad_()

    def f(r, s, t):
        return AffPlus2Tokens.fromg3splatter(inp["c0"], inp["c1"], inp["c2"], r, s, t).g

    assert torch.autograd.gradcheck(f, (r, s, t))


def test_frozen():
    tok = AffPlus2Tokens.fromg3splatter(**make_inputs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        tok.g = tok.g * 2


def test_indexing():
    tok = AffPlus2Tokens.fromg3splatter(**make_inputs())
    t0 = tok[0]
    assert t0.batch_size == torch.Size([N])
    assert t0.g.shape == (N, 2, 3)
    torch.testing.assert_close(t0.g, tok.g[0])

    sl = tok[:, :3]
    assert sl.batch_size == torch.Size([B, 3])
    assert sl.u.shape == (B, 3, 3 * C)


def test_stack_and_cat():
    tok = AffPlus2Tokens.fromg3splatter(**make_inputs())
    assert torch.cat([tok, tok], dim=1).batch_size == torch.Size([B, 2 * N])
    assert torch.stack([tok, tok], dim=0).batch_size == torch.Size([2, B, N])
