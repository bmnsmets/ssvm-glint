"""
Tests for the GLInT classifier and its semantic-only CLS token.
"""

import pytest
import torch

from ssvm_glint.glint import (
    GLInTClassifier,
    InvariantMixingAttention,
    VariationalAffPlus2Invariants,
)
from ssvm_glint.tokens import AffPlus2Tokens, random_affine

B, N, C, D, H = 5, 7, 2, 16, 4


def make_tokens(seed=0):
    """Random tokens with det(A) > 0 and u of dimension 3C."""
    gen = torch.Generator().manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=gen)

    return AffPlus2Tokens.fromg3splatter(
        c0=rn(B, N, C), c1=rn(B, N, C), c2=rn(B, N, C),
        r=rn(B, N), s=0.3 * rn(B, N, 3), t=rn(B, N, 2),
    )  # fmt: skip


def make_model(pool):
    torch.manual_seed(0)
    return GLInTClassifier(
        in_dim=3 * C, num_classes=10, embed_dim=D, depth=2, num_heads=H, pool=pool
    ).eval()


@pytest.mark.parametrize("pool", ["cls", "mean"])
def test_logits_shape(pool):
    assert make_model(pool)(make_tokens()).shape == (B, 10)


def test_mean_pool_has_no_cls_token():
    assert make_model("mean").cls_token is None


def test_cls_logits_permutation_invariant():
    model, tokens = make_model("cls"), make_tokens()
    perm = torch.randperm(N)
    permuted = AffPlus2Tokens(g=tokens.g[:, perm], u=tokens.u[:, perm])
    with torch.no_grad():
        torch.testing.assert_close(model(permuted), model(tokens))


def test_attention_cls_shapes():
    attn = InvariantMixingAttention(embed_dim=D, num_heads=H).eval()
    u, inv_emb, cls = torch.randn(B, N, D), torch.randn(B, N, D), torch.randn(B, 1, D)
    assert attn(u, inv_emb).shape == (B, N, D)
    y, y_cls = attn(u, inv_emb, cls)
    assert y.shape == (B, N, D)
    assert y_cls.shape == (B, 1, D)


def test_alpha_mixes_per_head():
    attn = InvariantMixingAttention(embed_dim=D, num_heads=H).eval()
    d = D // H
    u, inv_emb, cls = torch.randn(B, N, D), torch.randn(B, N, D), torch.randn(B, 1, D)
    with torch.no_grad():
        attn.alpha.copy_(torch.tensor([0.0, 1.0, 0.0, 1.0]))
        attn.beta.copy_(1 - attn.alpha)
    inputs = {}
    attn.query.register_forward_hook(
        lambda m, args, out: inputs.setdefault("x", args[0])
    )
    attn(u, inv_emb, cls)
    x = inputs["x"]  # [B,N+1,D]
    torch.testing.assert_close(x[:, :1], cls)  # CLS is not mixed
    for k, src in enumerate((u, inv_emb, u, inv_emb)):
        block = slice(k * d, (k + 1) * d)  # channels of head k
        torch.testing.assert_close(x[:, 1:, block], src[..., block])


def test_alpha_receives_gradient():
    attn = InvariantMixingAttention(embed_dim=D, num_heads=H)
    torch.nn.init.normal_(attn.value.weight)  # zero at init
    attn(torch.randn(B, N, D), torch.randn(B, N, D)).sum().backward()
    assert attn.alpha.grad is not None
    assert (attn.alpha.grad != 0).all()


def test_cls_token_receives_gradient():
    model = make_model("cls")
    model(make_tokens()).sum().backward()
    assert model.cls_token.grad is not None
    assert model.cls_token.grad.abs().sum() > 0


def test_geometry_affects_logits():
    model, tokens = make_model("cls"), make_tokens(seed=0)
    other_g = AffPlus2Tokens(g=make_tokens(seed=1).g, u=tokens.u)
    with torch.no_grad():
        for layer in model.layers:  # residual branches are zero at init
            torch.nn.init.normal_(layer.attention.value.weight)
            torch.nn.init.normal_(layer.ff[-1].weight)
        assert not torch.allclose(model(other_g), model(tokens))


def test_invariant_noise_only_in_training():
    invariants, tokens = VariationalAffPlus2Invariants(sigma_max=1.0), make_tokens()
    with torch.no_grad():
        invariants.eval()
        torch.testing.assert_close(invariants(tokens), invariants(tokens))
        invariants.train()
        assert not torch.allclose(invariants(tokens), invariants(tokens))


def test_sigma_loss_value():
    model = make_model("cls")
    sigmas = torch.tensor([0.1, 0.5, 1.0, 2.0, 3.0])
    with torch.no_grad():
        model.invariants.log_sigmas.copy_(sigmas.log())
    expected = (sigmas - model.invariants.sigma_max).abs().mean()
    torch.testing.assert_close(model.invariants.sigma_loss(), expected)
    with torch.no_grad():
        model.invariants.log_sigmas.fill_(
            torch.tensor(model.invariants.sigma_max).log()
        )
    torch.testing.assert_close(model.invariants.sigma_loss(), torch.tensor(0.0))


def test_aux_loss_raises_small_sigmas():
    model = GLInTClassifier(
        in_dim=3 * C, num_classes=10, embed_dim=D, depth=1, num_heads=H,
        sigma_max=1.0, sigma_loss_weight=0.1,
    )  # fmt: skip
    model.aux_loss().backward()
    assert model.invariants.log_sigmas.grad is not None
    assert (model.invariants.log_sigmas.grad < 0).all()


def test_aux_loss_off_by_default():
    assert make_model("cls").aux_loss() == 0


def test_log_sigmas_named_per_group():
    model = GLInTClassifier(in_dim=3 * C, num_classes=10, embed_dim=D, depth=1, num_heads=H, sigma_max=0.1)
    with torch.no_grad():
        model.invariants.log_sigmas.copy_(torch.arange(5.0))
    log_sigmas = model.log_sigmas()
    assert list(log_sigmas) == [f"log_sigma/{k}" for k in ("affi", "t", "loga", "theta", "S")]
    assert list(log_sigmas.values()) == pytest.approx(torch.arange(5.0).tolist())


def test_invariants_ignore_autocast():
    """Under bf16 autocast the features of float32 tokens are computed in float32, also for strongly sheared tokens."""
    invariants = VariationalAffPlus2Invariants().eval()
    torch.manual_seed(0)
    tokens = random_affine(make_tokens(), shear=5.0)
    expected = invariants(tokens)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = invariants(tokens)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected)
