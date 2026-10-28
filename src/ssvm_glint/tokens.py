import torch
from tensordict import TensorClass
from torch import Tensor

from ssvm_glint.math import (
    exp_so2,
    exp_sym2,
    inv2x2,
    inv_apply2x2,
    polar2x2,
    polar2x2_a,
    polar2x2_R,
    polar2x2_S,
)


class AffPlus2Tokens(TensorClass["frozen", "tensor_only"]):  # ty: ignore[unsupported-base]
    """
    Represent a set of geometric tokens using the affine group with positive determinant in 2D (Aff^+(2)).
    Each token pairs a group element (geometric part `g`) with a feature vector (semantic part `u`).

    Internally, group elements are stored as a tensor of shape [...,2,3] to match the (2 upper rows of the) matrix representation of Aff^+(2) as a subgroup of GL(3).
    If `tokens` is an instance of `AffPlus2Tokens` then `x = tokens.g[...,:,2]` and `A = tokens.g[...,:,0:2]`.

    Invariant: det(A) > 0. This is not checked on construction; `a`, `R`, `S` and `polar()` return NaN when it is violated.

    Shapes: `g` must be [...,2,3] and `u` must be [...,D] with matching leading dimensions `...`,
    which become the batch size.

    Instances are frozen: use `tokens.replace(u=...)` to obtain tokens with new semantic features, this reuses `g` without copying.
    """

    g: Tensor  # [B,N,2,3]    geometric part
    u: Tensor  # [B,N,D]    semantic part

    def __post_init__(self):
        if self.g.shape[-2:] != (2, 3):
            raise ValueError(f"g must have shape [...,2,3], got {tuple(self.g.shape)}")
        if self.u.shape[:-1] != self.g.shape[:-2]:
            raise ValueError(
                f"u must have shape [...,D] matching g [...,2,3], got u {tuple(self.u.shape)} and g {tuple(self.g.shape)}"
            )
        # batch_size cannot be assigned directly on a frozen tensorclass, so set it on the underlying tensordict
        self._tensordict.batch_size = self.g.shape[:-2]

    @classmethod
    def fromg3splatter(
        cls, c0: Tensor, c1: Tensor, c2: Tensor, r: Tensor, s: Tensor, t: Tensor
    ) -> "AffPlus2Tokens":
        """
        Convert from the Kaji-Ochiai parameterization of the g3splatter
        (S. Kaji and H. Ochiai, "A concise parametrization of affine transformation", SIAM J. Imaging Sci., 2016).

        The tokens are built as:
            A = exp_so2(r) @ exp_sym2(s)
            x = t
            u = cat(c0, c1, c2),
            i.e. u[..., :C] = c0, u[..., C:2C] = c1 and u[..., 2C:] = c2

        No normalization is applied here; `get_g3splat_dataloaders` standardizes `u`, `t` and log(a)
        with training-set statistics once both splits are loaded.

        Note: the `t` argument becomes `x`; it is not the `t` property of the result, which is A^-1 x.

        Args:
            c0 (Tensor) [B,N,C]: 0-th order Gaussian coefficients
            c1 (Tensor) [B,N,C]: 1-th order Gaussian coefficients
            c2 (Tensor) [B,N,C]: 2-th order Gaussian coefficients
            r (Tensor) [B,N]: rotation angle
            s (Tensor) [B,N,3]: scaling/shearing coefficients
            t (Tensor) [B,N,2]: translation vector

        Returns:
            AffPlus2Tokens: with batch size [B,N], `g` of shape [B,N,2,3] and `u` of shape [B,N,3C],
                with the dtype and device of `c0`.
        """
        assert c0.shape == c1.shape == c2.shape
        assert c0.shape[:2] == r.shape == s.shape[:2] == t.shape[:2]
        assert s.shape[-1] == 3
        assert t.shape[-1] == 2
        B, N = r.shape

        u = torch.cat((c0, c1, c2), dim=-1)  # [B,N,3C]
        g = torch.zeros(B, N, 2, 3).to(u)  # [B,N,2,3]
        g[..., :, 2] = t
        R = exp_so2(r)
        S = exp_sym2(s)
        g[..., :, :2] = torch.matmul(R, S)
        return AffPlus2Tokens(g=g, u=u)

    @property
    def A(self) -> Tensor:
        """
        Linear (GL^+(2)) part of the group element, shape [...,2,2].
        """
        return self.g[..., :, :2]

    @property
    def x(self) -> Tensor:
        """
        Translation (R^2,+) part of the group element, shape [...,2].
        """
        return self.g[..., :, 2]

    @property
    def a(self) -> Tensor:
        """
        Isotropic scale factor of A = a * R @ S, i.e. a = sqrt(det(A)), shape [...].
        """
        return polar2x2_a(self.A)

    @property
    def t(self) -> Tensor:
        """
        t = A^-1 x, shape [...,2].

        Note: intentionally without a minus sign, so this is *not* the translation part -A^-1 x of the inverse group element.
        """
        return inv_apply2x2(self.A, self.x)

    @property
    def R(self) -> Tensor:
        """
        Rotation factor of the polar decomposition A = a * R @ S, shape [...,2,2].
        """
        return polar2x2_R(self.A)

    @property
    def S(self) -> Tensor:
        """
        SL(2) factor of the polar decomposition A = a * R @ S, symmetric positive definite with det(S) = 1, shape [...,2,2].
        """
        return polar2x2_S(self.A)

    @property
    def ntokens(self) -> int:
        return self.g.shape[1]

    def polar(self) -> tuple[Tensor, Tensor, Tensor]:
        """
        Polar decomposition A = a * R @ S in one pass, cheaper than accessing `a`, `R` and `S` separately.

        Returns:
            a (Tensor) [...]: isotropic scale factor sqrt(det(A))
            R (Tensor) [...,2,2]: rotation in SO(2)
            S (Tensor) [...,2,2]: symmetric positive definite with det(S) = 1
        """
        return polar2x2(self.A)

    def canonical_invariant(self) -> Tensor:
        """
        Position of the token mean relative to each token, i.e. the group product g^-1 gmean,
        where gmean is the elementwise mean of `g` over the token dimension.

        Writing gmean = (Am, xm), the result is (A^-1 Am, A^-1 (xm - x)). Note that gmean need not
        itself lie in Aff^+(2); the formula is applied regardless, and is exact whenever det(Am) > 0.

        Equivariance: replacing every token by h g for a fixed h in Aff^+(2) leaves the result unchanged,
        since gmean is linear in g.

        Returns:
            Tensor [...,2,3]: same shape as `g`.
        """
        gmean = self.g.mean(dim=-3, keepdim=True)  # [...,1,2,3]
        Ainv = inv2x2(self.A)  # [...,2,2]
        A = torch.matmul(Ainv, gmean[..., :, :2])  # [...,2,2]
        x = torch.matmul(Ainv, (gmean[..., :, 2] - self.x)[..., None])  # [...,2,1]
        return torch.cat((A, x), dim=-1)

    def relative(self) -> Tensor:
        """
        Relative group elements g_i^-1 g_j for all token pairs, with token i as the reference.

        Writing g_i = (A_i, x_i), entry (i, j) is (A_i^-1 A_j, A_i^-1 (x_j - x_i)).

        Invariance: replacing every token by h g for a fixed h in Aff^+(2) leaves the result unchanged.

        Returns:
            Tensor [...,N,N,2,3]: entry [..., i, j, :, :] is g_i^-1 g_j.
        """
        Ainv = inv2x2(self.A)[..., :, None, :, :]  # [...,N,1,2,2]
        A = torch.matmul(Ainv, self.A[..., None, :, :, :])  # [...,N,N,2,2]
        dx = self.x[..., None, :, :] - self.x[..., :, None, :]  # [...,N,N,2], x_j - x_i
        x = torch.matmul(Ainv, dx[..., None])  # [...,N,N,2,1]
        return torch.cat((A, x), dim=-1)


def random_affine(
    tokens: AffPlus2Tokens,
    rotate: float = 0.0,
    shear: float = 0.0,
    log_scale: float = 0.0,
    translate: float = 0.0,
    translate_scale: float = 1.0,
) -> AffPlus2Tokens:
    """
    Left action g -> h g with a random h = (H, b) in Aff^+(2) per sample, shared by all tokens of that sample.

    H = exp_so2(theta) @ exp_sym2([s1, -s1, s3]) * exp(s) acts about the origin, where the traceless
    exp([[s1,s3],[s3,-s1]]) is a volume preserving stretch (det 1). theta, s1 and s3, s and both components
    of b are drawn uniformly from [-rotate, rotate], [-shear, shear], [-log_scale, log_scale] and
    [-translate, translate]; a range of 0 turns that transformation off. b is multiplied by
    `translate_scale`, e.g. 1 / (c * tstd) to act on tokens normalized by `get_g3splat_dataloaders`.

    Args:
        tokens (AffPlus2Tokens): batch size [...,N], with the tokens of one sample along the last dimension.
        rotate (float): maximum rotation angle in radians.
        shear (float): maximum of the stretch coefficients s1 and s3.
        log_scale (float): maximum log of the isotropic scale factor.
        translate (float): maximum translation per component.
        translate_scale (float): factor applied to the sampled translation.

    Returns:
        AffPlus2Tokens: batch size [...,N] with g = (H A, H x + translate_scale * b) and the same `u`.
    """
    dtype = torch.promote_types(tokens.g.dtype, torch.float32)
    g = tokens.g.to(dtype)  # [...,N,2,3]
    shape = g.shape[:-3]

    def uniform(r: float, *s: int) -> Tensor:
        return (2 * torch.rand(*shape, *s, dtype=dtype, device=g.device) - 1) * r

    theta, s1, s3, s = (
        uniform(rotate),
        uniform(shear),
        uniform(shear),
        uniform(log_scale),
    )  # [...]
    b = uniform(translate, 2) * translate_scale  # [...,2]

    S = exp_sym2(torch.stack((s1, -s1, s3), dim=-1))  # [...,2,2]
    H = (exp_so2(theta) @ S * s.exp()[..., None, None])[..., None, :, :]  # [...,1,2,2]
    A = torch.matmul(H, g[..., :2])  # [...,N,2,2]
    x = torch.matmul(H, g[..., 2:]) + b[..., None, :, None]  # [...,N,2,1]
    return AffPlus2Tokens(g=torch.cat((A, x), dim=-1).to(tokens.g.dtype), u=tokens.u)
