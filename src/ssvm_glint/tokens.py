import torch
from torch import Tensor
from tensordict import TensorClass
from ssvm_glint.exp import exp_sym2, exp_so2
from ssvm_glint.polar import polar2x2, polar2x2_a, polar2x2_R, polar2x2_S


@torch.compile()
def _inv_apply2x2(A: Tensor, x: Tensor) -> Tensor:
    """
    Compute A^-1 x for A with shape [...,2,2] and x with shape [...,2], using A^-1 = adj(A) / det(A).
    """
    A00, A01, A10, A11 = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
    det = A00 * A11 - A01 * A10  # [...]
    y0 = A11 * x[..., 0] - A01 * x[..., 1]
    y1 = A00 * x[..., 1] - A10 * x[..., 0]
    return torch.stack([y0, y1], -1) / det[..., None]


class AffPlus2Tokens(TensorClass["frozen", "tensor_only"]):
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
    def fromg3splatter(cls, c0, c1, c2, r, s, t) -> "AffPlus2Tokens":
        """
        Convert from the Kaji-Ochiai parameterization of the g3splatter
        (S. Kaji and H. Ochiai, "A concise parametrization of affine transformation", SIAM J. Imaging Sci., 2016).

        The tokens are built as:
            A = exp_so2(r) @ exp_sym2(s)
            x = t
            u = cat(c0, c1, c2),
            i.e. u[..., :C] = c0, u[..., C:2C] = c1 and u[..., 2C:] = c2

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
        return _inv_apply2x2(self.A, self.x)

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

    def polar(self) -> tuple[Tensor, Tensor, Tensor]:
        """
        Polar decomposition A = a * R @ S in one pass, cheaper than accessing `a`, `R` and `S` separately.

        Returns:
            a (Tensor) [...]: isotropic scale factor sqrt(det(A))
            R (Tensor) [...,2,2]: rotation in SO(2)
            S (Tensor) [...,2,2]: symmetric positive definite with det(S) = 1
        """
        return polar2x2(self.A)
