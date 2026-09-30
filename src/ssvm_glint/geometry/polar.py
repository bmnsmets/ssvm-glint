import torch
from torch import Tensor


def _polar2x2(A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Polar decomposition A = a * R @ S of A in GL^+(2), in closed form.
    For A = R @ P with P symmetric: A00 + A11 = cos(r) tr(P) and A10 - A01 = sin(r) tr(P), with tr(P) > 0.
    S = R^T A / a is computed elementwise, using that it is symmetric.

    Args:
        A (Tensor): with shape [...,2,2] and det(A) > 0
    Returns:
        a (Tensor) [...]: isotropic scale factor sqrt(det(A))
        R (Tensor) [...,2,2]: rotation in SO(2)
        S (Tensor) [...,2,2]: symmetric positive definite with det(S) = 1
    """
    A00, A01, A10, A11 = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
    c = A00 + A11  # [...]
    s = A10 - A01  # [...]
    n = torch.hypot(c, s)  # [...]
    a = (A00 * A11 - A01 * A10).sqrt()  # [...]
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    R = R / n[..., None, None]  # [...,2,2]
    w = 1 / (n * a)  # [...]
    S00 = (c * A00 + s * A10) * w
    S01 = (c * A01 + s * A11) * w
    S11 = (c * A11 - s * A01) * w
    S = torch.stack([torch.stack([S00, S01], -1), torch.stack([S01, S11], -1)], -2)
    return a, R, S


# Separate compiled entry points so that the unused factors are eliminated from each graph.
@torch.compile()
def polar2x2(A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Polar decomposition A = a * R @ S of A in GL^+(2), see `_polar2x2`.
    """
    return _polar2x2(A)


@torch.compile()
def polar2x2_a(A: Tensor) -> Tensor:
    """
    Isotropic scale factor a = sqrt(det(A)) of the polar decomposition A = a * R @ S.
    """
    return _polar2x2(A)[0]


@torch.compile()
def polar2x2_R(A: Tensor) -> Tensor:
    """
    Rotation factor R of the polar decomposition A = a * R @ S.
    """
    return _polar2x2(A)[1]


@torch.compile()
def polar2x2_S(A: Tensor) -> Tensor:
    """
    SL(2) factor S of the polar decomposition A = a * R @ S.
    """
    return _polar2x2(A)[2]
