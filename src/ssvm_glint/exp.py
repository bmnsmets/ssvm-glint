from torch import Tensor
import torch


@torch.compile()
def sinhc(x: torch.Tensor) -> torch.Tensor:
    """
    sinhc(Tensor) -> Tensor

    Computes the hyperbolic sinc function: sinh(x) / x.
    Numerically stable across float16, bfloat16, float32, and float64.

    Args:
        x (Tensor)
    Returns:
        Tensor: same shape as the input tensor.
    """
    small = x.abs() < 0.2  # safety zone sufficient for float16

    # direct computation
    x_safe = torch.where(small, 1, x)
    exact = torch.sinh(x_safe) / x_safe
    # Taylor series approximation for small x
    # sinh(x)/x = 1 + x^2 / 6 + x^4 / 120 + O(x^6)
    x2 = x**2
    taylor = 1.0 + x2 / 6.0 + (x2**2) / 120.0

    return taylor.where(small, exact)


@torch.compile()
def exp_so2(r: Tensor) -> Tensor:
    """
    exp_so2(Tensor) -> Tensor

    Compute so(2) -> SO(2) matrix exponential
    exp(
        [[0, -r],
         [r, 0]]
    ) =
    [[cos(r), -sin(r)],
     [sin(r), cos(r)]]
    """
    c, s = r.cos(), r.sin()  # [...], [...]
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])  # [2,2,...]
    return R.permute(*range(2, R.ndim), 0, 1).contiguous()  # [...,2,2]


@torch.compile()
def exp_sym2(s: Tensor) -> Tensor:
    """
    exp_sym2(Tensor) -> Tensor

    Compute sym(2) -> SPD(2) matrix exponential:
    exp(
        [[s1, s3],
         [s3, s2]]
    )
    =
    exp((s1+s2)/2) * (
        cosh(a) * [[1,0],[0,1]] + sinh(a)/a * [[(s1-s2)/2, s3],[s2,(s2-s1)/2]]
    )
    where a = 1/2 * sqrt((s1-s2)**2 + 4*s3**2 + eps).

    Args:
        s (Tensor): with shape [...,3]
    Returns:
        Tensor: with shape [...,2,2]
    """
    assert s.shape[-1] == 3
    s1, s2, s3 = s[..., 0], s[..., 1], s[..., 2]

    a = (
        1 / 2 * torch.sqrt((s1 - s2).pow(2) + 4 * s3.pow(2) + torch.finfo(s.dtype).eps)
    )  # [...]
    ch, shc = torch.cosh(a), sinhc(a)  # [...,1,1], [...,1,1]
    cheye = ch[..., None, None] * torch.eye(2).to(ch)  # [...,2,2]
    m = torch.stack(
        [torch.stack([(s1 - s2) / 2, s3]), torch.stack([s3, (s2 - s1) / 2])]
    )  # [2,2,...]
    m = m.permute(*range(2, m.ndim), 0, 1)  # [...,2,2]
    b = torch.exp((s1 + s2) / 2)[..., None, None]  # [...,1,1]
    return b * (cheye + shc[..., None, None] * m)
