import math
from types import SimpleNamespace
from typing import Literal, overload

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssvm_glint.config import git_root
from ssvm_glint.math import exp_so2, exp_sym2
from ssvm_glint.splatting.ssim import SSIM


@overload
def rasterize(
    c0: Tensor,
    c1: Tensor,
    c2: Tensor,
    t: Tensor,
    r: Tensor,
    s: Tensor,
    height: int,
    width: int,
    resolution: float | None = None,
    coverage: Literal[False] = False,
) -> Tensor: ...


@overload
def rasterize(
    c0: Tensor,
    c1: Tensor,
    c2: Tensor,
    t: Tensor,
    r: Tensor,
    s: Tensor,
    height: int,
    width: int,
    resolution: float | None = None,
    *,
    coverage: Literal[True],
) -> tuple[Tensor, Tensor]: ...


@torch.compile()
def rasterize(
    c0: Tensor,  # [B,N,C]
    c1: Tensor,  # [B,N,C]
    c2: Tensor,  # [B,N,C]
    t: Tensor,  # [B,N,2]
    r: Tensor,  # [B,N]
    s: Tensor,  # [B,N,3]
    height: int,  # output height in pixels
    width: int,  # output width in pixels
    resolution: float | None = None,  # pixels / spatial unit
    coverage: bool = False,  # return Gaussian coverage map in addition to the image
) -> Tensor | tuple[Tensor, Tensor]:
    """
    rasterize(c0, c1, c2, s, r, t, height, width, resolution) -> Tensor

    Rasterize the Gaussian stacks to a height x width grid.

    Args:
        c0 (Tensor): [B,N,C]
        c1 (Tensor): [B,N,C]
        c2 (Tensor): [B,N,C]
        t (Tensor): [B,N,2]
        r (Tensor): [B,N]
        s (Tensor): [B,N,3]
        height (int): height of the output image in pixels
        width (int): width of the output image in pixels
        resolution (float): Pixels per spatial unit. Default: max(height, width)
        coverage: (bool): return the Gaussian coverage map in addition to the image.
    Returns:
        Tensor [B,C,height,width]: rasterized image.
        -or-
        Tensor [B,C,height,width], Tensor [B,height,width]: rasterized image and coverage map.
    """
    B, N, _ = c0.shape
    assert c0.shape == c1.shape == c2.shape
    assert s.shape == (B, N, 3) and r.shape == (B, N) and t.shape == (B, N, 2)
    assert height > 0 and width > 0
    assert resolution is None or resolution > 0

    # Default resolution equates to the largest size of the pixel grid equating to 1 spatial unit.
    if resolution == None:
        resolution = max(height, width)

    # Inverse affine transforms from the geometric parameters t, r and s
    Ainv = torch.matmul(exp_sym2(-s), exp_so2(-r))  # [B,N,2,2]
    tinv = -Ainv @ t[..., None]  # [B,N,2,1]

    # (y,x) coordinate grid (0,0) is top left, (min(1,H/res),min(1,W/res)) bottom right.
    ys = torch.linspace(0, height / resolution, height)  # [H]
    xs = torch.linspace(0, width / resolution, width)  # [W]
    grid = torch.meshgrid(ys, xs, indexing="ij")  # [H],[W]
    grid = torch.stack(grid, dim=-1).contiguous()[..., None].to(Ainv)  # [H,W,2,1]

    # (y,x) coordinate grid for every wavelet [B,N,H,W,2]
    ps = (Ainv[:, :, None, None, :, :] @ grid + tinv[:, :, None, None, :, :])[..., 0]
    ps = ps.permute(4, 0, 2, 3, 1).clip(-20, 20).contiguous()  # [2,B,H,W,N]
    assert not ps.isnan().any()
    y, x = ps[0], ps[1]  # [B,H,W,N], [B,H,W,N]

    w0 = torch.exp(-(x**2 + y**2))  # [B,H,W,N]
    w1 = -2 * y / 0.7 * torch.exp(-((y / 0.7) ** 2 + x**2))  # [B,H,W,N]
    w2 = (4 * (y / 0.5) ** 2 - 1) * torch.exp(-((y / 0.5) ** 2 + x**2))  # [B,H,W,N]

    img0 = torch.einsum("bnc,bhwn->bchw", c0, w0)
    img1 = torch.einsum("bnc,bhwn->bchw", c1, w1)
    img2 = torch.einsum("bnc,bhwn->bchw", c2, w2)

    img = (img0 + img1 + img2).clip(0, 1)  # [B,C,H,W]

    if coverage:
        return img, w0.sum(dim=-1).mul(2).clip(0, 1)
    else:
        return img


@torch.compile()
def rasterize_partition(
    t: Tensor,  # [B,N,2]
    r: Tensor,  # [B,N]
    s: Tensor,  # [B,N,3]
    height: int,  # output height in pixels
    width: int,  # output width in pixels
    resolution: float | None = None,  # pixels / spatial unit
) -> Tensor:
    """
    rasterize(s, r, t, height, width, resolution) -> Tensor

    Rasterize the Gaussian stacks to a height x width grid.

    Args:
        t (Tensor): [B,N,2]
        r (Tensor): [B,N]
        s (Tensor): [B,N,3]
        height (int): height of the output image in pixels
        width (int): width of the output image in pixels
        resolution (float): Pixels per spatial unit. Default: max(height, width)
    Returns:
        Tensor [B,C,height,width]: rasterized image.
    """
    from matplotlib import colormaps

    B, N, _ = t.shape
    assert s.shape == (B, N, 3) and r.shape == (B, N) and t.shape == (B, N, 2)
    assert height > 0 and width > 0
    assert resolution is None or resolution > 0

    c = colormaps["rainbow"](torch.linspace(0, 1, N).numpy())[:, :3]  # [N,3]
    c = torch.tensor(list(c))
    c = torch.cat((c.float(), torch.tensor([[1.0, 1.0, 1.0]])), dim=0)
    c = 1 - c.to(t).unsqueeze(0).repeat(B, 1, 1)

    # Default resolution equates to the largest size of the pixel grid equating to 1 spatial unit.
    if resolution == None:
        resolution = max(height, width)

    # Inverse affine transforms from the geometric parameters t, r and s
    Ainv = torch.matmul(exp_sym2(-s), exp_so2(-r))  # [B,N,2,2]
    tinv = -Ainv @ t[..., None]  # [B,N,2,1]

    # (y,x) coordinate grid (0,0) is top left, (min(1,H/res),min(1,W/res)) bottom right.
    ys = torch.linspace(0, height / resolution, height)  # [H]
    xs = torch.linspace(0, width / resolution, width)  # [W]
    grid = torch.meshgrid(ys, xs, indexing="ij")  # [H],[W]
    grid = torch.stack(grid, dim=-1).contiguous()[..., None].to(Ainv)  # [H,W,2,1]

    # (y,x) coordinate grid for every wavelet [B,N,H,W,2]
    ps = (Ainv[:, :, None, None, :, :] @ grid + tinv[:, :, None, None, :, :])[..., 0]
    ps = ps.permute(4, 0, 2, 3, 1).clip(-20, 20).contiguous()  # [2,B,H,W,N]
    assert not ps.isnan().any()
    y, x = ps[0], ps[1]  # [B,H,W,N], [B,H,W,N]

    w0 = torch.exp(-(x**2 + y**2))  # [B,H,W,N]
    # Add a fake "Gaussian" for a neutral background
    w0 = torch.cat((w0, 0.0 * torch.ones_like(w0[..., :1])), dim=-1)  # [B,H,W,N+1]
    w0 = torch.softmax(5 * w0, -1).clip(0, 1)

    img = torch.einsum("bnc,bhwn->bchw", c, w0)

    return img


def splat_batch(
    batch: torch.Tensor,  # [B,C,H,W]
    nstacks: int | Literal["auto"] = "auto",  # number of Gaussian stacks to output
    initial_stack_multiplier: int = 4,  # start with initial_stack_multiplier * nstacks Gaussian stacks
    learning_rate: float = 0.05,
    learning_rate_gamma: float = 0.95,
    iterations: int = 100,
    l1_loss_weight: float = 0.9,
    ssim_loss_weight: float = 0.1,
    initialization_inset: float = 0.05,
    coverage: bool = True,
):
    B, C, H, W = batch.shape
    if nstacks == "auto":
        nstacks = H * W // 32
    N = initial_stack_multiplier * nstacks

    # initialize SSIM module for loss calculation
    ssim = SSIM(
        data_range=1.0,
        size_average=False,
        channel=C,
        nonnegative_ssim=True,
        K=(0.1, 0.3) if batch.dtype == torch.float16 else (0.02, 0.06),
        win_size=1 + 2 * min(5, max(H, W) // 16),
        win_sigma=min(1.5, max(H, W) / 70),
    )

    ### 1. Initialization ###

    # initialize color values
    lb, ub = batch.aminmax()
    lb, ub = lb.detach(), ub.detach()
    lb, ub = lb + (ub - lb) / 5, ub - (ub - lb) / 5
    c0 = torch.distributions.Uniform(low=lb, high=ub).sample((B, N, C)).to(batch)
    c1 = torch.distributions.Normal(loc=0, scale=0.05).sample((B, N, C)).to(batch)
    c2 = torch.distributions.Normal(loc=0, scale=0.05).sample((B, N, C)).to(batch)

    # initialize translations
    inset = initialization_inset
    y = (
        torch.distributions.Uniform(low=inset, high=H / max(H, W) - inset)
        .sample((B, N))
        .to(batch)
    )
    x = (
        torch.distributions.Uniform(low=inset, high=W / max(H, W) - inset)
        .sample((B, N))
        .to(batch)
    )
    t = torch.stack((y, x), dim=-1).to(batch)

    # initialize rotations
    r = (
        torch.distributions.Uniform(low=-torch.pi, high=torch.pi)
        .sample((B, N))
        .to(batch)
    )

    # initialize scales and shears
    lscale = -math.log(max(H, W))
    hscale = -2
    s12 = torch.distributions.Uniform(low=lscale, high=hscale).sample((B, N, 2))
    s3 = torch.zeros(B, N, 1)
    s = torch.cat((s12, s3), dim=-1).to(batch)

    for z in (c0, c1, c2, t, r, s):
        z.requires_grad_()

    ### 2. First optimization run ###
    optimizer = torch.optim.AdamW(
        [c0, c1, c2, t, r, s],
        lr=learning_rate,
        betas=(0.5, 0.6),
        weight_decay=0,
        eps=torch.finfo(batch.dtype).eps,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=learning_rate_gamma
    )

    for i in range(iterations // 2):
        if coverage:
            preds, covmap = rasterize(c0, c1, c2, t, r, s, H, W, coverage=True)
            covloss = (1 - covmap).abs().mean(dim=(-2, -1)).sum()
        else:
            preds = rasterize(c0, c1, c2, t, r, s, H, W, coverage=False)
            covloss = 0

        l1_loss = (batch - preds).abs().mean(dim=(-2, -1)).sum()
        ssim_loss = (1 - ssim(batch, preds)).sum()
        dloss = l1_loss_weight * l1_loss + ssim_loss_weight * ssim_loss

        closs = (c0 + c1 + c2).abs().mean(dim=-2).sum()
        isoloss = (s[..., 0] - s[..., 1]).pow(2).mean(dim=-1).sum()
        shearloss = s[..., 2].abs().mean(dim=-1).sum()

        loss = dloss + 0.8 * closs + 0.01 * isoloss + 0.2 * shearloss + covloss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    ### 3. Culling ###
    # keep nstacks most significant stacks
    def _cull(a, idx):
        return torch.stack([a[i, idx[i]] for i in range(len(a))], dim=0)

    optimizer.zero_grad()
    preds = rasterize(c0, c1, c2, t, r, s, H, W)
    l1_loss = (batch - preds).abs().sum()
    ssim_loss = (1 - ssim(batch, preds)).sum()
    dloss = l1_loss_weight * l1_loss + ssim_loss_weight * ssim_loss
    dloss.backward()

    assert c0.grad is not None
    significance = c0.grad.detach().abs().sum(dim=-1) + s.detach().sum(dim=-1)  # [B,N]
    significance = s[..., 0] + s[..., 1]
    _, indices = torch.sort(significance, dim=-1, descending=True)  # [B,N]
    indices = indices[:, :nstacks]  # [B,nstacks]

    c0 = _cull(c0.detach(), indices)
    c1 = _cull(c1.detach(), indices)
    c2 = _cull(c2.detach(), indices)
    t = _cull(t.detach(), indices)
    r = _cull(r.detach(), indices)
    s = _cull(s.detach(), indices)

    ### 4. Second optimization run on data loss only ###
    for z in (c0, c1, c2, t, r, s):
        z.requires_grad_()

    optimizer = torch.optim.AdamW(
        [c0, c1, c2, t, r, s],
        lr=learning_rate,
        betas=(0.5, 0.6),
        weight_decay=0,
        eps=torch.finfo(batch.dtype).eps,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=learning_rate_gamma
    )

    for i in range((iterations + 1) // 2):
        if coverage:
            preds, covmap = rasterize(c0, c1, c2, t, r, s, H, W, coverage=True)
            covloss = (1 - covmap).abs().mean(dim=(-2, -1)).sum()
        else:
            preds = rasterize(c0, c1, c2, t, r, s, H, W, coverage=False)
            covloss = 0

        l1_loss = (batch - preds).abs().mean(dim=(-2, -1)).sum()
        ssim_loss = (1 - ssim(batch, preds)).sum()
        dloss = l1_loss_weight * l1_loss + ssim_loss_weight * ssim_loss + covloss
        optimizer.zero_grad()
        dloss.backward()
        optimizer.step()
        scheduler.step()

    return c0.detach(), c1.detach(), c2.detach(), t.detach(), r.detach(), s.detach()


def splat_dataloader(loader: DataLoader, device, **kwargs):
    c0s, c1s, c2s, ts, rs, ss, lbls = [], [], [], [], [], [], []
    for images, labels in tqdm(loader):
        images = images.to(device)  # [B,C,H,W]
        c0, c1, c2, t, r, s = splat_batch(images, **kwargs)
        c0s.append(c0.cpu())
        c1s.append(c1.cpu())
        c2s.append(c2.cpu())
        ts.append(t.cpu())
        rs.append(r.cpu())
        ss.append(s.cpu())
        lbls.append(labels.cpu())

    return (
        torch.cat(c0s),
        torch.cat(c1s),
        torch.cat(c2s),
        torch.cat(ts),
        torch.cat(rs),
        torch.cat(ss),
        torch.cat(lbls),
    )


def plot_grid(x, y, title: str | None = None):
    import matplotlib.pyplot as plt

    if isinstance(x, Tensor):
        x = x.detach().cpu().float().permute(0, 2, 3, 1)
    else:
        x = [z.detach().cpu().float() for z in x]
        x = [z[0] if z.ndim == 4 else z for z in x]
        x = [z.permute(1, 2, 0) for z in x]

    if isinstance(y, Tensor):
        y = y.detach().cpu()
        y = [int(y[i].item()) for i in range(y.shape[0])]
    n = len(x)
    nrows, ncols = math.ceil(n / 10), min(10, n)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(1.5 * ncols, 1.7 * nrows + 0.5),
        subplot_kw={"xticks": [], "yticks": []},
    )
    fig.patch.set_visible(False)
    if title != None:
        fig.suptitle(title, fontsize=18, color="white")

    for i, ax in enumerate(axs.flat):
        if i < n:
            ax.imshow(x[i], cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(str(y[i]), fontsize=14, color="white", fontweight="bold")
        ax.axis("off")
        ax.patch.set_visible(False)

    plt.tight_layout()
    plt.show()


def test(
    batch: Tensor,
    repeats: int = 10,
    nstacks: int | Literal["auto"] = "auto",
    learning_rate: float = 0.05,
    coverage: bool = True,
):
    from time import perf_counter_ns

    import numpy as np
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = 32

    _, C, H, W = batch.shape

    ssim = SSIM(
        data_range=1.0,
        size_average=False,
        channel=C,
        nonnegative_ssim=True,
        K=(0.1, 0.3) if batch.dtype == torch.float16 else (0.02, 0.06),
        win_size=1 + 2 * min(5, max(H, W) // 16),
        win_sigma=min(1.5, max(H, W) / 70),
    )

    dlosses, times, nans = [], [], []
    for i in tqdm(range(repeats + 1)):
        t0 = perf_counter_ns()
        c0, c1, c2, t, r, s = splat_batch(
            batch, nstacks=nstacks, learning_rate=learning_rate, coverage=coverage
        )
        z = rasterize(c0, c1, c2, t, r, s, H, W)

        l1_loss = (batch - z).abs().mean(dim=(-3, -2, -1))
        ssim_loss = 1 - ssim(batch, z)
        dloss = (0.9 * l1_loss + 0.1 * ssim_loss).tolist()

        if batch.device == "cuda":
            torch.cuda.synchronize()

        if i == 0:
            continue

        dlosses.extend(dloss)
        times.append((perf_counter_ns() - t0) / 1e9)
        nans.append(z.isnan().any(dim=(1, 2, 3)).sum().item())

    print(
        f"\
dloss = {np.mean(dlosses) * 1000:.2f}e-3 ± {np.var(dlosses) * 1000:.2f}e-3, \
time = {np.mean(times):.2f}s, \
nans = {np.sum(nans)}/{batch.shape[0] * repeats}"
    )


def mnist_test():
    import torchvision
    from torchvision.transforms.v2 import Compose, PILToTensor, ToDtype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    dataroot = git_root() / "data"
    ds = torchvision.datasets.MNIST(
        dataroot,
        transform=Compose([PILToTensor(), ToDtype(torch.float32, scale=True)]),
        download=True,
    )
    idx = range(512)
    batch = torch.stack([ds[i][0] for i in idx]).to(device)
    test(batch, repeats=5, nstacks=16, coverage=False)
    test(batch, repeats=5, nstacks=16, coverage=False)
    test(batch, repeats=5, nstacks=16, coverage=False)


def mnist_example():
    import torchvision
    from torchvision.transforms.v2 import Compose, PILToTensor, ToDtype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    dataroot = git_root() / "data"
    ds = torchvision.datasets.MNIST(
        dataroot,
        transform=Compose([PILToTensor(), ToDtype(torch.float16, scale=True)]),
        download=True,
    )
    indices = [
        1,
        3,
        5,
        7,
        2,
        0,
        13,
        15,
        17,
        4,
    ]  # + [21, 6, 16, 10, 9, 11, 18, 29, 31, 19]
    x = torch.stack([ds[i][0] for i in indices]).to(device)
    y = torch.tensor([ds[i][1] for i in indices]).to(device)

    plot_grid(x, y, "MNIST")
    nstacks = 16
    c0, c1, c2, t, r, s = splat_batch(x, nstacks=nstacks, coverage=False)
    h, w = 128, 128
    z = rasterize(c0, c1, c2, t, r, s, h, w)
    plot_grid(z, y, f"g3splatter with {nstacks} stacks @{h}x{w}")

    partitions = rasterize_partition(t, r, s, h, w)
    zpart = 1 - z * partitions
    plot_grid(
        zpart, y, f"g3splatter with {nstacks} stacks @{h}x{w} with colored splats"
    )


def fashion_test():
    import torchvision
    from torchvision.transforms.v2 import PILToTensor, ToDtype, Compose

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    dataroot = git_root() / "data"
    ds = torchvision.datasets.FashionMNIST(
        dataroot,
        transform=Compose([PILToTensor(), ToDtype(torch.float32, scale=True)]),
        download=True,
    )
    idx = range(512)
    batch = torch.stack([ds[i][0] for i in idx]).to(device)
    test(batch.to(torch.float32), repeats=5, nstacks=32, coverage=False)
    test(batch.to(torch.bfloat16), repeats=5, nstacks=32, coverage=False)
    test(batch.to(torch.float16), repeats=5, nstacks=32, coverage=False)


def fashion_example():
    import torchvision
    from torchvision.transforms.v2 import PILToTensor, ToDtype, Compose
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    dataroot = git_root() / "data"
    ds = torchvision.datasets.FashionMNIST(
        dataroot,
        transform=Compose([PILToTensor(), ToDtype(torch.float32, scale=True)]),
        download=True,
    )
    indices = [1, 16, 5, 3, 19, 8, 18, 6, 23, 0] + [2, 21, 7, 20, 22, 9, 32, 14, 35, 11]
    x = torch.stack([ds[i][0] for i in indices]).to(device)
    y = torch.tensor([ds[i][1] for i in indices]).to(device)

    plot_grid(x, y, "FashionMNIST")

    c0, c1, c2, t, r, s = splat_batch(x, nstacks=16, coverage=False)
    h, w = 128, 128
    z = rasterize(c0, c1, c2, t, r, s, h, w)
    plot_grid(z, y, f"g3splatter with 16 stacks @{h}x{w}")

    # c0, c1, c2, t, r, s = splat_batch(x, nstacks=32, coverage=False)
    # z = rasterize(c0, c1, c2, t, r, s, h, w)
    # plot_grid(z, y, f"g3splatter with 32 stacks @{h}x{w}")


def cifar_test():
    import torchvision
    from torchvision.transforms.v2 import PILToTensor, ToDtype, Compose

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    dataroot = git_root() / "data"
    ds = torchvision.datasets.CIFAR10(
        dataroot,
        transform=Compose([PILToTensor(), ToDtype(torch.float32, scale=True)]),
        download=True,
    )
    idx = range(512)
    batch = torch.stack([ds[i][0] for i in idx]).to(device)
    test(batch.to(torch.float32), repeats=3, nstacks=32)
    # test(batch.to(torch.bfloat16), repeats=5, nstacks=32)
    test(batch.to(torch.float16), repeats=3, nstacks=32)


def cifar_example():
    import torchvision
    from torchvision.transforms.v2 import PILToTensor, ToDtype, Compose
    import warnings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    dataroot = git_root() / "data"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ds = torchvision.datasets.CIFAR10(
            dataroot,
            transform=Compose([PILToTensor(), ToDtype(torch.float32, scale=True)]),
            download=True,
        )
    indices = [29, 4, 6, 9, 3, 27, 0, 7, 8, 1] + [30, 5, 13, 17, 10, 40, 19, 11, 62, 2]
    x = torch.stack([ds[i][0] for i in indices]).to(device)
    y = torch.tensor([ds[i][1] for i in indices]).to(device)

    plot_grid(x, y, "CIFAR10")

    c0, c1, c2, t, r, s = splat_batch(x, nstacks=24)
    h, w = 128, 128
    z = rasterize(c0, c1, c2, t, r, s, h, w)
    plot_grid(z, y, f"g3splatter with 24 stacks @{h}x{w}")

    # c0, c1, c2, t, r, s = splat_batch(x, nstacks=48)
    # z = rasterize(c0, c1, c2, t, r, s, h, w)
    # plot_grid(z, y, f"g3splatter with 48 stacks @{h}x{w}")


def imagenette_example():
    import torchvision
    from torchvision.transforms.v2 import PILToTensor, ToDtype, Compose

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 64

    dataroot = git_root() / "data"
    ds = torchvision.datasets.Imagenette(
        dataroot,
        transform=Compose([PILToTensor(), ToDtype(torch.float16, scale=True)]),
        download=True,
        size="160px",
    )
    indices = [0, 963, 1918, 2911, 3769, 4710, 5666, 6627, 7558, 8509] + [
        1,
        964,
        1919,
        2912,
        3770,
        4711,
        5667,
        6628,
        7559,
        8510,
    ]
    x = [ds[i][0].to(device) for i in indices]
    y = torch.tensor([ds[i][1] for i in indices]).to(device)

    plot_grid(x, y, "Imagenette")

    splat_config = SimpleNamespace(
        nstacks=128,
        coverage=True,
        initial_stack_multiplier=4,
        learning_rate=0.08,
        learning_rate_gamma=0.98,
        iterations=150,
        initialization_inset=0.15,
    )

    splats = [
        splat_batch(x[i].unsqueeze(0), **vars(splat_config)) for i in range(len(x))
    ]
    zs = [rasterize(*splats[i], *x[i].shape[-2:]) for i in range(len(splats))]
    plot_grid(zs, y, f"g3splatter with {splat_config.nstacks} stacks")


if __name__ == "__main__":
    mnist_example()
