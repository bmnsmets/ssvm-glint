from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
import torchvision
from safetensors import safe_open
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, PILToTensor, ToDtype

from ssvm_glint.tokens import AffPlus2Tokens, random_affine


def get_mnist_dataloaders(
    root: Path,
    train_batch_size: int,
    test_batch_size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
):
    trainset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=Compose([PILToTensor(), ToDtype(dtype, scale=True)]),
    )
    testset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=Compose([PILToTensor(), ToDtype(dtype, scale=True)]),
    )

    xs = torch.stack([x for x, _ in trainset])
    ys = torch.stack([torch.tensor(y) for _, y in trainset])
    trainset = torch.utils.data.TensorDataset(xs.to(device), ys.to(device))
    xs = torch.stack([x for x, _ in testset])
    ys = torch.stack([torch.tensor(y) for _, y in testset])
    testset = torch.utils.data.TensorDataset(xs.to(device), ys.to(device))

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


class AffPlus2TokenDataset(Dataset):
    """
    Dataset of (tokens, label) pairs, where sample i is `(tokens[i], labels[i])`.

    Implements `__getitems__` so that a DataLoader fetches a whole batch with a single indexing operation;
    use it with `collate_fn=_no_collate` since the batch is already assembled.

    If given, `transform` is applied to the tokens of every fetched sample or batch, e.g. a random augmentation.
    """

    def __init__(
        self,
        tokens: AffPlus2Tokens,
        labels: Tensor,
        transform: Callable[[AffPlus2Tokens], AffPlus2Tokens] | None = None,
    ):
        assert tokens.batch_size[0] == labels.shape[0]
        self.tokens = tokens
        self.labels = labels
        self.transform = transform

    def _transform(self, tokens: AffPlus2Tokens) -> AffPlus2Tokens:
        return tokens if self.transform is None else self.transform(tokens)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[AffPlus2Tokens, Tensor]:
        return self._transform(self.tokens[index]), self.labels[index]

    def __getitems__(self, idx: list[int]) -> tuple[AffPlus2Tokens, Tensor]:
        idx_t = torch.as_tensor(idx, device=self.labels.device)
        return self._transform(self.tokens[idx_t]), self.labels[idx_t]


def _no_collate(batch):
    return batch


def _g3splat_stats(tokens: AffPlus2Tokens) -> dict[str, Tensor | tuple[Tensor, Tensor]]:
    """
    Dataset-wide normalization statistics, computed in float32 over every token of `tokens`.

    `u` gets a scalar mean and standard deviation over all its elements. `t` gets a 2D mean vector
    but a single scalar scale, the standard deviation over both components jointly, i.e.
    sqrt((var(t[0]) + var(t[1])) / 2). The scale is deliberately one scalar rather than one per
    component, so the rescaling stays isotropic and does not distort the geometry; each component
    then has unit variance up to how much the two differ from each other.

    log(a) is only centered, never rescaled, so it has no std. The scale factor `a` enters
    `affine_invariant()` through A^-1, so widening log(a) directly widens that feature: on the
    MNIST splats, standardizing log(a) from its raw std of 0.20 to 1.0 stretches `a` from
    [0.04, 0.30] to [0.02, 452] and gives `affine_invariant()` tails out to |.| ~ 200. Keeping the
    raw spread leaves A^-1 well conditioned.

    Args:
        tokens (AffPlus2Tokens): the split the statistics are taken over, normally the training set

    Returns:
        dict: "u" and "t" mapping to a (mean, std) pair, "loga" to a mean.
    """
    eps = torch.finfo(torch.float32).eps
    tokens = tokens.to(torch.float32)
    u, t, loga = tokens.u, tokens.t, tokens.a.log()
    tm = t.mean(dim=tuple(range(t.ndim - 1)))
    return {
        "u": (u.mean(), u.std().clamp_min(eps)),
        "t": (tm, (t - tm).std().clamp_min(eps)),
        "loga": loga.mean(),
    }


def _normalize_g3splat(
    tokens: AffPlus2Tokens, stats: dict[str, Tensor | tuple[Tensor, Tensor]]
) -> AffPlus2Tokens:
    """
    Normalize `u`, `t = A^-1 x` and log(a) with the statistics from `_g3splat_stats`.

    The rotation R and the SL(2) factor S of the polar decomposition A = a * R @ S are left
    untouched; the isotropic scale is only recentered, as a <- a / exp(mean). The group element is
    then rebuilt from the normalized parts, so that the result satisfies
    `result.t == (tokens.t - mean) / std` and `result.a.log()` is centered.

    Returns:
        AffPlus2Tokens: same batch size, dtypes and device as `tokens`.
    """
    gdtype, dtype = tokens.g.dtype, tokens.u.dtype
    tokens = tokens.to(torch.float32)
    um, ustd = stats["u"]
    tm, tstd = stats["t"]
    lm = stats["loga"]

    a, R, S = tokens.polar()
    a = (a.log() - lm).exp()  # [M,N]
    A = a[..., None, None] * torch.matmul(R, S)  # [M,N,2,2]
    t = (tokens.t - tm) / tstd  # [M,N,2]
    x = torch.matmul(A, t[..., None])  # [M,N,2,1]

    g = torch.cat((A, x), dim=-1)  # [M,N,2,3]
    u = (tokens.u - um) / ustd
    return AffPlus2Tokens(g=g.to(gdtype), u=u.to(dtype))


def load_g3splat_dataset(
    path: str | Path,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> AffPlus2TokenDataset:
    """
    Load a g3splat safetensors file as an AffPlus2TokenDataset with tokens of batch size [M,N] and int64 labels [M].
    The group elements are kept in float32, since in lower precision det(A) of strongly sheared tokens
    cancels to <= 0; only `u` is cast to `dtype`.
    """
    with safe_open(path, framework="pt", device=str(device or "cpu")) as f:
        params = {k: f.get_tensor(k).float() for k in ("c0", "c1", "c2", "r", "s", "t")}
        labels = f.get_tensor("lbl").long()
    tokens = AffPlus2Tokens.fromg3splatter(**params)
    tokens = tokens.replace(u=tokens.u.to(dtype))
    return AffPlus2TokenDataset(tokens, labels)


def get_g3splat_dataloaders(
    train_path: str | Path,
    test_path: str | Path,
    train_batch_size: int,
    test_batch_size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    augment: dict[str, float] | None = None,
):
    """
    DataLoaders yielding (AffPlus2Tokens [B,N], labels [B]) batches.

    The semantic part `u`, the position `t = A^-1 x` and the log isotropic scale log(a) are
    normalized with statistics taken over the *training* set only, so the test set is mapped
    through the same transform rather than being normalized by its own statistics.

    `u` and `t` are standardized to unit variance. log(a) is only centered: rescaling it would
    widen `a` and with it `affine_invariant()`, which depends on A^-1. See `_g3splat_stats`.

    If given, `augment` holds the ranges `rotate`, `shear`, `log_scale` and `translate` of
    `random_affine`, in raw splat coordinates, and every training batch gets a fresh random transform.
    The test set is never augmented. The normalization commutes with the action up to the translation:
    with c = exp(mean log a) and tstd the scale of `t`, h = (H, b) on raw tokens is h = (H, b / (c tstd))
    on normalized tokens, so the augmentation is applied to the normalized tokens directly.
    """
    trainset = load_g3splat_dataset(train_path, dtype=dtype, device=device)
    testset = load_g3splat_dataset(test_path, dtype=dtype, device=device)

    stats = _g3splat_stats(trainset.tokens)
    transform = None
    if augment is not None:
        loga_mean, tstd = stats["loga"], stats["t"][1]
        assert isinstance(loga_mean, Tensor)
        translate_scale = 1 / (loga_mean.exp() * tstd).item()  # 1 / (c tstd)
        transform = partial(random_affine, **augment, translate_scale=translate_scale)
    trainset = AffPlus2TokenDataset(
        _normalize_g3splat(trainset.tokens, stats), trainset.labels, transform
    )
    testset = AffPlus2TokenDataset(
        _normalize_g3splat(testset.tokens, stats), testset.labels
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_no_collate,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_no_collate,
    )
    return train_loader, test_loader


def _load_rasterized(
    path: str | Path,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.utils.data.TensorDataset:
    with safe_open(path, framework="pt", device=str(device or "cpu")) as f:
        img = f.get_tensor("img")  # [M,28,28] uint8
        labels = f.get_tensor("lbl").long()  # [M]
    return torch.utils.data.TensorDataset(img[:, None].to(dtype) / 255, labels)


def get_rasterized_dataloaders(
    train_path: str | Path,
    test_path: str | Path,
    train_batch_size: int,
    test_batch_size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
):
    """
    DataLoaders for the rasterized splat files written by scripts/rasterize.py, yielding
    (images [B,1,28,28] in [0,1], labels [B]) batches like `get_mnist_dataloaders`.
    """
    train_loader = torch.utils.data.DataLoader(
        _load_rasterized(train_path, dtype, device),
        batch_size=train_batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        _load_rasterized(test_path, dtype, device),
        batch_size=test_batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def get_fashion_dataloaders(
    root: Path,
    train_batch_size: int,
    test_batch_size: int,
    dtype: torch.dtype = torch.float32,
):
    trainset = torchvision.datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=Compose([PILToTensor(), ToDtype(dtype, scale=True)]),
    )
    testset = torchvision.datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=Compose([PILToTensor(), ToDtype(dtype, scale=True)]),
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def get_cifar10_dataloaders(
    root: Path,
    train_batch_size: int,
    test_batch_size: int,
    dtype: torch.dtype = torch.float32,
):
    trainset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=Compose([PILToTensor(), ToDtype(dtype, scale=True)]),
    )
    testset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=Compose([PILToTensor(), ToDtype(dtype, scale=True)]),
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
