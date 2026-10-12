import argparse
import warnings
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssvm_glint.config import load_config_file, print_config_tree
from ssvm_glint.data import (
    get_cifar10_dataloaders,
    get_fashion_dataloaders,
    get_mnist_dataloaders,
)
from ssvm_glint.splatting import rasterize, splat_batch
from ssvm_glint.splatting.ssim import SSIM

PARAMS = ("c0", "c1", "c2", "t", "r", "s")


def reconstruction_loss(
    images: Tensor,
    params: tuple[Tensor, ...],
    l1_loss_weight: float,
    ssim_loss_weight: float,
) -> Tensor:
    """
    Per-sample data loss `l1_loss_weight * L1 + ssim_loss_weight * (1 - SSIM)` of a splat, with the
    SSIM settings of `splat_batch`. NaN losses are mapped to +inf so diverged splats rank last.

    Args:
        images (Tensor): [B,C,H,W] the splatted images
        params (tuple[Tensor, ...]): (c0, c1, c2, t, r, s) as returned by `splat_batch`

    Returns:
        Tensor [B]: reconstruction loss per sample.
    """
    _, C, H, W = images.shape
    ssim = SSIM(
        data_range=1.0,
        size_average=False,
        channel=C,
        nonnegative_ssim=True,
        K=(0.1, 0.3) if images.dtype == torch.float16 else (0.02, 0.06),
        win_size=1 + 2 * min(5, max(H, W) // 16),
        win_sigma=min(1.5, max(H, W) / 70),
    )
    c0, c1, c2, t, r, s = params
    preds = rasterize(c0, c1, c2, t, r, s, H, W)  # [B,C,H,W]
    l1_loss = (images - preds).abs().mean(dim=(-3, -2, -1))  # [B]
    ssim_loss = 1 - ssim(images, preds)  # [B]
    loss = l1_loss_weight * l1_loss + ssim_loss_weight * ssim_loss
    return loss.nan_to_num(nan=torch.inf)


@torch.no_grad()
def splat_best(
    loader: DataLoader,
    repeats: int,
    keep: int,
    device: torch.device,
    output_dtype: torch.dtype,
    splatter_kwargs: dict,
) -> dict[str, Tensor]:
    """
    Splat every sample `repeats` times and keep the `keep` splats with the lowest reconstruction loss.

    The repeats are done per batch, so the selection is per sample even if the loader shuffles.

    Returns:
        dict: "c0", "c1", "c2" [K*M,N,C], "t" [K*M,N,2], "r" [K*M,N], "s" [K*M,N,3] and "lbl" [K*M],
        with M the number of samples and K = `keep`. Rows are rank-major: rows j*M to (j+1)*M hold
        the j-th best splat of every sample.
    """
    out = {k: [] for k in (*PARAMS, "lbl")}
    for images, labels in tqdm(loader):
        images = images.to(device)  # [B,C,H,W]
        with torch.enable_grad():
            runs = [splat_batch(images, **splatter_kwargs) for _ in range(repeats)]
        losses = torch.stack(
            [
                reconstruction_loss(
                    images,
                    run,
                    splatter_kwargs["l1_loss_weight"],
                    splatter_kwargs["ssim_loss_weight"],
                )
                for run in runs
            ],
            dim=1,
        )  # [B,R]
        idx = losses.topk(keep, dim=1, largest=False).indices  # [B,K], best first
        rows = torch.arange(len(images), device=device)[:, None]  # [B,1]
        for i, k in enumerate(PARAMS):
            p = torch.stack([run[i] for run in runs], dim=1)  # [B,R,N,...]
            out[k].append(p[rows, idx].cpu().to(output_dtype))  # [B,K,N,...]
        out["lbl"].append(labels.cpu()[:, None].expand(-1, keep))  # [B,K]

    # [M,K,...] -> [K*M,...]
    return {k: torch.cat(v).transpose(0, 1).flatten(0, 1) for k, v in out.items()}


if __name__ == "__main__" and not hasattr(__builtins__, "__IPYTHON__"):
    parser = argparse.ArgumentParser(description="Splatting")
    parser.add_argument("config_file", type=Path, help="Configuration file path")
    args = parser.parse_args()

    dtypes = {"float32": torch.float32, "float16": torch.float16}

    # Load the config from file.
    config = load_config_file(args.config_file)

    # Resolve relative paths.
    config.dataroot = (args.config_file.parent / config.dataroot).resolve()
    config.output_train = (args.config_file.parent / config.output_train).resolve()
    config.output_test = (args.config_file.parent / config.output_test).resolve()

    config.output_train.parent.mkdir(parents=True, exist_ok=True)
    config.output_test.parent.mkdir(parents=True, exist_ok=True)

    # Select device.
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve dtypes.
    config.output_dtype = dtypes[config.output_dtype]
    config.train_dtype = dtypes[config.train_dtype]

    print_config_tree(config)

    torch.manual_seed(config.seed)

    if config.dataset == "mnist":
        train_loader, test_loader = get_mnist_dataloaders(
            config.dataroot,
            config.batch_size,
            config.batch_size,
            dtype=config.train_dtype,
        )
    elif config.dataset == "fashionmnist":
        train_loader, test_loader = get_fashion_dataloaders(
            config.dataroot,
            config.batch_size,
            config.batch_size,
            dtype=config.train_dtype,
        )
    elif config.dataset == "cifar10":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            train_loader, test_loader = get_cifar10_dataloaders(
                config.dataroot,
                config.batch_size,
                config.batch_size,
                dtype=config.train_dtype,
            )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    print("Splatting train set:")
    save_file(
        splat_best(
            train_loader,
            config.train_repeats,
            config.train_keep,
            config.device,
            config.output_dtype,
            vars(config.splatter),
        ),
        config.output_train,
    )

    print("Splatting test set:")
    save_file(
        splat_best(
            test_loader,
            config.test_repeats,
            config.test_keep,
            config.device,
            config.output_dtype,
            vars(config.splatter),
        ),
        config.output_test,
    )
