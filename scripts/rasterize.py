import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor
from tqdm import tqdm

from ssvm_glint.splatting import rasterize

PARAMS = ("c0", "c1", "c2", "t", "r", "s")
BATCH_SIZE = 1024


@torch.no_grad()
def rasterize_splats(splats: dict[str, Tensor], device: torch.device) -> Tensor:
    """
    Rasterize every splat to a 28x28 8-bit image, keeping the row order.

    Args:
        splats (dict): "c0", "c1", "c2" [M,N,1], "t" [M,N,2], "r" [M,N], "s" [M,N,3] as saved by splat.py

    Returns:
        Tensor [M,28,28]: uint8 images, row i rasterized from splat i.
    """
    M = len(splats["c0"])
    imgs = []
    for i in tqdm(range(0, M, BATCH_SIZE)):
        c0, c1, c2, t, r, s = (splats[k][i : i + BATCH_SIZE].to(device, torch.float32) for k in PARAMS)
        img = rasterize(c0, c1, c2, t, r, s, 28, 28)  # [B,1,28,28] in [0,1]
        imgs.append((img[:, 0] * 255).round().to(torch.uint8).cpu())  # [B,28,28]
    return torch.cat(imgs)


if __name__ == "__main__" and not hasattr(__builtins__, "__IPYTHON__"):
    parser = argparse.ArgumentParser(description="Rasterize splats to 28x28 8-bit images")
    parser.add_argument("splat_files", type=Path, nargs="+", help="Splat safetensors files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for path in args.splat_files:
        out = path.parent.parent / "mnist-splat-rasterized" / f"{path.stem}-rasterized.safetensors"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Rasterizing {path} -> {out}")
        splats = load_file(path)
        save_file({"img": rasterize_splats(splats, device), "lbl": splats["lbl"]}, out)
