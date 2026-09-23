import sys
import argparse
from pathlib import Path
import torch
import torchvision
from torchvision.transforms.v2 import PILToTensor, ToDtype, Compose
import warnings
from safetensors.torch import save_file

sys.path.append("..")
from utils import (
    load_config_file,
    print_config_tree,
    get_mnist_dataloaders,
    get_fashion_dataloaders,
    get_cifar10_dataloaders,
)
from g3splatter import splat_dataloader


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

    c0s, c1s, c2s, ts, rs, ss, lbls = [], [], [], [], [], [], []
    for i in range(config.repeats):
        print(f"Splatting {i+1}/{config.repeats}:")
        c0, c1, c2, t, r, s, lbl = splat_dataloader(
            train_loader, device=config.device, **vars(config.splatter)
        )
        c0s.append(c0.cpu().to(config.output_dtype))
        c1s.append(c1.cpu().to(config.output_dtype))
        c2s.append(c2.cpu().to(config.output_dtype))
        ts.append(t.cpu().to(config.output_dtype))
        rs.append(r.cpu().to(config.output_dtype))
        ss.append(s.cpu().to(config.output_dtype))
        lbls.append(lbl.cpu())

    save_file(
        {
            "c0": torch.cat(c0s),
            "c1": torch.cat(c1s),
            "c2": torch.cat(c2s),
            "t": torch.cat(ts),
            "r": torch.cat(rs),
            "s": torch.cat(ss),
            "lbl": torch.cat(lbls),
        },
        config.output_train,
    )

    c0s, c1s, c2s, ts, rs, ss, lbls = [], [], [], [], [], [], []
    for i in range(config.repeats):
        print(f"Splatting {i+1}/{config.repeats}:")
        c0, c1, c2, t, r, s, lbl = splat_dataloader(
            test_loader, device=config.device, **vars(config.splatter)
        )
        c0s.append(c0.cpu().to(config.output_dtype))
        c1s.append(c1.cpu().to(config.output_dtype))
        c2s.append(c2.cpu().to(config.output_dtype))
        ts.append(t.cpu().to(config.output_dtype))
        rs.append(r.cpu().to(config.output_dtype))
        ss.append(s.cpu().to(config.output_dtype))
        lbls.append(lbl.cpu())

    save_file(
        {
            "c0": torch.cat(c0s),
            "c1": torch.cat(c1s),
            "c2": torch.cat(c2s),
            "t": torch.cat(ts),
            "r": torch.cat(rs),
            "s": torch.cat(ss),
            "lbl": torch.cat(lbls),
        },
        config.output_test,
    )
