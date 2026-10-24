import argparse
import math
import warnings
from pathlib import Path

import torch
import wandb
from safetensors.torch import save_model
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ssvm_glint.config import load_config_file, obj_to_dict, print_config_tree
from ssvm_glint.data import (
    get_cifar10_dataloaders,
    get_fashion_dataloaders,
    get_g3splat_dataloaders,
    get_mnist_dataloaders,
    get_rasterized_dataloaders,
)
from ssvm_glint.glint import GLInTClassifier
from ssvm_glint.training import trainingrun
from ssvm_glint.vit import ViT


def run(config_file: Path, resume: bool, repeat_index: int):
    """One training run, as a fresh call of this script would do."""
    # Load the config from file
    config = load_config_file(config_file)

    # Resolve relative paths
    config.data.root = (config_file.parent / config.data.root).resolve()
    config.logging.path = (config_file.parent / config.logging.path).resolve()
    (config.logging.path / config.logging.name).mkdir(parents=True, exist_ok=True)

    # Select device
    config.training.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Resolve dtype
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    config.training.dtype = dtypes[config.training.dtype]

    # Resolve model class
    model_classes = {"ViT": ViT, "GLInT": GLInTClassifier}
    model_class_name = config.model.class_name
    model_class = model_classes[model_class_name]

    # Print final configuration
    print_config_tree(config)

    # Set seed if defined
    if hasattr(config.training, "seed"):
        torch.manual_seed(config.training.seed + repeat_index)

    # Prepare data
    if config.data.dataset == "mnist":
        train_loader, test_loader = get_mnist_dataloaders(
            root=config.data.root,
            train_batch_size=config.training.batch_size,
            test_batch_size=config.training.test_batch_size,
            dtype=config.training.dtype,
            device=config.training.device,
        )
    elif config.data.dataset == "mnist-splat-16":
        # Optional training augmentation, rotate in degrees and translate in raw splat coordinates.
        augment = None
        if hasattr(config.data, "augment"):
            augment = obj_to_dict(config.data.augment)
            augment["rotate"] = math.radians(augment.get("rotate", 0.0))
        train_loader, test_loader = get_g3splat_dataloaders(
            train_path=config.data.root
            / "mnist-splat"
            / "g3splat-16-train.safetensors",
            test_path=config.data.root / "mnist-splat" / "g3splat-16-test.safetensors",
            train_batch_size=config.training.batch_size,
            test_batch_size=config.training.test_batch_size,
            dtype=config.training.dtype,
            device=config.training.device,
            augment=augment,
        )
    elif config.data.dataset == "mnist-splat-16-rasterized":
        train_loader, test_loader = get_rasterized_dataloaders(
            train_path=config.data.root
            / "mnist-splat-rasterized"
            / "g3splat-16-train-rasterized.safetensors",
            test_path=config.data.root
            / "mnist-splat-rasterized"
            / "g3splat-16-test-rasterized.safetensors",
            train_batch_size=config.training.batch_size,
            test_batch_size=config.training.test_batch_size,
            dtype=config.training.dtype,
            device=config.training.device,
        )
    elif config.data.dataset == "fashionmnist":
        train_loader, test_loader = get_fashion_dataloaders(
            root=config.data.root,
            train_batch_size=config.training.batch_size,
            test_batch_size=config.training.test_batch_size,
            dtype=config.training.dtype,
        )
    elif config.data.dataset == "cifar10":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            train_loader, test_loader = get_cifar10_dataloaders(
                root=config.data.root,
                train_batch_size=config.training.batch_size,
                test_batch_size=config.training.test_batch_size,
                dtype=config.training.dtype,
            )
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")

    # Infer the dimension of the semantic part of the tokens from the data
    if model_class is GLInTClassifier:
        config.model.kwargs.in_dim = train_loader.dataset.tokens.u.shape[-1]

    # Instantiate model
    model = model_class(**obj_to_dict(config.model.kwargs))
    model.to(config.training.device)

    # Log some info to the config that gets passed to WandB
    config.num_parameters = sum(p.numel() for p in model.parameters())
    config.model_summary = str(model)

    wandb_run = (
        wandb.init(
            project=config.logging.project,
            tags=config.logging.tags,
            config=obj_to_dict(config),
            dir=config.logging.path,
            save_code=True,
        )
        if config.logging.wandb
        else None
    )

    if wandb_run:
        wandb_run.watch(model)

    # No weight decay on LayerNorm parameters and, for GLInT, on the noise scales
    # and CLS token
    no_decay = {
        id(p)
        for m in model.modules()
        if isinstance(m, torch.nn.LayerNorm)
        for p in m.parameters()
    }
    if model_class is GLInTClassifier:
        no_decay |= {
            id(p)
            for n, p in model.named_parameters()
            if n.split(".")[-1] in ("log_sigmas", "cls_token", "alpha", "beta")
        }
    params = list(model.parameters())
    param_groups = [
        {"params": [p for p in params if id(p) not in no_decay]},
        {"params": [p for p in params if id(p) in no_decay], "weight_decay": 0.0},
    ]

    opt = torch.optim.AdamW(
        param_groups,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    sched = SequentialLR(
        opt,
        schedulers=[
            LinearLR(
                opt,
                start_factor=config.training.warmup_factor,
                total_iters=config.training.warmup_steps,
            ),
            CosineAnnealingLR(
                opt,
                T_max=config.training.total_steps - config.training.warmup_steps,
                eta_min=config.training.min_lr,
            ),
        ],
        milestones=[config.training.warmup_steps],
    )

    metric_fns = {"error": lambda yp, y: (yp.argmax(dim=1) != y).float().mean()}

    history = trainingrun(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=opt,
        total_steps=config.training.total_steps,
        train_loader=train_loader,
        test_loader=test_loader,
        test_every_n_steps=config.training.test_every_n_steps,
        scheduler=sched,
        metric_fns=metric_fns,  # ty: ignore[invalid-argument-type]
        aux_loss_fn=model.aux_loss if model_class is GLInTClassifier else None,  # ty: ignore[invalid-argument-type]
        log_fn=model.log_sigmas if model_class is GLInTClassifier else None,  # ty: ignore[invalid-argument-type]
        autocast_dtype=config.training.dtype,
        checkpoint_path=config.logging.path / "last.pt",
        resume=resume,
        wandb_run=wandb_run,
    )

    if wandb_run:
        wandb_run.finish()

    model_path = (
        config.logging.path
        / config.logging.name
        / f"{wandb_run.name if wandb_run else 'last'}.safetensors"
    )
    print(f"Saving model to: {model_path}")
    save_model(model, model_path)
    print("Done.")


if __name__ == "__main__" and not hasattr(__builtins__, "__IPYTHON__"):
    parser = argparse.ArgumentParser(description="Image Classifier Training")
    parser.add_argument("config_file", type=Path, help="Configuration file path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Number of independent runs"
    )
    args = parser.parse_args()

    for i in range(args.repeat):
        run(args.config_file, resume=args.resume and i == 0, repeat_index=i)
