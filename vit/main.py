import argparse
from pathlib import Path
import warnings
import torch
import torchvision
from torchvision.transforms.v2 import PILToTensor, ToDtype, Compose
import wandb

from vit import ViT, ViT_SASP

import sys

sys.path.append("..")
from utils import load_config_file, print_config_tree, obj_to_dict
from training import run


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


def get_mnist_splat_dataloaders(
    train_path: str | Path,
    test_path: str | Path,
    train_batch_size: int,
    test_batch_size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
):
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    x_train, y_train = train_data["rasterized28x28"], train_data["lbl"]
    x_train, y_train = x_train.to(dtype).to(device), y_train.to(device)
    x_test, y_test = test_data["rasterized28x28"], test_data["lbl"]
    x_test, y_test = x_test.to(dtype).to(device), y_test.to(device)
    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    testset = torch.utils.data.TensorDataset(x_test, y_test)

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


if __name__ == "__main__" and not hasattr(__builtins__, "__IPYTHON__"):
    parser = argparse.ArgumentParser(description="Image Classifier Training")
    parser.add_argument("config_file", type=Path, help="Configuration file path")
    args = parser.parse_args()

    # Load the config from file
    config = load_config_file(args.config_file)

    # Resolve relative paths
    config.data.root = (args.config_file.parent / config.data.root).resolve()
    config.logging.path = (args.config_file.parent / config.logging.path).resolve()
    (config.logging.path / config.logging.name).mkdir(parents=True, exist_ok=True)

    # Select device
    config.training.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Resolve dtype
    dtypes = {"float32": torch.float32, "float16": torch.float16}
    config.training.dtype = dtypes[config.training.dtype]

    # Resolve model class
    model_classes = {"ViT": ViT, "ViT_SASP": ViT_SASP}
    model_class_name = config.architecture.class_name
    model_class = model_classes[model_class_name]

    # Print final configuration
    print_config_tree(config)

    # Set seed if defined
    if hasattr(config.training, "seed"):
        torch.manual_seed(config.training.seed)

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
        train_loader, test_loader = get_mnist_splat_dataloaders(
            train_path=config.data.root / "mnist-splat" / "g3splat-16-train.pl",
            test_path=config.data.root / "mnist-splat" / "g3splat-16-test.pl",
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

    # Instantiate model
    model = model_class(**obj_to_dict(config.model))
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
        )
        if config.logging.wandb
        else None
    )

    if wandb_run:
        wandb_run.watch(model)

    opt = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", **vars(config.training.scheduler)
    )

    metrics = run(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        epochs=config.training.epochs,
        optimizer=opt,
        scheduler=sched,
        wandb_run=wandb_run,
        device=config.training.device,
    )

    if wandb_run:
        wandb_run.finish()

    model_path = (
        config.logging.path
        / config.logging.name
        / f"{wandb_run.name if wandb_run else "last"}.pth"
    )
    print(f"Saving model to: {model_path}")
    torch.save(model, model_path)
    print("Done.")
