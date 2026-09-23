from pathlib import Path
import torch
import torchvision
from torchvision.transforms.v2 import PILToTensor, ToDtype, Compose
from safetensors import safe_open


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


def get_g3splat_dataloaders(
    train_path: str | Path,
    test_path: str | Path,
    train_batch_size: int,
    test_batch_size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
):
    with safe_open(train_path, framework="pt", device=device) as f:
        c0 = f.get_tensor("c0").to(dtype)
        c1 = f.get_tensor("c1").to(dtype)
        c2 = f.get_tensor("c2").to(dtype)
        r = f.get_tensor("r").to(dtype)
        s = f.get_tensor("s").to(dtype)
        t = f.get_tensor("t").to(dtype)
        lbl = f.get_tensor("lbl").to(dtype)
        trainset = torch.utils.data.TensorDataset(c0, c1, c2, r, s, t, lbl)

    with safe_open(test_path, framework="pt", device=device) as f:
        c0 = f.get_tensor("c0").to(dtype)
        c1 = f.get_tensor("c1").to(dtype)
        c2 = f.get_tensor("c2").to(dtype)
        r = f.get_tensor("r").to(dtype)
        s = f.get_tensor("s").to(dtype)
        t = f.get_tensor("t").to(dtype)
        lbl = f.get_tensor("lbl").to(dtype)
        testset = torch.utils.data.TensorDataset(c0, c1, c2, r, s, t, lbl)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=True,
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
