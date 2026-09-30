from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms.v2 import PILToTensor, ToDtype, Compose
from safetensors import safe_open
from ssvm_glint.tokens import AffPlus2Tokens


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
    """

    def __init__(self, tokens: AffPlus2Tokens, labels: Tensor):
        assert tokens.batch_size[0] == labels.shape[0]
        self.tokens = tokens
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, i: int) -> tuple[AffPlus2Tokens, Tensor]:
        return self.tokens[i], self.labels[i]

    def __getitems__(self, idx: list[int]) -> tuple[AffPlus2Tokens, Tensor]:
        idx = torch.as_tensor(idx, device=self.labels.device)
        return self.tokens[idx], self.labels[idx]


def _no_collate(batch):
    return batch


def load_g3splat_dataset(
    path: str | Path,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> AffPlus2TokenDataset:
    """
    Load a g3splat safetensors file as an AffPlus2TokenDataset with tokens of batch size [M,N] and int64 labels [M].
    The group elements are computed in float32 and then cast to `dtype`.
    """
    with safe_open(path, framework="pt", device=str(device or "cpu")) as f:
        params = {k: f.get_tensor(k).float() for k in ("c0", "c1", "c2", "r", "s", "t")}
        labels = f.get_tensor("lbl").long()
    tokens = AffPlus2Tokens.fromg3splatter(**params).to(dtype)
    return AffPlus2TokenDataset(tokens, labels)


def get_g3splat_dataloaders(
    train_path: str | Path,
    test_path: str | Path,
    train_batch_size: int,
    test_batch_size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
):
    """
    DataLoaders yielding (AffPlus2Tokens [B,N], labels [B]) batches.
    """
    trainset = load_g3splat_dataset(train_path, dtype=dtype, device=device)
    testset = load_g3splat_dataset(test_path, dtype=dtype, device=device)

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
        drop_last=True,
        collate_fn=_no_collate,
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
