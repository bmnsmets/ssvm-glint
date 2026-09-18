import torch
from typing import Callable
from tqdm import tqdm

def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device | str | None = None,
) -> tuple[float, float]:
    model.train()
    model.to(device)
    
    train_cum_loss = 0
    train_correct = 0

    for x, y in tqdm(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        yp = model(x)
        batch_loss = loss_fn(yp, y)
        batch_loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        batch_correct = (yp.argmax(dim=1) == y).sum().item()

        train_cum_loss += batch_loss.item() * batch_size
        train_correct += batch_correct

    n_train_samples = len(train_loader.dataset)
    train_loss = train_cum_loss / n_train_samples
    train_acc = train_correct / n_train_samples

    return train_loss, train_acc


def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: torch.device | str | None = None,
) -> tuple[float, float]:

    model.eval()
    model.to(device)

    test_matrix = torch.zeros(10, 10)
    test_cum_loss = 0
    test_correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            yp = model(x)
            batch_loss = loss_fn(yp, y)

            batch_size = x.size(0)
            batch_correct = (yp.argmax(dim=1) == y).sum().item()

            test_cum_loss += batch_loss.item() * batch_size
            test_correct += batch_correct

            # get the predicted label
            y_argmax = torch.argmax(yp, 1)

            for i in range(len(y_argmax)):
                test_matrix[y_argmax[i]][y[i]] += 1

    n_test_samples = len(test_loader.dataset)
    test_loss = test_cum_loss / n_test_samples
    test_acc = test_correct / n_test_samples

    return test_loss, test_acc, test_matrix.T


def run(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    wandb_run=None,
    device: torch.device | str | None = None,
) -> dict:

    metrics = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "test_matrix": [],
    }

    for epoch in range(epochs):

        train_loss, train_acc = train(
            model, train_loader, optimizer, loss_fn, device=device
        )
        test_loss, test_acc, test_matrix = test(
            model, test_loader, loss_fn, device=device
        )

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss)
            else:
                scheduler.step()

        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["test_loss"].append(test_loss)
        metrics["test_acc"].append(test_acc)
        metrics["test_matrix"].append(test_matrix)

        if wandb_run:
            wandb_run.log(
                {
                    "test_err": (1 - test_acc),
                    "train_loss": train_loss,
                    "train_err": (1 - train_acc),
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        print(
            f"Epoch {epoch+1}/{epochs} - Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.5f}"
        )

    return metrics