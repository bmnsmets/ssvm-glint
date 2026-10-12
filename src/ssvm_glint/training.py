import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import wandb
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


def trainingrun(
    model: nn.Module,
    loss_fn: Callable,
    optimizer: Optimizer,
    total_steps: int,
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
    test_every_n_steps: int | None = None,
    scheduler: LRScheduler | None = None,
    metric_fns: dict[str, Callable] | None = None,
    aux_loss_fn: Callable[[], torch.Tensor] | None = None,
    autocast_dtype: torch.dtype | None = None,
    grad_clip_norm: float | None = None,
    checkpoint_path: Path | str | None = None,
    resume: bool = False,
    wandb_run: wandb.Run | None = None,
) -> dict[str, list]:
    """
    Step-based training loop on (x, y) batches with loss_fn(model(x), y).

    If given, aux_loss_fn() is added to the training loss each step (e.g. a
    regularizer on the parameters). It is logged as `aux_loss`; `train_loss`
    and the evaluation use loss_fn only.

    Every test_every_n_steps steps (and at the last step) the loss and each
    metric_fns[name](yp, y) are averaged over test_loader, logged, and a
    checkpoint is written to checkpoint_path. If that checkpoint exists,
    training resumes from it. ReduceLROnPlateau steps on the train loss at
    each step.

    Returns the history of logged values, one entry per evaluation.
    """
    device = next(model.parameters()).device
    eval_fns = {"loss": loss_fn, **(metric_fns or {})}
    scaler = torch.amp.GradScaler(device.type, enabled=autocast_dtype == torch.float16)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    distributed = dist.is_available() and dist.is_initialized()
    rank0 = not distributed or dist.get_rank() == 0

    # Resume
    start_step, history = 0, {}
    if resume and checkpoint_path and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        getattr(model, "module", model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        torch.set_rng_state(ckpt["rng"].cpu())
        if ckpt["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])
        start_step, history = ckpt["step"], ckpt["history"]
        if rank0:
            print(f"Resuming from step {start_step} ({checkpoint_path})")

    batches = infinite_batches(train_loader, start_step)

    pbar = tqdm(
        range(start_step, total_steps),
        initial=start_step,
        total=total_steps,
        disable=not rank0,
    )
    for step in pbar:
        x, y = next(batches)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        model.train()
        optimizer.zero_grad()
        with torch.autocast(
            device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
        ):
            task_loss = loss_fn(model(x), y)
            aux_loss = aux_loss_fn() if aux_loss_fn is not None else None
            loss = task_loss if aux_loss is None else task_loss + aux_loss
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        train_loss = task_loss.detach()
        if distributed:
            train_loss = train_loss.clone()
            dist.all_reduce(train_loss)
            train_loss /= dist.get_world_size()
        train_loss = train_loss.item()

        pbar.set_postfix(loss=f"{train_loss:.4f}", refresh=False)

        if wandb_run is not None:
            log = {"train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
            if aux_loss is not None:
                log["aux_loss"] = aux_loss.item()
            wandb_run.log(log, step=step)

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(train_loss)
            else:
                scheduler.step()

        if test_every_n_steps is None or (
            (step + 1) % test_every_n_steps and step + 1 != total_steps
        ):
            continue

        # Evaluate, log, checkpoint
        assert test_loader is not None
        with torch.no_grad():
            model.eval()
            sums = {k: torch.zeros((), device=device) for k in eval_fns}
            n = 0
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(
                    device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_dtype is not None,
                ):
                    yp = getattr(model, "module", model)(x)
                for k, fn in eval_fns.items():
                    sums[k] += fn(yp.float(), y).float() * x.size(0)
                n += x.size(0)
            totals = torch.stack(
                [*sums.values(), torch.tensor(n, device=device, dtype=torch.float32)]
            )
            if distributed:
                dist.all_reduce(
                    totals
                )  # SUM across ranks, one collective for all metrics
            test = {k: (v / totals[-1]).item() for k, v in zip(sums, totals[:-1])}

        entry = {
            "step": step + 1,
            **{f"test_{k}": v for k, v in test.items()},
        }

        for k, v in entry.items():
            history.setdefault(k, []).append(v)
        if wandb_run:
            wandb_run.log(entry, step=step + 1)

        if checkpoint_path and rank0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(f"{checkpoint_path}.tmp")
            torch.save(
                {
                    "model": getattr(model, "module", model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "scaler": scaler.state_dict(),
                    "rng": torch.get_rng_state(),
                    "cuda_rng": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                    "step": step + 1,
                    "history": history,
                },
                tmp,
            )
            os.replace(tmp, checkpoint_path)  # atomic: a crash never corrupts it

        if distributed:
            dist.barrier()

    return history


def infinite_batches(loader: DataLoader, start_step: int = 0) -> Iterator[Any]:
    """Yields batches forever. Each pass re-invokes __iter__, so a
    shuffling sampler reshuffles; DistributedSampler needs set_epoch."""
    if len(loader) == 0:
        raise ValueError("loader is empty")
    epoch, skip = divmod(start_step, len(loader))
    while True:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            if i < skip:
                continue
            yield batch
        skip = 0
        epoch += 1
