"""
Tests for ssvm_glint.training.


DDP tests use 2 CPU ranks with the gloo backend; CUDA tests are skipped
when no GPU is available.
"""

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset, TensorDataset

from ssvm_glint.training import infinite_batches, trainingrun


def accuracy(yp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (yp.argmax(-1) == y).float().mean()


def make_data(n: int, seed: int) -> TensorDataset:
    """Linearly separable 4-class problem; the labelling is shared across seeds."""
    w = torch.randn(16, 4, generator=torch.Generator().manual_seed(42))
    x = torch.randn(n, 16, generator=torch.Generator().manual_seed(seed))
    return TensorDataset(x, (x @ w).argmax(-1))


def make_model(dropout: float = 0.0) -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(16, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 4)
    )


def params_vec(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])


def run_single(
    device="cpu",
    total_steps=100,
    checkpoint_path=None,
    autocast_dtype=None,
    grad_clip_norm=None,
    shuffle=False,
    test=True,
    test_every_n_steps=25,
    plateau=True,
):
    """Train the toy model in a single process; returns (model, history, optimizer)."""
    train, testset = make_data(1000, 0), make_data(300, 1)
    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
        if plateau
        else torch.optim.lr_scheduler.StepLR(opt, 30, 0.5)
    )
    history = trainingrun(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=opt,
        total_steps=total_steps,
        train_loader=DataLoader(train, batch_size=50, shuffle=shuffle),
        test_loader=DataLoader(testset, batch_size=100) if test else None,
        test_every_n_steps=test_every_n_steps if test else None,
        scheduler=sched,
        metric_fns={"acc": accuracy},
        autocast_dtype=autocast_dtype,
        grad_clip_norm=grad_clip_norm,
        checkpoint_path=checkpoint_path,
    )
    return model, history, opt


# Single process


def test_learns_and_evaluates(tmp_path):
    _, history, _ = run_single(checkpoint_path=tmp_path / "ckpt.pt")
    assert history["test_acc"][-1] > 0.8
    assert history["step"] == [25, 50, 75, 100]


def test_checkpoint_written_atomically(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    run_single(checkpoint_path=ckpt)
    assert ckpt.exists()
    assert not Path(f"{ckpt}.tmp").exists()


def test_resume_matches_uninterrupted(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    ref, ref_history, _ = run_single()
    run_single(total_steps=50, checkpoint_path=ckpt)
    res, res_history, _ = run_single(checkpoint_path=ckpt)
    assert torch.equal(params_vec(ref), params_vec(res))
    assert res_history["step"] == [25, 50, 75, 100]
    assert res_history["test_loss"][-1] == pytest.approx(
        ref_history["test_loss"][-1], abs=1e-6
    )


def test_resume_restores_scheduler(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    _, _, ref_opt = run_single(plateau=False)
    run_single(total_steps=50, checkpoint_path=ckpt, plateau=False)
    _, _, res_opt = run_single(checkpoint_path=ckpt, plateau=False)
    assert ref_opt.param_groups[0]["lr"] == res_opt.param_groups[0]["lr"]


def test_final_step_evaluated():
    _, history, _ = run_single(total_steps=60, test_every_n_steps=25)
    assert history["step"] == [25, 50, 60]


def test_without_test_loader():
    _, history, _ = run_single(total_steps=30, test=False)
    assert history == {}


def test_aux_loss_fn_is_minimized():
    def run(aux):
        model = make_model()
        layer = model[3]
        assert isinstance(layer, nn.Linear)
        bias = layer.bias
        trainingrun(
            model,
            nn.CrossEntropyLoss(),
            torch.optim.AdamW(model.parameters(), lr=1e-2),
            total_steps=100,
            train_loader=DataLoader(make_data(1000, 0), batch_size=50),
            aux_loss_fn=(lambda: 10 * (bias - 0.5).pow(2).sum()) if aux else None,
        )
        return (bias.detach() - 0.5).abs().max()

    assert run(aux=True) < 0.1 < run(aux=False)


def test_test_every_without_test_loader_rejected():
    model = make_model()
    with pytest.raises(AssertionError):
        trainingrun(
            model,
            nn.CrossEntropyLoss(),
            torch.optim.AdamW(model.parameters()),
            total_steps=30,
            train_loader=DataLoader(make_data(1000, 0), batch_size=50),
            test_loader=None,
            test_every_n_steps=10,
        )


# infinite_batches


def take(it, n: int) -> list[list[int]]:
    """First n batches of an index dataset, as lists of indices."""
    return [batch[0].tolist() for _, batch in zip(range(n), it)]


def index_loader(n: int = 10, batch_size: int = 3, **kwargs) -> DataLoader:
    return DataLoader(TensorDataset(torch.arange(n)), batch_size=batch_size, **kwargs)


def test_infinite_batches_cycles():
    # 10 samples, batch size 3 -> 4 batches per epoch, last one partial
    epoch = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert take(infinite_batches(index_loader()), 10) == (epoch * 3)[:10]


@pytest.mark.parametrize("start_step", [0, 1, 4, 6, 9])
def test_infinite_batches_start_step(start_step):
    loader = index_loader()
    full = take(infinite_batches(loader), start_step + 5)
    assert take(infinite_batches(loader, start_step), 5) == full[start_step:]


def test_infinite_batches_reshuffles_each_epoch():
    loader = index_loader(n=12, batch_size=4, shuffle=True)
    batches = take(infinite_batches(loader), 9)
    epochs = [[x for b in batches[i : i + 3] for x in b] for i in range(0, 9, 3)]
    for order in epochs:
        assert sorted(order) == list(range(12))
    assert len({tuple(order) for order in epochs}) > 1


@pytest.mark.parametrize("start_step", [0, 2, 3, 7])
def test_infinite_batches_distributed_sampler_resume(start_step):
    # set_epoch makes DistributedSampler deterministic per epoch, so resuming
    # mid-epoch reproduces exactly the batches of the uninterrupted run
    def loader():
        dataset = TensorDataset(torch.arange(12))
        sampler = DistributedSampler(dataset, num_replicas=2, rank=0, seed=0)
        return DataLoader(dataset, batch_size=2, sampler=sampler)

    full = take(infinite_batches(loader()), start_step + 6)
    assert take(infinite_batches(loader(), start_step), 6) == full[start_step:]
    # each epoch (3 batches of rank 0's 6 samples) uses a different permutation
    assert full[:3] != full[3:6]


def test_infinite_batches_empty_loader():
    with pytest.raises(ValueError):
        next(infinite_batches(index_loader(n=0)))


# CUDA


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "autocast_dtype",
    [None, torch.bfloat16, torch.float16],
    ids=["fp32", "bf16", "fp16"],
)
def test_cuda(tmp_path, autocast_dtype):
    kwargs = {
        "device": "cuda",
        "checkpoint_path": tmp_path / "ckpt.pt",
        "autocast_dtype": autocast_dtype,
        "grad_clip_norm": 1.0,
        "shuffle": True,
    }
    _, history, _ = run_single(**kwargs)
    assert history["test_acc"][-1] > 0.8
    _, history, _ = run_single(total_steps=150, **kwargs)
    assert history["step"] == [25, 50, 75, 100, 125, 150]


# DDP


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ddp_worker(rank, world_size, port, ckpt_dir, total_steps, out):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(1234 + rank)  # differing per-rank RNG, as in real launches

    # 301 test samples: uneven shards across ranks
    train, testset = make_data(1000, 0), make_data(301, 1)
    model = DDP(make_model(dropout=0.1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    history = trainingrun(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=opt,
        total_steps=total_steps,
        train_loader=DataLoader(
            train, batch_size=25, sampler=DistributedSampler(train, seed=0)
        ),
        test_loader=DataLoader(
            Subset(testset, range(rank, len(testset), world_size)), batch_size=50
        ),
        test_every_n_steps=20,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=20, factor=0.5
        ),
        metric_fns={"acc": accuracy},
        checkpoint_path=Path(ckpt_dir) / "ckpt.pt",
    )
    out[rank] = {
        "params": params_vec(model.module),
        "lr": opt.param_groups[0]["lr"],
        "history": history,
    }
    dist.destroy_process_group()


def run_ddp(world_size, ckpt_dir, total_steps) -> dict:
    with mp.get_context("spawn").Manager() as manager:
        out = manager.dict()
        mp.start_processes(
            _ddp_worker,
            args=(world_size, _free_port(), ckpt_dir, total_steps, out),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )
        return dict(out)


@pytest.fixture(scope="module")
def ddp_runs(tmp_path_factory):
    """Two-rank gloo run to 100 steps, then resumed to 160, shared by all DDP tests."""
    ckpt_dir = tmp_path_factory.mktemp("ddp")
    out = run_ddp(2, ckpt_dir, 100)
    ckpt_state = torch.load(ckpt_dir / "ckpt.pt", weights_only=False)
    out_resumed = run_ddp(2, ckpt_dir, 160)
    return out, ckpt_state, out_resumed


ddp = pytest.mark.skipif(
    not dist.is_available(), reason="torch.distributed not available"
)


@ddp
def test_ddp_all_ranks_return(ddp_runs):
    out, _, _ = ddp_runs
    assert sorted(out) == [0, 1]


@ddp
def test_ddp_ranks_in_sync(ddp_runs):
    out, _, _ = ddp_runs
    assert torch.equal(out[0]["params"], out[1]["params"])
    assert out[0]["lr"] == out[1]["lr"]
    assert out[0]["history"] == out[1]["history"]


@ddp
def test_ddp_learns(ddp_runs):
    out, _, _ = ddp_runs
    assert out[0]["history"]["test_acc"][-1] > 0.8


@ddp
def test_ddp_sharded_metrics_match_full_test_set(ddp_runs):
    out, ckpt_state, _ = ddp_runs
    model = make_model()
    model.load_state_dict(ckpt_state["model"])
    model.eval()
    x, y = make_data(301, 1).tensors
    with torch.no_grad():
        yp = model(x)
    history = out[0]["history"]
    assert history["test_loss"][-1] == pytest.approx(
        nn.CrossEntropyLoss()(yp, y).item(), abs=1e-5
    )
    assert history["test_acc"][-1] == pytest.approx(accuracy(yp, y).item(), abs=1e-6)


@ddp
def test_ddp_checkpoint_unwrapped(ddp_runs):
    _, ckpt_state, _ = ddp_runs
    assert not any(k.startswith("module.") for k in ckpt_state["model"])


@ddp
def test_ddp_resume(ddp_runs):
    _, _, out_resumed = ddp_runs
    assert torch.equal(out_resumed[0]["params"], out_resumed[1]["params"])
    assert out_resumed[0]["history"]["step"] == list(range(20, 161, 20))
