"""
Tests for the g3splat / AffPlus2Tokens part of ssvm_glint.data.

Uses small synthetic data, so the tests do not depend on the data files.
CUDA tests are skipped when no GPU is available.
"""

import pytest
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from ssvm_glint.data import (
    AffPlus2TokenDataset,
    _no_collate,
    get_g3splat_dataloaders,
    load_g3splat_dataset,
)
from ssvm_glint.tokens import AffPlus2Tokens

M, N, C = 20, 4, 1


def make_dataset(m=M, device="cpu"):
    """Dataset where labels[i] = i and u[i] = i, so pairing can be checked after shuffling."""
    g = torch.randn(m, N, 2, 3, device=device)
    u = torch.arange(m, dtype=torch.float32, device=device)[:, None, None].expand(m, N, 3)
    labels = torch.arange(m, device=device)
    return AffPlus2TokenDataset(AffPlus2Tokens(g=g, u=u), labels)


def make_g3splat_params(m=M, seed=0):
    """Synthetic g3splat file contents, stored as float16 like the real files."""
    gen = torch.Generator().manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=gen).half()

    return dict(
        c0=rn(m, N, C),
        c1=rn(m, N, C),
        c2=rn(m, N, C),
        r=rn(m, N),
        s=rn(m, N, 3),
        t=rn(m, N, 2),
        lbl=torch.randint(0, 10, (m,), generator=gen),
    )


def write_g3splat(path, **kwargs):
    params = make_g3splat_params(**kwargs)
    save_file(params, str(path))
    return params


# ---------------------------------------------------------------------------
# AffPlus2TokenDataset
# ---------------------------------------------------------------------------


def test_len():
    assert len(make_dataset()) == M


def test_getitem():
    ds = make_dataset()
    tok, lbl = ds[3]
    assert isinstance(tok, AffPlus2Tokens)
    assert tok.batch_size == (N,)
    assert lbl.item() == 3
    torch.testing.assert_close(tok.g, ds.tokens.g[3])


def test_getitems():
    ds = make_dataset()
    tok, lbl = ds.__getitems__([5, 0, 7])
    assert tok.batch_size == (3, N)
    assert lbl.tolist() == [5, 0, 7]
    torch.testing.assert_close(tok.g, ds.tokens.g[[5, 0, 7]])


def test_mismatched_lengths():
    ds = make_dataset()
    with pytest.raises(AssertionError):
        AffPlus2TokenDataset(ds.tokens, ds.labels[:-1])


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("drop_last, sizes", [(True, [8, 8]), (False, [8, 8, 4])])
def test_loader_batch_sizes(drop_last, sizes):
    loader = DataLoader(
        make_dataset(), batch_size=8, drop_last=drop_last, collate_fn=_no_collate
    )
    batches = list(loader)
    assert [tok.batch_size[0] for tok, _ in batches] == sizes
    for tok, lbl in batches:
        assert isinstance(tok, AffPlus2Tokens)
        assert tok.batch_size[1:] == (N,)
        assert lbl.shape == tok.batch_size[:1]


def test_loader_no_shuffle_preserves_order():
    loader = DataLoader(make_dataset(), batch_size=8, collate_fn=_no_collate)
    labels = torch.cat([lbl for _, lbl in loader])
    assert labels.tolist() == list(range(M))


def test_loader_shuffle_is_permutation_and_keeps_pairs():
    torch.manual_seed(0)
    loader = DataLoader(
        make_dataset(), batch_size=8, shuffle=True, collate_fn=_no_collate
    )
    labels = []
    for tok, lbl in loader:
        # u was built so that u[i] == i == labels[i]
        assert (tok.u[:, 0, 0].long() == lbl).all()
        labels.append(lbl)
    labels = torch.cat(labels)
    assert labels.tolist() != list(range(M))
    assert sorted(labels.tolist()) == list(range(M))


def test_batch_works_with_training_loop():
    """ssvm_glint.training.trainingrun unpacks (x, y) and calls x.to(device, non_blocking=True) and x.size(0)."""
    loader = DataLoader(make_dataset(), batch_size=8, collate_fn=_no_collate)
    x, y = next(iter(loader))
    x, y = x.to("cpu", non_blocking=True), y.to("cpu", non_blocking=True)
    assert isinstance(x, AffPlus2Tokens)
    assert x.size(0) == y.shape[0] == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_loader_on_cuda():
    loader = DataLoader(
        make_dataset(device="cuda"), batch_size=8, shuffle=True, collate_fn=_no_collate
    )
    tok, lbl = next(iter(loader))
    assert tok.g.device.type == tok.u.device.type == lbl.device.type == "cuda"


# ---------------------------------------------------------------------------
# loading g3splat files
# ---------------------------------------------------------------------------


def test_load_g3splat_dataset(tmp_path):
    path = tmp_path / "splats.safetensors"
    params = write_g3splat(path)
    ds = load_g3splat_dataset(path)

    assert len(ds) == M
    assert ds.tokens.batch_size == (M, N)
    assert ds.tokens.u.shape == (M, N, 3 * C)
    assert ds.labels.dtype == torch.int64
    assert ds.labels.tolist() == params["lbl"].tolist()

    expected = AffPlus2Tokens.fromg3splatter(
        **{k: v.float() for k, v in params.items() if k != "lbl"}
    )
    torch.testing.assert_close(ds.tokens.g, expected.g)
    torch.testing.assert_close(ds.tokens.u, expected.u)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_load_g3splat_dataset_dtype(tmp_path, dtype):
    path = tmp_path / "splats.safetensors"
    params = write_g3splat(path)
    ds = load_g3splat_dataset(path, dtype=dtype)
    assert ds.tokens.g.dtype == ds.tokens.u.dtype == dtype
    assert ds.labels.dtype == torch.int64
    # group elements are computed in float32 and only then cast
    expected = AffPlus2Tokens.fromg3splatter(
        **{k: v.float() for k, v in params.items() if k != "lbl"}
    )
    torch.testing.assert_close(ds.tokens.g, expected.g.to(dtype), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_g3splat_dataset_cuda(tmp_path):
    path = tmp_path / "splats.safetensors"
    write_g3splat(path)
    ds = load_g3splat_dataset(path, device="cuda")
    assert ds.tokens.g.device.type == ds.labels.device.type == "cuda"


def test_get_g3splat_dataloaders(tmp_path):
    train_path, test_path = tmp_path / "train.safetensors", tmp_path / "test.safetensors"
    write_g3splat(train_path, m=20, seed=0)
    write_g3splat(test_path, m=12, seed=1)
    train_loader, test_loader = get_g3splat_dataloaders(
        train_path, test_path, train_batch_size=8, test_batch_size=5
    )

    # drop_last=True for the train loader only, so the test loader keeps its partial batch
    assert len(train_loader) == 2
    assert len(test_loader) == 3
    for loader, bs in ((train_loader, 8), (test_loader, 5)):
        for tok, lbl in loader:
            assert isinstance(tok, AffPlus2Tokens)
            assert tok.batch_size[1] == N
            assert tok.batch_size[0] <= bs
            assert lbl.shape == (tok.batch_size[0],)
            assert lbl.dtype == torch.int64


def test_get_g3splat_dataloaders_normalization(tmp_path):
    """u, t and log(a) are standardized over the training set, and the test set uses the same transform."""
    train_path, test_path = tmp_path / "train.safetensors", tmp_path / "test.safetensors"
    write_g3splat(train_path, m=32, seed=0)
    write_g3splat(test_path, m=16, seed=1)
    train_loader, test_loader = get_g3splat_dataloaders(
        train_path, test_path, train_batch_size=8, test_batch_size=8
    )

    # the statistics are recomputed from the rebuilt group elements, so they land close to the
    # targets rather than exactly on them
    tol = dict(atol=1e-3, rtol=0)
    tok = train_loader.dataset.tokens
    zero, one = torch.zeros(()), torch.ones(())
    torch.testing.assert_close(tok.u.mean(), zero, **tol)
    torch.testing.assert_close(tok.u.std(), one, **tol)
    torch.testing.assert_close(tok.t.mean(dim=(0, 1)), torch.zeros(2), **tol)
    torch.testing.assert_close(tok.t.std(), one, **tol)
    # one scalar scale over both components jointly, so each component is only roughly 1
    torch.testing.assert_close(tok.t.std(dim=(0, 1)), torch.ones(2), atol=0.1, rtol=0)
    # log(a) is only centered, so it keeps the raw spread of the training set
    raw = load_g3splat_dataset(train_path).tokens
    torch.testing.assert_close(tok.a.log().mean(), zero, **tol)
    torch.testing.assert_close(tok.a.log().std(), raw.a.log().std(), **tol)

    # the test set is mapped through the training transform, so it is *not* standardized itself
    test_tok = test_loader.dataset.tokens
    assert abs(test_tok.u.mean().item()) > 1e-2
    assert abs(test_tok.t.std().item() - 1.0) > 1e-2

    # R and S are untouched by the normalization
    torch.testing.assert_close(tok.R, raw.R, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tok.S, raw.S, atol=1e-3, rtol=1e-3)
