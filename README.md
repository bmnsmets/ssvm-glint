# GLInT: Geometric Learned Invariant Transformer

This repository contains the code and experiments accompanying the _Symmetry Fingerprinting with Learnable Invariants Transformers_ article submission to [SSVM 2027](https://ssvm2027.github.io/).

## Getting started

We assume you use Linux or WSL on Windows.
We use the [`uv`](https://docs.astral.sh/uv/) package and project manager, install via
```console
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then go to the repository's root and do
```console
uv sync
```
to setup the `ssvm-glint` project environment with CUDA 13 support (the default).

For CPU only, CUDA 11 or CUDA 12 support replace the default `cu13` dependency group:
```console
uv sync --no-group cu13 --group {cpu|cu11|cu12}
```
Note that a later plain `uv sync` or `uv run` switches the environment back to CUDA 13.
Either pass the same `--no-group cu13 --group ...` flags to every `uv run`, or skip the automatic sync after the initial `uv sync`:
```console
export UV_NO_SYNC=1
```

## Project layout

```
src/ssvm_glint/        installable package (`uv sync` installs it in editable mode)
    exp.py, polar.py   Lie group maths (exponentials, polar decomposition)
    tokens.py          geometric tokens (AffPlus2Tokens)
    glint.py, vit.py   GLInT and the ViT baseline
    splatting/         higher order Gaussian splatter
    config.py          experiment configuration loading and printing
    data.py            datasets and dataloaders
    training.py        generic training loop
scripts/               command line entry points
configs/               TOML configuration files for the scripts
tests/                 pytest suite, run with `uv run pytest`
```

## Splatting

The higher order Gaussian splatter in `src/ssvm_glint/splatting` is based on [Victor Klomp's Bachelor project](https://github.com/SuperVK/sparse-image-repr-bep). 
You can see a demo in action by running `uv run python -m ssvm_glint.splatting.g3splatter`.
To splat the MNIST dataset run `uv run scripts/splat.py configs/mnist-g3splat16.toml`.
The `mnist-g3splat16.toml` file contains the splatter configuration as used in the paper.

## Training

To train a model run `uv run scripts/train.py configs/glint-mnist.toml` (or `configs/vit-mnist.toml` for the ViT baseline).
