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
uv sync --extra {cpu|cu11|cu12|cu13}
```
to setup the `ssvm-glint` project environment with CPU, CUDA 11, CUDA 12 or CUDA 13 support.

## Project layout

```
src/ssvm_glint/        installable package (`uv sync` installs it in editable mode)
    geometry/          Lie group maths and geometric tokens, shared by everything else
    splatting/         higher order Gaussian splatter
    models/            GLInT and the ViT baseline
    config.py          experiment configuration loading and printing
    data.py            datasets and dataloaders
    training.py        generic training loop
scripts/               command line entry points and their configuration files
tests/                 pytest suite, run with `uv run pytest`
```
Since the package is installed, `from ssvm_glint.geometry import AffPlus2Tokens` etc. works from any directory, notebook or interactive session.

## Splatting

The higher order Gaussian splatter in `src/ssvm_glint/splatting` is based on [Victor Klomp's Bachelor project](https://github.com/SuperVK/sparse-image-repr-bep). 
You can see a demo in action by running `uv run python -m ssvm_glint.splatting.g3splatter`.
To splat the MNIST dataset run `uv run scripts/splat.py scripts/mnist-g3splat16.toml`.
The `mnist-g3splat16.toml` file contains the splatter configuration as used in the paper.

## Training

To train a model run `uv run scripts/train.py scripts/glint-mnist.toml` (or `scripts/vit-mnist.toml` for the ViT baseline).
