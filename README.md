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

## Splatting

The higher order Gaussian splatter in `./splatting` is based on [Victor Klomp's Bachelor project](https://github.com/SuperVK/sparse-image-repr-bep). 
You can see a demo in action by running `./splatting/g3splatter.py`.
To splat the MNIST dataset run `./splatting/main.py ./splatting/mnist-g3splat16.toml`.
The `mnist-g3splat16.toml` file contains the splatter configuration as used in the paper.