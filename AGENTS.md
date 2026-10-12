# AGENTS.md

Code and experiments for the SSVM 2027 paper *Symmetry Fingerprinting with Learnable Invariants Transformers* (GLInT): transformers on geometric tokens in Aff^+(2), compared against a ViT baseline on (splatted) MNIST.

## Environment

- Use `uv` for everything: `uv run <cmd>` or activate the venv manually.
- PyTorch comes from a dependency group: `cu13` (default), `cu12`, `cu11` or `cpu`. A plain `uv run` syncs back to `cu13`. On a machine with another group, either pass `--no-group cu13 --group <group>` to every `uv run` or set `UV_NO_SYNC=1`.

## Commands

- Tests: `uv run pytest` (CUDA and DDP tests in `tests/test_training.py` skip without GPUs).
- Train: `uv run scripts/train.py scripts/<config>.toml` (`glint-mnist.toml`, `vit-mnist.toml`), optionally with `--resume`. Runs take minutes on a GPU and log to wandb; do not start one unless asked.
- Splat MNIST: `uv run scripts/splat.py scripts/mnist-g3splat16.toml`.

## Layout

```
src/ssvm_glint/
    math.py          Lie group maths (exponentials, polar decomposition)
    tokens.py        geometric tokens (AffPlus2Tokens)
    glint.py         GLInT model
    vit.py           ViT baseline
    splatting/       higher order Gaussian splatter
    config.py        TOML config loading and printing
    data.py          datasets and dataloaders
    training.py      generic training loop (wandb logging, checkpoints, DDP)
scripts/             entry points and their TOML configs
tests/               pytest suite
data/                datasets (gitignored except data/mnist-splat)
logs/                checkpoints (logs/<config name>/<run name>.safetensors) and wandb files (gitignored)
```

## Experiments

- Configs are TOML files in `scripts/`; paths in them are relative to the config file's directory.
- wandb project `ssvm-glint`. Each step logs `train_loss` and `lr`; every `test_every_n_steps` steps logs `test_loss` and `test_error`. Gradient histograms are logged via `watch`, so the local `logs/wandb/*/files/wandb-summary.json` files are large and mostly histograms. Query runs via the wandb API instead (see `.claude/scripts/wandb_runs.py`).

## Conventions

- Annotate tensor shapes in comments, e.g. `# [B,N,D]`, using B batch, N tokens, D features, C channels, H/W height/width.
- Group elements of Aff^+(2) are stored as `[...,2,3]` tensors (the top two rows of the GL(3) matrix); `det(A) > 0` is assumed, not checked.
- Docstrings: concise Google style, with tensor shapes on every tensor argument and return value.
- Keep changes to the geometric maths covered by tests in `tests/`.

## Working principles

- **Ask, don't guess.** If a request has more than one reasonable reading (which group action, which normalization, which tensor layout), state your assumption or ask. Never silently pick a convention for the geometric maths.
- **Simplest thing that works.** Implement only what was asked: no speculative config options, no abstractions with a single user, no error handling for states that can't occur. This is research code; clarity beats generality.
- **Surgical changes.** Touch only what the task needs. Don't reformat or refactor nearby code, and don't change hyperparameters in `scripts/*.toml` unless asked. Point out unrelated problems (dead code, suspect maths) instead of fixing them. Remove only imports and functions that your own change left unused.
- **Verify against a goal.** Turn a task into a check you can run before calling it done: a failing test that reproduces the bug, a shape assertion, or an invariance check under random group elements for changes to `math.py`, `tokens.py` or `glint.py`. For multi-step work, write a short plan with a check for each step.
