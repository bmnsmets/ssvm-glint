from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple


def git_root() -> Path:
    """
    Returns the root directory of the git repo this file is in.
    """
    import subprocess

    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
    ).stdout.strip()

    if out and Path(out).exists():
        return Path(out)

    for p in Path(__file__).parents:
        if (p / ".git").exists():
            return p

    raise RuntimeError("Could not determine git repo root. Are we in a git repo?")


def load_config_file(path: Path | str) -> SimpleNamespace:
    """
    Load TOML config file with every dictionary converted to a SimpleNamespace.
    """
    import tomllib
    from types import SimpleNamespace

    def parse(d):
        x = SimpleNamespace()
        for k, v in d.items():
            setattr(x, k, parse(v)) if isinstance(v, dict) else setattr(x, k, v)
        return x

    with open(path, "rb") as fp:
        cfg = tomllib.load(fp)

    return parse(cfg)


def flatten_config(config: SimpleNamespace) -> List[Tuple[str, Any]]:
    """
    Flatten a recursive SimpleNamespace using a period to separate the naming hierarchy.

    :param config: Description
    :type config: SimpleNamespace
    :return: Description
    :rtype: List[Tuple[str, Any]]
    """
    from types import SimpleNamespace

    lst = []
    for k, v in config.__dict__.items():
        if isinstance(v, SimpleNamespace):
            lst += [(k + "." + x, y) for (x, y) in flatten_config(v)]
        else:
            lst.append((k, v))
    return lst


def obj_to_dict(obj: Any) -> Any:
    if type(obj) is dict:
        res = {}
        for k, v in obj.items():
            res[k] = obj_to_dict(v)
        return res
    elif type(obj) is list:
        return [obj_to_dict(item) for item in obj]
    elif type(obj) is SimpleNamespace:
        return obj_to_dict(vars(obj))
    else:
        return obj


def print_config_table(config: SimpleNamespace) -> None:
    """
    Print out the configuration in a table.

    :param config: the configuration (i.e. recursive SimpleNamespace) to print
    :type config: SimpleNamespace
    """
    from rich import box
    from rich.console import Console
    from rich.table import Table

    table = Table(row_styles=["dim", ""], box=box.ROUNDED)
    table.add_column("Configuration", justify="left")
    table.add_column(justify="left")

    for k, v in flatten_config(config):
        table.add_row(k, str(v))

    console = Console()
    console.print(table)


def print_config_tree(config: SimpleNamespace) -> None:
    """
    Print out the configuration in a tree.

    :param config: the configuration (i.e. recursive SimpleNamespace) to print
    :type config: SimpleNamespace
    """
    import rich
    from rich.tree import Tree

    def _walk(cfg, tree):
        for k, v in cfg.__dict__.items():
            if isinstance(v, SimpleNamespace):
                subtree = tree.add(f"[b]{k}")
                _walk(v, subtree)
            else:
                tree.add(f"{k} = {v} [dim]:{type(v).__name__}")

    tree = Tree("[b royal_blue1]Configuration")
    _walk(config, tree)
    rich.print(tree)
