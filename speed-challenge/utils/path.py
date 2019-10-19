from pathlib import Path
from shutil import rmtree

from typing import Union


def to_path(path: Union[str, Path]) -> Path:
    """Convert str to Path."""

    if not isinstance(path, Path):
        path = Path(path)

    return path


def create(path: Union[str, Path]):
    """Create folder if it does not exists."""

    path = to_path(path)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def remove(path: Union[str, Path]):
    """Remove the given folder recursively."""

    path = to_path(path)

    if not path.exists():
        return

    if path.is_dir():
        rmtree(path)
    else:
        path.unlink()


def clear(path: Union[str, Path], pattern: str = '*'):
    """Remove all folders and files in the directory that match the pattern."""

    path = to_path(path)

    if not path.is_dir():
        raise ValueError(f'{str(path)} is not a directory!')

    for it in path.glob(pattern):
        remove(it)
