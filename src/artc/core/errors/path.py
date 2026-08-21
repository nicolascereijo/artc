import os
from pathlib import Path


def check_path_accessible(path: Path) -> bool:
    if path.as_posix() == ".":
        return False

    try:
        # is_dir() and os.access() are stat based checks. Unlike
        # materializing list(path.iterdir()), they don't scale with the
        # directory's contents, which matters since this runs once per
        # file added.
        return path.is_dir() and os.access(path, os.R_OK | os.X_OK)
    except (PermissionError, FileNotFoundError, NotADirectoryError):
        return False


def check_file_readable(*, path: Path, name: str) -> bool:
    if name is None or name == "":
        return False

    try:
        with (path/name).open('r'):
            pass
        return True
    except (PermissionError, FileNotFoundError):
        return False


def validate_path(*, path: Path, name: str) -> bool:
    if check_path_accessible(Path(path)) and check_file_readable(path=Path(path), name=name):
        return True
    return False
