import os
from pathlib import Path


def check_path_accessible(path: Path) -> bool:
    """Checks whether 'path' is a directory this process can read and enter.

    Args:
        path: Directory to check.

    Returns:
        'True' if 'path' is a readable, listable directory.
    """
    if path.as_posix() == ".":
        return False

    try:
        # 'is_dir' and 'os.access' are stat based checks. Unlike materializing
        # 'list(path.iterdir())', they don't scale with the directory's
        # contents, which matters since this runs once per file added.
        return path.is_dir() and os.access(path, os.R_OK | os.X_OK)
    except (PermissionError, FileNotFoundError, NotADirectoryError):
        return False


def check_file_readable(*, path: Path, name: str) -> bool:
    """Checks whether 'name' names a readable file inside 'path'.

    Args:
        path: Directory expected to contain the file.
        name: File name, including its extension.

    Returns:
        'True' if 'path / name' can be opened for reading.
    """
    if name == "":
        return False

    try:
        with (path / name).open("r"):
            pass
        return True
    except (
        PermissionError, FileNotFoundError, IsADirectoryError, OSError,
    ):
        return False


def validate_path(*, path: Path, name: str) -> bool:
    """Checks whether 'name' names a readable file inside a readable 'path'.

    Args:
        path: Directory expected to contain the file.
        name: File name, including its extension.

    Returns:
        'True' if both 'check_path_accessible' and 'check_file_readable' pass.
    """
    return check_path_accessible(path) and check_file_readable(
        path=path, name=name
    )
