from pathlib import Path


def check_path_accessible(path: Path) -> bool:
    if path.as_posix() == ".":
        return False

    try:
        _ = list(path.iterdir())
        return path.exists()
    except (PermissionError, FileNotFoundError):
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
