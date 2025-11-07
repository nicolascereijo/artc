from pathlib import Path

from .. import errors
from .. import configurations as config


def get_extension(file: Path) -> str:
    extension = file.suffix

    if extension != "":
        return extension

    return ""


def check_audio_corruption(file_path: Path) -> bool:
    try:
        with file_path.open('rb') as file:
            file.read()
        return True
    except (FileNotFoundError, PermissionError, IsADirectoryError, OSError):
        return False


def check_audio_format(*, path: Path, name: str, configuration_path: Path) -> bool:
    if (not check_audio_corruption(path/name)
            or not errors.check_path_accessible(configuration_path.parent)
            or not configuration_path.is_file()
            or not ((path/name).is_file())
            or get_extension(path/name) not in config.read_config(
                "extensions", configuration_path)):
        return False
    return True
