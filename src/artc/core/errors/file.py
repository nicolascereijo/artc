from pathlib import Path

import artc.core.configurations as config

from .path import check_path_accessible


def get_extension(file: Path) -> str:
    """Returns 'file's extension, including the leading dot.

    Args:
        file: File whose extension to read.

    Returns:
        'file.suffix', or '""' if 'file' has no extension.
    """
    return file.suffix


def check_audio_corruption(file_path: Path) -> bool:
    """Checks whether 'file_path' can be opened and fully read as bytes.

    Args:
        file_path: Audio file to check.

    Returns:
        'True' if the file was read without error.
    """
    try:
        with file_path.open("rb") as file:
            _ = file.read()
        return True
    except (FileNotFoundError, PermissionError, IsADirectoryError, OSError):
        return False


def check_audio_format(
    *, path: Path, name: str, configuration_path: Path
) -> bool:
    """Checks whether 'path / name' is a valid, uncorrupted audio file.

    Args:
        path: Directory expected to contain the file.
        name: File name, including its extension.
        configuration_path: Path to the TOML configuration file listing the
            valid audio extensions.

    Returns:
        'True' if the file exists, is not corrupted, and its extension is
        one of the '"extensions"' entry in the TOML configuration.
    """
    return (
        check_audio_corruption(path / name)
        and check_path_accessible(configuration_path.parent)
        and configuration_path.is_file()
        and (path / name).is_file()
        and get_extension(path / name) in config.read_config("extensions")
    )
