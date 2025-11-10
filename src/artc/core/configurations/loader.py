"""
Configuration loader for ARtC

Provides a safe, centralized method for reading and parsing the TOML
configuration file. The resulting dictionary is consumed by the
configuration cache and used across the framework

Author: Nicolás Cereijo Ranchal
Part of the ARtC (Audio Real-time Comparator) framework
"""

from pathlib import Path
from typing import Any

import tomllib


def load_config(path: Path) -> dict[str, Any]:
    """Load and parse the ARtC configuration file

    Args:
        path: Path to the configuration TOML file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the TOML syntax is invalid or unreadable
    """
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML syntax in configuration file: {path}") from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error while loading configuration file: {path}"
        ) from exc
