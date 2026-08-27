from pathlib import Path

import pytest

import artc.core.configurations as config


@pytest.fixture()
def setup() -> Path:
    current_path = Path(__file__)

    if current_path.parent.name == "tests":
        config_path = (
            current_path.parent.parent / "configurations" / "artc_config.toml"
        )
    else:
        config_path = (
            current_path.parent / "configurations" / "artc_config.toml"
        )

    return config_path


def test_load_config(setup: Path) -> None:
    configuration_file = setup

    assert len(config.load_config(configuration_file)) != 0

    with pytest.raises(FileNotFoundError):
        _ = config.load_config(Path(""))
    with pytest.raises(FileNotFoundError):
        _ = config.load_config(Path("invalid_configuration_path"))
    with pytest.raises(FileNotFoundError):
        _ = config.load_config(Path("invalid_configuration_path.toml"))


def test_read_config() -> None:
    assert (
        len(config.read_config("extensions"))  # pyright: ignore[reportAny]
        != 0
    )
    assert (
        len(config.read_config("stats"))  # pyright: ignore[reportAny]
        != 0
    )

    with pytest.raises(KeyError):
        config.read_config("invalid_section")
