import pytest
from pathlib import Path

import artc.core.configurations as config


@pytest.fixture()
def setup():
    current_path = Path(__file__)

    if current_path.parent.name == "tests":
        config_path = (
            current_path.parent.parent / "configurations" / "artc_config.toml"
        )
    else:
        config_path = current_path.parent / "configurations" / "artc_config.toml"

    return config_path


def test_load_config(setup):
    configuration_file = setup

    assert len(config.load_config(configuration_file)) != 0

    with pytest.raises(FileNotFoundError):
        config.load_config(Path(""))
    with pytest.raises(FileNotFoundError):
        config.load_config(Path("invalid_configuration_path"))
    with pytest.raises(FileNotFoundError):
        config.load_config(Path("invalid_configuration_path.toml"))


def test_read_config():
    assert len(config.read_config("extensions")) != 0
    assert len(config.read_config("stats")) != 0

    with pytest.raises(KeyError):
        config.read_config("invalid_section")
