import pytest
from pathlib import Path

import artc.core.configurations as config


@pytest.fixture()
def setup():
    current_path = Path(__file__)

    if current_path.parent.name == "tests":
        config_path = (
            current_path.parent.parent
            / "configurations"
            / "default_configurations.json"
        )
    else:
        config_path = (
            current_path.parent / "configurations" / "default_configurations.json"
        )

    return config_path


def test_open_config(setup):
    configuration_file = setup

    assert len(config.open_config(configuration_file)) != 0

    assert len(config.open_config(Path(""))) == 0
    assert len(config.open_config(Path("invalid_configuration_path"))) == 0
    assert len(config.open_config(Path("invalid_configuration_path.json"))) == 0


def test_read_config(setup):
    configuration_file = setup

    assert len(config.read_config("all", configuration_file)) != 0
    assert len(config.read_config("extensions", configuration_file)) != 0

    assert len(config.read_config("invalid_section", configuration_file)) == 0
    assert len(config.read_config("all", Path("invalid_configuration_path"))) == 0
