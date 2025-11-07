import pytest
from pathlib import Path

import artc.core.errors as errors


@pytest.fixture()
def setup():
    current_path = Path(__file__)

    if current_path.parent.name == "tests":
        data_path = current_path.parent / "fixtures"
        config_path = current_path.parent.parent / "configurations"
    else:
        data_path = current_path.parent / "tests" / "fixtures"
        config_path = current_path.parent / "configurations"

    return config_path, "default_configurations.json", data_path


# --------------------------------------------------------------------------------------------------
# Tests for core.errors.validations.file
# --------------------------------------------------------------------------------------------------
def test_get_extension():
    assert errors.get_extension(Path("path_example/file.mp3")) == ".mp3"
    assert errors.get_extension(Path("path_example/file.wav")) == ".wav"

    assert len(errors.get_extension(Path("path_example/file"))) == 0
    assert len(errors.get_extension(Path("path_example/"))) == 0


def test_check_audio_format(setup):
    path, name, data_path = setup
    configuration_file = path / name

    assert errors.check_audio_format(
        path=data_path, name="little-waves.mp3", configuration_path=configuration_file
    )
    assert errors.check_audio_format(
        path=data_path, name="waves-in-caves.wav", configuration_path=configuration_file
    )

    assert (
        errors.check_audio_format(
            path=Path(""), name="", configuration_path=configuration_file
        )
        is False
    )
    assert (
        errors.check_audio_format(
            path=Path(""), name="invalid_file", configuration_path=configuration_file
        )
        is False
    )
    assert (
        errors.check_audio_format(
            path=Path(""),
            name="invalid_file.pdf",
            configuration_path=configuration_file,
        )
        is False
    )
    assert (
        errors.check_audio_format(
            path=data_path, name="invalid_file", configuration_path=configuration_file
        )
        is False
    )


# --------------------------------------------------------------------------------------------------
# Tests for core.errors.validations.path
# --------------------------------------------------------------------------------------------------
def test_check_path_accessible(setup):
    path, _, _ = setup

    assert errors.check_path_accessible(path)

    assert errors.check_path_accessible(Path("")) is False
    assert errors.check_path_accessible(Path("invalid_path")) is False


def test_check_file_readable(setup):
    path, name, _ = setup

    assert errors.check_file_readable(path=path, name=name)

    assert errors.check_file_readable(path=Path(""), name=name) is False
    assert errors.check_file_readable(path=Path("invalid_path"), name=name) is False
    assert errors.check_file_readable(path=path, name="") is False
    assert errors.check_file_readable(path=path, name="invalid_name") is False


def test_validate_path(setup):
    path, name, _ = setup

    assert errors.validate_path(path=path, name=name)

    assert errors.validate_path(path=Path(""), name="") is False
    assert errors.validate_path(path=Path(""), name=name) is False
    assert errors.validate_path(path=Path("invalid_path"), name=name) is False
    assert errors.validate_path(path=path, name="") is False
    assert errors.validate_path(path=Path(""), name="invalid_name") is False
