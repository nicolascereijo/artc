import json
from pathlib import Path
from typing import Any

from .. import errors


logger = errors.logger_config.LoggerSingleton().get_logger()


def open_config(configuration_path: Path) -> dict:
    try:
        if (
            errors.check_path_accessible(configuration_path.parent)
            and configuration_path.is_file()
        ):
            with configuration_path.open("r") as file:
                return json.load(file)
    except ValueError:
        logger.critical("Could not access or parse configuration file")

    return {}


def read_config(config_section: str, configuration_path: Path) -> Any:
    """
    Read and retrieve configuration settings from a specified section of a configuration file.

    Args:
        config_section (str): The section of the configuration file to read. Available options:
            - "all": Returns the entire configuration file as a dictionary.
            - "processes": Returns the maximum number of allowed processes.
            - "memory": Returns the maximum memory usage setting from the "sysconfig" section.
            - "sampling": Returns the number of samples per audio chunk.
            - "extensions": Returns the list of valid audio file extensions.
            - "stats": Returns the list of statistics available for comparisons.
        configuration_path (Path): The path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration settings for the specified section.
        float/int: An specific value from a configuration field.
        None: If the specified section does not exist.

    Note:
        This function assumes that the configuration file follows a specific structure divided
        into sections, such as 'sysconfig' or 'audio', which contain different settings. It
        returns different parts of the specified section based on the 'config_section'
        parameter. If the specified section or the configuration file itself does not exist,
        appropriate error messages are logged.
    """
    config = open_config(configuration_path)

    if config is not None:
        cases = {
            "all": config,
            "processes": config.get("sysconfig", {}).get("max_processes"),
            "memory": config.get("sysconfig", {}).get("max_memory_usage"),
            "sampling": config.get("audio", {}).get("samples_per_chunk"),
            "extensions": config.get("audio", {}).get("valid_extensions", []),
            "stats": config.get("stats", []),
        }

        case_function = cases.get(config_section)
        if case_function is not None:
            return case_function
        else:
            logger.error(
                "The specified section of the configuration file does not exist."
            )

    else:
        logger.critical("""
            Error, configuration file does not exist or is not accessible.
            A configuration file is required to run the program.
        """)

    return {}
