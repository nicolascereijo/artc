import os
import sys
import argparse
import json
import importlib.resources
from logging import Logger
from typing import Callable, TypedDict, cast

from artc.core import tests


class _CommandsConfig(TypedDict, total=False):
    commands: list[str]


def about() -> None:
    print("""
        ARtC (Audio Real-time Comparator)
        ================================

        ARtC is a software suite designed to compare and analyze audio files in real time.

        Version: 1.0b4
        Author: Nicolás Cereijo Ranchal
        Author Email: nicolascereijo.careers@protonmail.com
        URL: https://github.com/NicolasCereijo/artc
        License: MIT

        Development Status:
        -------------------
        This project is currently in active development. The goal is to provide a reliable tool
        for real-time audio comparison, but certain features may still be under testing or not
        fully stable.
        """)


def load_commands(commands_path: str, *, logger: Logger) -> list[str]:
    """
    Opens and reads a JSON file containing a list of commands. If the file is found and
    correctly formatted, the list of commands is returned. If the file is missing or an error
    occurs during reading, a warning is logged and a default list of commands is returned.

    Args:
        commands_path (str): The path to the JSON file containing the commands.
        logger (Logger): A logger instance used to log warnings if the file is missing or
            invalid.

    Returns:
        List[str]: A list of command names loaded from the file, or a default list if an error
            occurs.
    """
    try:
        with open(commands_path, "r") as file:
            config = cast(_CommandsConfig, json.load(file))
            return config.get("commands", [])
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("""
            The JSON file was not found, a default list of commands is
            being used instead.
            """)
        return ["welcome", "test"]


def parse_args(commands_path: str, *, logger: Logger) -> argparse.Namespace:
    """
    Configures an argument parser with predefined commands loaded from a JSON file. The primary
    command is optional and defaults to "welcome" if not provided. Additional arguments for the
    selected command are captured and stored for further processing.

    Args:
        commands_path (str): The path to the JSON file containing the available commands.
        logger (Logger): A logger instance used to log warnings or errors when loading commands.

    Returns:
        argparse.Namespace: A namespace object containing the parsed arguments.

    Raises:
        SystemExit: If an invalid command or arguments are provided, the program exits with an
            error message.
    """
    parser = argparse.ArgumentParser(
        description="Main argument parser for the ARtC suite", exit_on_error=False
    )

    _ = parser.add_argument(
        "command",
        choices=load_commands(commands_path, logger=logger),
        nargs="?",
        default="welcome",
        help="Primary command to execute. Valid options include the globally defined ARtC commands.",
    )
    _ = parser.add_argument(
        "command_args",
        nargs=argparse.REMAINDER,
        help="Optional additional arguments for the selected command, passed as-is to the handler.",
    )

    try:
        return parser.parse_args()
    except Exception:
        logger.error("""
            Invalid command or arguments provided, please check the available commands.
            """)
        sys.exit(1)


def handle_command(command: str, *, command_args: list[str], logger: Logger) -> None:
    """
    Executes a given command by delegating it to the appropriate handler.

    The function processes the specified command and performs the necessary actions, which may
    include executing tasks, logging messages or validating configuration files. If required
    resources are unavailable, it logs critical errors and may terminate execution.

    When the 'test' command is used, the function automatically adds a default configuration
    file (`artc/core/tests/pytest.ini`) for `pytest`. This can be overwritten by specifying a
    custom configuration file using the `-c` argument, e.g., `python -m artc test -c
    path_to_custom_config/pytest.ini`.

    Args:
        command (str): The command to execute.
        command_args (list): Additional arguments passed to the command handler. Defaults to an
            empty list.
        logger (Logger): A logger instance used to log messages and errors.

    Raises:
        SystemExit: If a critical issue occurs that prevents execution from continuing.
    """

    if command == "test":
        test_main = cast(Callable[[list[str] | None], int], tests.main)
        _ = test_main(["-c", "src/artc/core/tests/pytest.ini"] + command_args)

    elif command == "welcome":
        configuration_path = str(
            importlib.resources.files("artc.core.configurations")
            / "default_configurations.json"
        )

        logger.info("""
            Starting the ARtC suite...\n
                |     '||''|.     .     ..|'''.|      .|'''.|            ||    .
               |||     ||   ||  .||.  .|'      '      ||..  '  ... ...  ...  .||.    ....
              |  ||    ||''|'    ||   ||               ''|||.   ||  ||   ||   ||   .|...||
             .''''|.   ||   |.   ||   '|.      .     .     '||  ||  ||   ||   ||   ||
            .|.  .||. .||.  '|' .||.   ''|....'      |'....|'   '|..'|. .||.  '|.'  '|...'\n
            """)
        about()

        if not os.access(configuration_path, os.R_OK):
            logger.critical(
                """Could not access configuration file, suite execution aborted. The
                default_configurations.json file should be located in the /core/configurations/
                folder. Check the directory and access permissions."""
            )
            sys.exit(1)
