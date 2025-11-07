import os
import sys
import importlib.resources
from logging import Logger
from argparse import Namespace
from typing import Callable, cast

from core import errors
import cli


def main() -> None:
    commands_path = str(importlib.resources.files("cli") / "commands.json")
    logger: Logger = errors.logger_config.LoggerSingleton().get_logger()

    if not os.access(commands_path, os.R_OK):
        logger.critical(
            """Could not access commands file, suite execution aborted. The
            commands.json file should be located in the /core/cli/ folder.
            Check the directory and access permissions."""
        )
        sys.exit(1)

    parsed_args: Namespace = cli.parse_args(commands_path, logger=logger)

    handle_command = cast(
        Callable[[str, list[str], Logger], None],
        cli.handle_command,
    )

    command = getattr(parsed_args, "command", "")
    command_args = getattr(parsed_args, "command_args", [])

    handle_command(command, command_args, logger)


if __name__ == "__main__":
    main()
