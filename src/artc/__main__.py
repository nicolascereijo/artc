import importlib.resources
import os
import sys
from typing import cast

from artc import cli
from artc.core import errors
from artc.types import HandleCommandFn, ParseArgsFn


def main() -> None:
    """Entry point for the ARtC command-line suite."""
    commands_path = str(
        importlib.resources.files("artc.cli") / "commands.json"
    )
    logger = errors.logger_config.LoggerSingleton().get_logger()

    if not os.access(commands_path, os.R_OK):
        logger.critical(
            f"Could not access the commands file at {commands_path}: " +
            "check the directory structure and access permissions"
        )
        sys.exit(1)

    # Narrow dynamic CLI entrypoints with explicit type casts.
    parse_args = cast(ParseArgsFn, cli.parse_args)
    handle_command = cast(HandleCommandFn, cli.handle_command)

    parsed_args = parse_args(commands_path, logger=logger)
    command = getattr(parsed_args, "command", "")
    command_args = getattr(parsed_args, "command_args", [])

    handle_command(command, command_args=command_args, logger=logger)


if __name__ == "__main__":
    main()
