import os
import sys
import importlib.resources
from typing import cast

from artc.core import errors
from artc import cli
from artc.types import ParseArgsFn, HandleCommandFn


def main() -> None:
    """Entry point for the ARtC command-line suite."""
    commands_path = str(importlib.resources.files("artc.cli") / "commands.json")
    logger = errors.logger_config.LoggerSingleton().get_logger()

    if not os.access(commands_path, os.R_OK):
        logger.critical("""
            Could not access commands file. Suite execution aborted.
            Expected location: /core/cli/commands.json
            Check the directory structure and access permissions.
        """)
        sys.exit(1)

    # Narrow dynamic CLI entrypoints with explicit type casts
    parse_args = cast(ParseArgsFn, cli.parse_args)
    handle_command = cast(HandleCommandFn, cli.handle_command)

    parsed_args = parse_args(commands_path, logger)
    command = getattr(parsed_args, "command", "")
    command_args = getattr(parsed_args, "command_args", [])

    handle_command(command, command_args, logger)


if __name__ == "__main__":
    main()
