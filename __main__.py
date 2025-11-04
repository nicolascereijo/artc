import os
import sys
import importlib.resources

import cli
from core import errors


def main():
    commands_path = str(importlib.resources.files('cli') / 'commands.json')
    logger = errors.logger_config.LoggerSingleton().get_logger()

    if os.access(commands_path, os.R_OK):
        parsed_args = cli.parse_args(commands_path, logger=logger)

        cli.handle_command(
            parsed_args.command,
            command_args=parsed_args.command_args,
            logger=logger
        )
    else:
        logger.critical(
            "Could not access commands file, suite execution aborted. The\n"
            "commands.json file should be located in the /core/cli/ folder.\n"
            "Check the directory and access permissions.")
        sys.exit(1)

if __name__ == "__main__":
    main()
