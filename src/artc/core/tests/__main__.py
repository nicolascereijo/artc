import importlib.resources
import os
import sys

import pytest

from .. import errors


def main(args: list[str] | None = None) -> int:
    """Runs the ARtC test suite and returns pytest's exit code.

    Args:
        args: Command line arguments to pass to 'pytest.main'. Defaults to
            'sys.argv[1:]' when 'None'.

    Returns:
        '0' if every test passed, pytest's exit code otherwise.
    """
    if args is None:
        args = sys.argv[1:]

    configuration_path = str(
        importlib.resources.files("artc.core.configurations")
        / "artc_config.toml"
    )
    logger = errors.logger_config.LoggerSingleton().get_logger()

    logger.info("""
        Running the main test suite for ARtC...

            |     '||''|.     .     ..|'''.|      .|'''.|            ||    .
           |||     ||   ||  .||.  .|'      '      ||..  '  ... ...  ...  .||.    ....
          |  ||    ||''|'    ||   ||               ''|||.   ||  ||   ||   ||   .|...||
         .''''|.   ||   |.   ||   '|.      .     .     '||  ||  ||   ||   ||   ||
        .|.  .||. .||.  '|' .||.   ''|....'      |'....|'   '|..'|. .||.  '|.'  '|...'
    """)

    if os.access(configuration_path, os.R_OK):
        result = int(pytest.main(args))

        if result == 0:
            logger.info("All executed tests were successful")
        else:
            logger.error("Bugs were found in the test set during execution")

        return result

    logger.critical(
        f"Could not access the configuration file at {configuration_path}: " +
        "check the directory structure and access permissions"
    )
    sys.exit(1)
