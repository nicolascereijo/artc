import importlib.resources
import os
import sys

import pytest

from .. import errors


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    configuration_path = str(
        importlib.resources.files("artc.core.configurations") / "artc_config.json"
    )
    logger = errors.logger_config.LoggerSingleton().get_logger()

    logger.info("""
        Running the main test suite for ARtC...\n
            |     '||''|.     .     ..|'''.|      .|'''.|            ||    .
           |||     ||   ||  .||.  .|'      '      ||..  '  ... ...  ...  .||.    ....
          |  ||    ||''|'    ||   ||               ''|||.   ||  ||   ||   ||   .|...||
         .''''|.   ||   |.   ||   '|.      .     .     '||  ||  ||   ||   ||   ||
        .|.  .||. .||.  '|' .||.   ''|....'      |'....|'   '|..'|. .||.  '|.'  '|...'\n
    """)

    if os.access(configuration_path, os.R_OK):
        result = pytest.main(args)

        if result == 0:
            logger.info("All executed tests were successful")
        else:
            logger.error("Bugs were found in the test set during execution")
    else:
        logger.critical("""
            Could not access configuration file, suite execution aborted. The
            default_configurations.json file should be located in the /core/configurations/
            folder. Check the directory and access permissions.
        """)
        sys.exit(1)
