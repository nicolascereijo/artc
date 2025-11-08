from typing import Callable
from argparse import Namespace
from logging import Logger

ParseArgsFn = Callable[[str, Logger], Namespace]
HandleCommandFn = Callable[[str, list[str], Logger], None]
