from pathlib import Path

from .config_cache import ConfigCache
from .loader import load_config

__all__ = [
    'DEFAULT_CONFIG_PATH',
    'ConfigCache',
    'config_cache',
    'get_flags',
    'load_config',
    'read_config',
    'reload_config',
]

DEFAULT_CONFIG_PATH = Path(__file__).parent / "artc_config.toml"

config_cache = ConfigCache(load_config(DEFAULT_CONFIG_PATH))

read_config = config_cache.read
reload_config = config_cache.reload
get_flags = config_cache.get_flags
