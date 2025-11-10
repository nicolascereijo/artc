"""
Configuration submodule for ARtC

Handles the loading, caching and access of the global configuration
file (`artc_config.toml`) at runtime

The configuration is parsed once using `tomllib` and kept in memory through a
lightweight cache object, ensuring near-zero overhead during subsequent reads

Author: Nicolás Cereijo Ranchal
Part of the ARtC (Audio Real-time Comparator) framework
"""

from pathlib import Path

from .config_cache import ConfigCache
from .loader import load_config

# Default configuration path
DEFAULT_CONFIG_PATH = Path(__file__).parent / "artc_config.toml"

# Global configuration cache
config_cache = ConfigCache(load_config(DEFAULT_CONFIG_PATH))

# Re-exposed public API
read_config = config_cache.read
reload_config = config_cache.reload
get_flags = config_cache.get_flags
