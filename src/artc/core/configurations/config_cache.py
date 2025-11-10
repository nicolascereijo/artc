"""
Runtime configuration cache for ARtC

Provides a lightweight in-memory layer to hold configuration values loaded from
the TOML file. Enables fast, safe access to runtime flags and system settings
without repeated disk reads

Author: Nicolás Cereijo Ranchal
Part of the ARtC (Audio Real-time Comparator) framework
"""

from typing import Any


class ConfigCache:
    """Runtime cache for the ARtC configuration dictionary"""

    def __init__(self, config_data: dict[str, Any]):
        self._data = config_data

    def read(self, section: str) -> Any:
        """Return a specific configuration section or field

        Supported keys:
            - "processes"
            - "memory"
            - "frontier_checks"
            - "full_checks"
            - "sampling"
            - "extensions"
            - "stats"

        Raises:
            KeyError: If the section name is invalid
        """
        mapping = {
            "processes": self._data.get("sysconfig", {}).get("max_processes"),
            "memory": self._data.get("sysconfig", {}).get("max_memory_usage"),
            "frontier_checks": self._data.get("type_flags", {}).get("frontier_checks"),
            "full_checks": self._data.get("type_flags", {}).get("full_checks"),
            "sampling": self._data.get("audio", {}).get("samples_per_chunk"),
            "extensions": self._data.get("audio", {}).get("valid_extensions"),
            "stats": self._data.get("stats", {}).get("values"),
        }

        if section not in mapping:
            raise KeyError(f"Unknown configuration section: '{section}'")
        return mapping[section]

    def get_flags(self) -> tuple[bool, bool]:
        """Return (frontier_checks, full_checks) as a tuple of booleans"""
        flags = self._data.get("type_flags", {})
        return (
            bool(flags.get("frontier_checks", False)),
            bool(flags.get("full_checks", False)),
        )

    def reload(self, new_data: dict[str, Any]) -> None:
        """Reload the cache with new configuration data"""
        self._data = new_data
