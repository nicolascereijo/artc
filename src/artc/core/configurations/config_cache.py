"""
Runtime configuration cache for ARtC

Provides a lightweight in-memory layer to hold configuration values loaded from
the TOML file. Enables fast, safe access to runtime flags and system settings
without repeated disk reads

Author: Nicolás Cereijo Ranchal
Part of the ARtC (Audio Real-time Comparator) framework
"""

from typing import Any


def _validate_schema(data: dict[str, Any]) -> None:
    """Check presence and type of every section ConfigCache.read() exposes

    A missing or malformed section here would otherwise fail much later,
    with a raw TypeError deep inside an unrelated consumer (e.g. audio
    format checks), instead of a clear error at load time.

    Raises:
        ValueError: If a required section is missing or has the wrong type
    """
    required: list[tuple[tuple[str, ...], type]] = [
        (("sysconfig", "max_processes"), int),
        (("sysconfig", "max_memory_usage"), int),
        (("type_flags", "frontier_checks"), bool),
        (("type_flags", "full_checks"), bool),
        (("audio", "samples_per_chunk"), int),
        (("audio", "valid_extensions"), list),
        (("stats", "values"), list),
        (("metric", "window_parameter"), dict),
    ]

    problems = []
    for path, expected_type in required:
        node: Any = data
        for key in path:
            if not isinstance(node, dict) or key not in node:
                problems.append(f"missing '{'.'.join(path)}'")
                break
            node = node[key]
        else:
            if not isinstance(node, expected_type):
                problems.append(
                    f"'{'.'.join(path)}' must be {expected_type.__name__}, "
                    f"got {type(node).__name__}"
                )

    if problems:
        raise ValueError("Invalid ARtC configuration: " + "; ".join(problems))


class ConfigCache:
    """Runtime cache for the ARtC configuration dictionary"""

    def __init__(self, config_data: dict[str, Any]):
        _validate_schema(config_data)
        self._data = config_data

    def read(self, section: str | tuple[str, str]) -> Any:
        """Return a specific configuration section or field

        Supported keys:
            - "processes"
            - "memory"
            - "frontier_checks"
            - "full_checks"
            - "sampling"
            - "extensions"
            - "stats"
            - ("window_parameter", metric_name)

        Raises:
            KeyError: If the section name (or metric_name) is invalid
        """
        if isinstance(section, tuple):
            category, metric_name = section
            if category != "window_parameter":
                raise KeyError(f"Unknown configuration section: '{section}'")

            window_parameters = self._data.get("metric", {}).get("window_parameter", {})
            if metric_name not in window_parameters:
                raise KeyError(f"Unknown metric for window_parameter: '{metric_name}'")
            return window_parameters[metric_name]

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
        _validate_schema(new_data)
        self._data = new_data
