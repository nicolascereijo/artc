from typing import Any


def _validate_schema(
    data: dict[str, Any],  # pyright: ignore[reportExplicitAny]
) -> None:
    """Checks presence and type of every section 'ConfigCache.read' exposes.

    A missing or malformed section here would otherwise fail much later,
    with a raw TypeError deep inside an unrelated consumer (e.g. audio
    format checks), instead of a clear error at load time.

    Args:
        data: Parsed configuration dictionary to validate.

    Raises:
        ValueError: If a required section is missing or has the wrong type.
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

    problems: list[str] = []
    for path, expected_type in required:
        node: Any = data  # pyright: ignore[reportExplicitAny]
        for key in path:
            if not isinstance(node, dict) or key not in node:
                problems.append(f"missing '{'.'.join(path)}'")
                break
            node = node[key]
        else:
            # This uses an exact type match rather than isinstance, because
            # 'bool' is a subclass of 'int' in Python and
            # 'isinstance(True, int)' is 'True'. Without that,
            # 'max_processes = true' would silently pass validation and later
            # be read as '1'. 'tomllib' only ever produces the plain builtin
            # types listed in 'required' above, never subclasses of them.
            if type(node) is not expected_type:
                problems.append(
                    f"'{'.'.join(path)}' must be {expected_type.__name__}, " +
                    f"got {type(node).__name__}"
                )

    if problems:
        raise ValueError("Invalid ARtC configuration: " + "; ".join(problems))


class ConfigCache:
    """Runtime cache for the ARtC configuration dictionary."""

    _data: dict[str, Any]  # pyright: ignore[reportExplicitAny]

    def __init__(
        self, config_data: dict[str, Any]  # pyright: ignore[reportExplicitAny]
    ):
        """Validates and stores the parsed configuration dictionary.

        Args:
            config_data: Parsed configuration dictionary, as returned by
                'load_config'.

        Raises:
            ValueError: If 'config_data' is missing a required section or has
                the wrong type, see '_validate_schema'.
        """
        _validate_schema(config_data)
        self._data = config_data

    def read(
        self, section: str | tuple[str, str]
    ) -> Any:  # pyright: ignore[reportExplicitAny]
        """Returns a specific configuration section or field.

        Args:
            section: Either one of the supported section names below, or a
                '(category, metric_name)' pair to read that metric's window
                parameter.

        Supported section names:
            'processes', 'memory', 'frontier_checks', 'full_checks',
            'sampling', 'extensions', 'stats' and
            '("window_parameter", metric_name)'.

        Returns:
            The value stored under 'section'.

        Raises:
            KeyError: If 'section' (or 'metric_name') is not one of the
                supported keys.
        """
        if isinstance(section, tuple):
            category, metric_name = section
            if category != "window_parameter":
                raise KeyError(f"Unknown configuration section: '{section}'")

            window_parameters = self._data.get("metric", {}).get(
                "window_parameter", {}
            )
            if metric_name not in window_parameters:
                raise KeyError(
                    f"Unknown metric for window_parameter: '{metric_name}'"
                )
            return window_parameters[metric_name]

        mapping = {
            "processes": self._data.get("sysconfig", {}).get("max_processes"),
            "memory": self._data.get("sysconfig", {}).get("max_memory_usage"),
            "frontier_checks": self._data.get("type_flags", {}).get(
                "frontier_checks"
            ),
            "full_checks": self._data.get("type_flags", {}).get("full_checks"),
            "sampling": self._data.get("audio", {}).get("samples_per_chunk"),
            "extensions": self._data.get("audio", {}).get("valid_extensions"),
            "stats": self._data.get("stats", {}).get("values"),
        }

        if section not in mapping:
            raise KeyError(f"Unknown configuration section: '{section}'")
        return mapping[section]

    def get_flags(self) -> tuple[bool, bool]:
        """Returns the '(frontier_checks, full_checks)' flags.

        Returns:
            A tuple of '(frontier_checks, full_checks)' booleans.
        """
        flags = self._data.get("type_flags", {})
        return (
            bool(flags.get("frontier_checks", False)),
            bool(flags.get("full_checks", False)),
        )

    def reload(
        self, new_data: dict[str, Any]  # pyright: ignore[reportExplicitAny]
    ) -> None:
        """Reloads the cache with new configuration data.

        Args:
            new_data: Parsed configuration dictionary to replace the current
                one with.

        Raises:
            ValueError: If 'new_data' is missing a required section or has the
                wrong type, see '_validate_schema'.
        """
        _validate_schema(new_data)
        self._data = new_data
