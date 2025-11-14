"""
Common type aliases, NumPy stubs and runtime type decorators for ARtC

Provides core scalar and array aliases used throughout the framework, as well
as optional runtime type enforcement decorators controlled by the configuration
flags defined in [type_flags] of the TOML file

Author: Nicolás Cereijo Ranchal
Part of the ARtC (Audio Real-time Comparator) framework
"""

import inspect
from argparse import Namespace
from functools import wraps
from logging import Logger
from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray

from artc.core.configurations import get_flags

# ─────────────────────────────────────────────────────────────
# CLI callables
# ─────────────────────────────────────────────────────────────
ParseArgsFn = Callable[[str, Logger], Namespace]
HandleCommandFn = Callable[[str, list[str], Logger], None]


# ─────────────────────────────────────────────────────────────
# Core scalar and array types
# ─────────────────────────────────────────────────────────────
FloatScalar = np.float32
NDArrayFloat = NDArray[FloatScalar]


# ─────────────────────────────────────────────────────────────
# Common callable signatures for NumPy-style transformations
# ─────────────────────────────────────────────────────────────
UnaryArrayFn = Callable[[NDArrayFloat], NDArrayFloat]
ScalarReduceFn = Callable[[NDArrayFloat], FloatScalar]


# ─────────────────────────────────────────────────────────────
# Typed aliases for common NumPy functions
# ─────────────────────────────────────────────────────────────
np_mean: ScalarReduceFn = cast(ScalarReduceFn, np.mean)
np_var: ScalarReduceFn = cast(ScalarReduceFn, np.var)
np_max: ScalarReduceFn = cast(ScalarReduceFn, np.max)
np_min: ScalarReduceFn = cast(ScalarReduceFn, np.min)
np_ravel: UnaryArrayFn = cast(UnaryArrayFn, np.ravel)


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────
def _get_param_value(func, args, kwargs, param_name: str) -> Any:
    """Retrieve the value of a parameter (by name) passed to a function

    Uses the function signature for accurate positional/keyword mapping, even
    when multiple decorators are stacked
    """
    original = inspect.unwrap(func)

    sig = inspect.signature(original)
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()

    return bound.arguments.get(param_name, None)


def _should_skip_check(level: str) -> bool:
    """Return True if type checking should be skipped based on config flags

    Logic:
        - If full_checks is enabled, always perform type checking
        - If only frontier_checks is enabled, check only frontier-level decorators
        - If both are disabled, skip all type checks
    """
    frontier_enabled, full_enabled = get_flags()

    if full_enabled:
        return False
    # if only frontier is enabled, skip if this decorator is not frontier
    if level == "frontier_checks":
        return not frontier_enabled
    return True


# ─────────────────────────────────────────────────────────────
# Decorator with dynamic flag control
# ─────────────────────────────────────────────────────────────
def NDArrayFloatCheck(param_name: str, level: str = "frontier_checks"):
    """Ensure the parameter is NDArrayFloat, conditional on config flag"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _should_skip_check(level):
                return func(*args, **kwargs)

            value = _get_param_value(func, args, kwargs, param_name)
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Parameter '{param_name}' must be NDArrayFloat, got {type(value).__name__}"
                )
            if value.dtype != FloatScalar:
                raise TypeError(
                    f"Parameter '{param_name}' must have dtype=FloatScalar, got dtype={value.dtype}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator
