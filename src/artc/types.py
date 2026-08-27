import inspect
from argparse import Namespace
from collections.abc import Callable
from functools import wraps
from logging import Logger
from typing import Any, Literal, ParamSpec, Protocol, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

P = ParamSpec("P")
R = TypeVar("R")


# ─────────────────────────────────────────────────────────────
# CLI callables
# ─────────────────────────────────────────────────────────────
class ParseArgsFn(Protocol):
    def __call__(self, commands_path: str, *, logger: Logger) -> Namespace: ...


class HandleCommandFn(Protocol):
    def __call__(
        self, command: str, *, command_args: list[str], logger: Logger
    ) -> None: ...


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
def _get_param_value(  # pyright: ignore[reportAny]
    func: Callable[..., Any],  # pyright: ignore[reportExplicitAny]
    args: tuple[Any, ...],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    param_name: str,
) -> Any:  # pyright: ignore[reportExplicitAny]
    """Reads the value bound to one parameter of a call to 'func'.

    Uses 'func's own signature to resolve positional and keyword arguments
    accurately, even when multiple decorators are stacked around it.

    Args:
        func: Callable whose signature is used to resolve 'param_name'.
        args: Positional arguments 'func' was called with.
        kwargs: Keyword arguments 'func' was called with.
        param_name: Name of the parameter to read.

    Returns:
        The value bound to 'param_name', or 'None' if it was not supplied
        and has no default.
    """
    original = inspect.unwrap(func)  # pyright: ignore[reportAny]

    sig = inspect.signature(original)  # pyright: ignore[reportAny]
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()

    return bound.arguments.get(param_name, None)


def _should_skip_check(
    level: Literal["frontier_checks", "full_checks"],
) -> bool:
    """Decides whether a type check at the given level should be skipped.

    'full_checks' being enabled always runs every check, regardless of
    'level'. Otherwise, only checks whose own 'level' is 'frontier_checks'
    run, and only if that flag is itself enabled.

    Args:
        level: Configuration flag gating this particular check.

    Returns:
        'True' if the check should be skipped.
    """
    # Deferred because 'artc.core' imports 'task_manager', which imports this
    # module's decorators at module scope. A top level import here would make
    # 'artc.types' depend on 'artc.core' finishing its own import first, which
    # is circular whenever 'artc.types' is imported before 'artc.core' is.
    from artc.core.configurations import get_flags

    frontier_enabled, full_enabled = get_flags()

    if full_enabled:
        return False
    if level == "frontier_checks":
        return not frontier_enabled
    return True


# ─────────────────────────────────────────────────────────────
# Decorator with dynamic flag control
# ─────────────────────────────────────────────────────────────
def NDArrayFloatCheck(
    param_name: str,
    level: Literal["frontier_checks", "full_checks"] = "frontier_checks",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Builds a decorator that checks one parameter is a valid NDArrayFloat.

    The check itself only runs when 'level' is enabled in the '[type_flags]'
    section of the TOML configuration, see '_should_skip_check'. This lets
    the framework ship the check everywhere while keeping it opt-in at
    runtime, since walking every parameter's signature on every call has a
    real cost.

    Args:
        param_name: Name of the decorated function's parameter to check.
        level: Configuration flag gating the check.

    Returns:
        A decorator that raises 'TypeError' when the check is enabled and
        'param_name' is not an 'NDArrayFloat' with the right dtype, and
        otherwise calls the decorated function unchanged.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if _should_skip_check(level):
                return func(*args, **kwargs)

            value = _get_param_value(  # pyright: ignore[reportAny]
                func, args, kwargs, param_name
            )
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Parameter '{param_name}' must be NDArrayFloat, got " +
                    f"{type(value).__name__}"  # pyright: ignore[reportAny]
                )
            if value.dtype != FloatScalar:
                raise TypeError(
                    f"Parameter '{param_name}' must have dtype=FloatScalar, " +
                    f"got dtype={value.dtype}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator
