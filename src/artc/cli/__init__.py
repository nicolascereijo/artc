import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

__all__ = ["handle_command", "parse_args"]

if TYPE_CHECKING:
    from artc.cli.commands import handle_command, parse_args


def __getattr__(name: str) -> Callable[..., object]:
    if name in __all__:
        module = importlib.import_module(".commands", __package__)

        func: Callable[..., object] = cast(Callable[..., object], getattr(module, name))

        globals()[name] = func  # Cache
        return func

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
