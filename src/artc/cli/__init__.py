import importlib
from typing import TYPE_CHECKING, Callable, cast

__all__ = ["parse_args", "handle_command"]

if TYPE_CHECKING:
    from artc.cli.commands import parse_args, handle_command


def __getattr__(name: str) -> Callable[..., object]:
    if name in __all__:
        module = importlib.import_module(".commands", __package__)

        func: Callable[..., object] = cast(Callable[..., object], getattr(module, name))

        globals()[name] = func  # Cache
        return func

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
