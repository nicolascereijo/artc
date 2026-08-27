import importlib
from types import ModuleType

__all__ = ["cli", "core"]

# Declarations only for the static type checker.
core: ModuleType
cli: ModuleType


def __getattr__(name: str) -> ModuleType:
    # 'core' pulls in numpy and librosa on import, and 'cli' pulls in argparse
    # and the command handlers. Deferred here so a caller who only needs one
    # of them isn't forced to pay for the other.
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module  # Cache
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")
