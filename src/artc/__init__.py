import importlib
from types import ModuleType

__all__ = ["core", "cli"]

# Declarations only for the static type checker
core: ModuleType
cli: ModuleType


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module  # Cache
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")
