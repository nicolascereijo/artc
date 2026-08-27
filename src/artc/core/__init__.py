import importlib
from collections.abc import Callable
from types import ModuleType
from typing import cast

from artc.types import NDArrayFloat

from . import analysis, configurations, datastructures, errors

# Declarations only for the static type checker. 'compare' is spelled out
# by hand, instead of importing it from 'task_manager' under
# 'TYPE_CHECKING', because that import is still resolved for type analysis
# and would form the same cycle described below.
ensembles: ModuleType
compare: Callable[..., list[tuple[str, NDArrayFloat]]]

__all__ = [
    "analysis",
    "compare",
    "configurations",
    "datastructures",
    "ensembles",
    "errors",
]


def __getattr__(name: str) -> ModuleType | Callable[..., object]:
    # 'ensembles' (and its 'charts' dependency) pull in scikit-learn and
    # matplotlib, which callers who only need 'compare()' aren't forced to
    # pay for on every 'import artc.core'.
    if name == "ensembles":
        module = importlib.import_module(".ensembles", __name__)
        globals()[name] = module  # Cache
        return module

    # 'task_manager' imports 'analysis' and 'configurations' back through
    # this package, so importing 'compare' here at module load time forms
    # an import cycle. Deferred until first use instead.
    if name == "compare":
        module = importlib.import_module(".task_manager", __name__)
        func = cast(Callable[..., object], module.compare)
        globals()[name] = func  # Cache
        return func

    raise AttributeError(f"module {__name__} has no attribute {name}")
