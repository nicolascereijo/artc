import importlib
from types import ModuleType

from .task_manager import compare
from . import analysis
from . import configurations
from . import datastructures
from . import errors

# Declaration only for the static type checker.
ensembles: ModuleType

__all__ = ["compare", "analysis", "configurations", "datastructures", "errors", "ensembles"]


def __getattr__(name: str) -> ModuleType:
    # `ensembles` (and its `charts` dependency) pull in scikit-learn and
    # matplotlib, which callers who only need `compare()` shouldn't have to
    # pay for on every `import artc.core`. Deferred here rather than
    # imported above, following the same pattern as `artc/__init__.py`.
    if name == "ensembles":
        module = importlib.import_module(".ensembles", __name__)
        globals()[name] = module  # Cache
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")
