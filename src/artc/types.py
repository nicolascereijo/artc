"""
Common type aliases and strongly-typed NumPy stubs for ARtC.
Used to improve static analysis across the project (BasedPyright, MyPy).
"""

from argparse import Namespace
from logging import Logger
from typing import Callable, cast

import numpy as np
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────
# CLI callables
# ─────────────────────────────────────────────────────────────
ParseArgsFn = Callable[[str, Logger], Namespace]
HandleCommandFn = Callable[[str, list[str], Logger], None]

# ─────────────────────────────────────────────────────────────
# Core scalar and array types
# ─────────────────────────────────────────────────────────────
# Define a single canonical scalar type used across ARtC.
FloatScalar = np.float32

# Strongly-typed NumPy array containing ARtC’s core scalar type
NDArrayFloat = NDArray[FloatScalar]

# ─────────────────────────────────────────────────────────────
# Common callable signatures for NumPy-style transformations
# ─────────────────────────────────────────────────────────────
UnaryArrayFn = Callable[[NDArrayFloat], NDArrayFloat]
ScalarReduceFn = Callable[[NDArrayFloat], FloatScalar]

# ─────────────────────────────────────────────────────────────
# Typed aliases for common NumPy functions
# ─────────────────────────────────────────────────────────────
# NumPy’s stubs often expose these as untyped (Any),
# so they are explicitly cast for static analyzers.
np_mean: ScalarReduceFn = cast(ScalarReduceFn, np.mean)
np_var: ScalarReduceFn = cast(ScalarReduceFn, np.var)
np_max: ScalarReduceFn = cast(ScalarReduceFn, np.max)
np_min: ScalarReduceFn = cast(ScalarReduceFn, np.min)
np_ravel: UnaryArrayFn = cast(UnaryArrayFn, np.ravel)
