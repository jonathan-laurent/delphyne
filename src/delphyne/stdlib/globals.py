"""
Global objects
"""

from delphyne.stdlib.computations import __Computation__


def stdlib_globals() -> dict[str, object]:
    return {__Computation__.__name__: __Computation__}
