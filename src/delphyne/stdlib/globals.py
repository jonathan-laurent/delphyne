"""
Global objects
"""

from delphyne.stdlib.computations import __Computation__


def stdlib_globals() -> dict[str, object]:
    """
    Return all global objects from the standard library that should
    always be accessible through their identifier in demonstration and
    command files (to be passed to `ObjectLoader` via the
    `extra_objects` option).
    """
    return {__Computation__.__name__: __Computation__}
