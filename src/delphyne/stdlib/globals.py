"""
Global objects
"""

from delphyne.analysis import ObjectLoader
from delphyne.stdlib.computations import __Computation__


def register_stdlib_globals() -> None:
    ObjectLoader.register_global(__Computation__)
