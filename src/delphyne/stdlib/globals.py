"""
Global objects
"""

from collections.abc import Sequence
from pathlib import Path

import delphyne.core_and_base as dp
from delphyne.stdlib.computations import __Computation__
from delphyne.stdlib.universal_queries import UniversalQuery


def stdlib_globals() -> dict[str, object]:
    """
    Return all global objects from the standard library that should
    always be accessible through their identifier in demonstration and
    command files (to be passed to `ObjectLoader` via the
    `extra_objects` option).
    """
    return {
        __Computation__.__name__: __Computation__,
        UniversalQuery.__name__: UniversalQuery,
    }


def stdlib_implicit_answer_generators_loader(
    data_dirs: Sequence[Path],
) -> dp.ImplicitAnswerGeneratorsLoader:
    def loader():
        from delphyne.stdlib import computations

        return [computations.generate_implicit_answer]

    return loader
