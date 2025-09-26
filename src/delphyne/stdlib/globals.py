"""
Global objects
"""

from collections.abc import Sequence
from pathlib import Path

import delphyne.core_and_base as dp
from delphyne.stdlib import computations, data
from delphyne.stdlib.universal_queries import UniversalQuery


def stdlib_globals() -> dict[str, object]:
    """
    Return all global objects from the standard library that should
    always be accessible through their identifier in demonstration and
    command files (to be passed to `ObjectLoader` via the
    `extra_objects` option).
    """
    queries = [computations.__Computation__, data.__LoadData__, UniversalQuery]
    return {q.__name__: q for q in queries}


def stdlib_implicit_answer_generators_loader(
    data_dirs: Sequence[Path],
) -> dp.ImplicitAnswerGeneratorsLoader:
    def loader():
        return [
            computations.generate_implicit_answer,
            data.load_implicit_answer_generator(data_dirs),
        ]

    return loader
