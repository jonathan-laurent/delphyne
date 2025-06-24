"""
A saturation-based strategy for finding invariants
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import delphyne as dp

# from delphyne import Branch, Computation, Failure, Strategy, Value, strategy
import why3_utils as why3

# fmt: off


@dataclass
class InvariantSuggestions:
    obligation_kind: Literal["post", "init", "preserved"]
    obligation: str
    tricks: Sequence[str]
    suggestions: Sequence[why3.Formula]


@dataclass
class SuggestInvariants(dp.Query[InvariantSuggestions]):
    unproved: why3.Obligation
