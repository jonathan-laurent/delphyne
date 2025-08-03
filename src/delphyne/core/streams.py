"""
Defining Policies for Delphyne
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping
from dataclasses import dataclass

from delphyne.core.refs import Tracked

#####
##### Budget
#####


@dataclass(frozen=True)
class Budget:
    """
    An immutable datastructure for tracking spent budget as an infinite
    vector with finite support.
    """

    values: Mapping[str, float]

    def __getitem__(self, key: str) -> float:
        return self.values.get(key, 0)

    def __add__(self, other: "Budget") -> "Budget":
        vals = dict(self.values).copy()
        for k, v in other.values.items():
            vals[k] = self[k] + v
        return Budget(vals)

    def __rmul__(self, const: float) -> "Budget":
        assert const >= 0
        values = {k: const * v for k, v in self.values.items()}
        return Budget(values)

    def __le__(self, limit: "BudgetLimit") -> bool:
        if not isinstance(limit, BudgetLimit):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented
        for k, v in limit.values.items():
            if self[k] > v:
                return False
        return True

    def __ge__(self, other: "Budget") -> bool:
        if not isinstance(other, Budget):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented
        for k, v in other.values.items():
            if self[k] < v:
                return False
        return True

    @staticmethod
    def zero() -> "Budget":
        return Budget({})


@dataclass(frozen=True)
class BudgetLimit:
    """
    An immutable datastructure for representing a budget limit as an
    infinite vector with finite support.
    """

    values: Mapping[str, float]

    def __getitem__(self, key: str) -> float:
        return self.values.get(key, math.inf)


#####
##### Generator Streams
#####


class SearchMeta:
    """
    All valid search metadata can inherit this class.
    """

    pass


@dataclass(frozen=True)
class Solution[T]:
    tracked: Tracked[T]
    meta: SearchMeta | None = None


@dataclass(frozen=False)
class Barrier:
    budget: Budget
    allow: bool


@dataclass(frozen=True)
class Spent:
    budget: Budget
    barrier: Barrier


type Stream[T] = Generator[Solution[T] | Barrier | Spent, None, None]


type StreamGen[T] = Generator[Barrier | Spent, None, T]
"""
Type signature for a generator that can spend budget, does not yield
results but ultimately returns a result. Useful to define the signature
of `take_one` for example.
"""


class AbstractSearchStream[T](ABC):
    @abstractmethod
    def gen(self) -> Stream[T]:
        pass
