"""
Defining Policies for Delphyne
"""

import math
from collections.abc import AsyncGenerator, Mapping
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

    def __le__(self, limit: "BudgetLimit") -> bool:
        assert isinstance(limit, BudgetLimit)
        for k, v in limit.values.items():
            if self[k] > v:
                return False
        return True

    def __ge__(self, other: "Budget") -> bool:
        assert isinstance(other, Budget)
        for k, v in other.values.items():
            if self[k] < v:
                return False
        return True


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


@dataclass(frozen=True)
class Yield[T]:
    value: T


@dataclass(frozen=True)
class Spent:
    budget: Budget


@dataclass(frozen=True)
class Barrier:
    budget: Budget


type StreamRet[T] = AsyncGenerator[Yield[Tracked[T]] | Spent | Barrier]
