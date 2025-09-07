"""
The Search Stream Protocol.

In order to allow the composition of heterogeneous search policies and
prompting policies, a standard protocol is defined for defining
resource-aware search iterators.

A search stream consists in an iterator that yields three kinds of
messages:

- `Solution` messages indicating that a solution has been found.
- `Barrier` messages asking authorization to spend a given amount of
  resources (for which an over-estimate is provided).
- `Spent` messages reporting actual resource spending.

The following invariants and guarantees must be offered and preserved by
stream combinators:

**Invariants**

- `Barrier` and `Spent` messages come in pairs and are associated
  using shared identifiers. Because search streams can spawn mulitple
  threads internally, multiple `Barrier` messages can be simultaneously
  pending (i.e., in waiting of a matching `Spent` message).
- A stream must eventually terminate if all spending requests are denied
  (this is why `loop` has a `stop_on_reject` argument).
- A stream can be interrupted before exhaustion (and be later garbage
  collected), provided that no `Barrier` messages are pending (i.e., an
  identical number of `Barrier` and `Spent` messages has been seen so
  far). If this condition does not hold, some actual resource spending
  might be unreported.

!!! warning
    Manually implementing the search stream protocol by yielding
    `Barrier` and `Spent` messages is error-prone. Standard stream
    combinators should usually be used instead (see `Stream` class from
    the standard library).
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
    vector with finite support. Each dimension corresponds to a
    different metric (e.g., number of requests, price in dollars...)

    Attributes:
        values: a mapping from metrics to spent budget. Metrics outside
            of this field are associated a spending of 0.

    !!! note
        We explicitly use type `float | int` for metrics, despite `int`
        being treated as a subtype of `float` by most Python type
        checkers. This is to avoid pydantic mistakenly converting
        integers into floats when parsing serialized budget data.
    """

    values: Mapping[str, float | int]

    def __getitem__(self, key: str) -> float | int:
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

    Elements outside of the finite support are associated an
    **infinite** limit. Hence the separation of `Budget` and
    `BudgetLimit`.
    """

    values: Mapping[str, float | int]

    def __getitem__(self, key: str) -> float | int:
        return self.values.get(key, math.inf)


#####
##### Generator Streams
#####


class SearchMeta:
    """
    Base class for valid search metadata.

    Search metadata can be attached to all solutions yielded by a search
    stream. See `ProbInfo` for an example.
    """

    pass


type BarrierId = int
"""
A unique identifier associated with a `Barrier` message, used to identify
a matching `Spent` message. Overlapping barrier messages must not share
the same identifier (barrier messages are considered to overlap if the
second one occurs before the `Spent` message associated with the first).
"""


@dataclass(frozen=True)
class Solution[T]:
    """
    A solution yielded by a search stream, which combines a tracked
    value with optional metadata.

    Attributes:
        tracked: A tracked value.
        meta: Optional metadata.
    """

    tracked: Tracked[T]
    meta: SearchMeta | None = None


@dataclass(frozen=False)
class Barrier:
    """
    Ask authorization for spending a given budget amount.

    Attributes:
        budget: an over-estimate of how much budget will be spent if the
            request is granted. An inaccurate estimate can be provided
            by a policy, although more budget could be actually be spent
            than is intended in this case.
        allow: a boolean flag that can be set to `False` by consumers of
            the stream to deny the request.
        id: a unique identifier, which is shared by a unique associated
            `Spent` message, to be yielded later.

    !!! warning
        Manually yielding `Spent` and `Barrier` messages is error-prone
        and usually not recommended. Use stream combinators instead (see
        the `Stream` class from the standard library).
    """

    budget: Budget
    allow: bool
    id: BarrierId

    def __init__(self, budget: Budget, id: BarrierId | None = None):
        import builtins

        self.budget = budget
        self.allow = True
        self.id = id if id is not None else builtins.id(self)
        pass


@dataclass(frozen=True)
class Spent:
    """
    Indicate that an actual amount of resources has been spent.

    Each `Spent` message is associated with a unique prior `Barrier`
    message that shares the same identifier.

    Attributes:
        budget: Amount of budget that was actually spent.
        barrier_id: Identifier of the prior associated `Barrier`
            message.

    !!! warning
        Manually yielding `Spent` and `Barrier` messages is error-prone
        and usually not recommended. Use stream combinators instead (see
        the `Stream` class from the standard library).
    """

    budget: Budget
    barrier_id: BarrierId


type StreamGen[T] = Generator[Solution[T] | Barrier | Spent, None, None]
"""
A search stream generator.

See [delphyne.core.streams][] for more explanations about the search
stream protocol.
"""


type StreamContext[T] = Generator[Barrier | Spent, None, T]
"""
Return type for monadic stream functions.

Consider an operator on streams such as `Stream.first`, which extracts
the first solution from a stream. When such an operator is called within
a stream generator, it is important for all underlying `Spent` and
`Barrier` messages to be forwarded. This is allowed by having this
method return a generator of type `StreamContext`.

More precisely, a value of type `StreamContext[T]` is a generator that
yields `Barrier` and `Spent` messages, and ultimately terminates with a
result of type `T`.
"""


class AbstractStream[T](ABC):
    """
    Base class for search streams.

    A search stream must be capable of producing a search stream
    generator. The standard library contains a subclass with more
    features (`Stream`).
    """

    @abstractmethod
    def gen(self) -> StreamGen[T]:
        """
        Produce a search stream generator, i.e. an iterator that yields
        `Barrier` and `Spent` messages along with solutions.
        """
        pass
