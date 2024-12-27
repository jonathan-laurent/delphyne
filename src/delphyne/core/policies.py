"""
Defining Policies for Delphyne
"""

import math
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from delphyne.core.environment import PolicyEnv
from delphyne.core.trees import Node, Tracked, Tree


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


@dataclass
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


@dataclass
class Yield[T]:
    value: T


@dataclass
class Spent:
    budget: Budget


@dataclass
class Barrier:
    budget: Budget


#####
##### Policies
#####

# The execution monad takes as an argument a general context that
# contains: the path to the templates, some demonstrations and the path
# to data.

# Do we need a `search_policy` decorator? It is nice when policies can
# be composed nicely.


type StreamRet[T] = AsyncGenerator[Yield[Tracked[T]] | Spent | Barrier]


class SearchPolicyFn[N: Node, **P](Protocol):
    # fmt: off
    def __call__[T](
        self,
        tree: Tree[N, T],
        env: PolicyEnv,
        policy: Any,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> StreamRet[T]:
    # fmt: on
        ...


class TreeTransformer[N: Node, M: Node](Protocol):
    def __call__[T](self, tree: Tree[N, T]) -> Tree[M, T]: ...


@dataclass
class SearchPolicy[N: Node, **P]:
    fn: SearchPolicyFn[N, P]

    # fmt: off
    def __call__[T](
        self,
        tree: Tree[N, T],
        env: PolicyEnv,
        policy: Any,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> StreamRet[T]:
    # fmt: on
        return self.fn(tree, env, policy, *args, **kwargs)

    # fmt: off
    def __matmul__[M: Node](
        self,
        tree_transformer: TreeTransformer[M, N]
    ) -> "SearchPolicy[M, P]":

        def fn[T](
            tree: Tree[M, T], env: PolicyEnv, policy: Any,
            *args: P.args, **kwargs: P.kwargs
        ) -> StreamRet[T]:
            new_tree = tree_transformer(tree)
            return self.fn(new_tree, env, policy, *args, **kwargs)

        return SearchPolicy(fn)
    # fmt: on


# fmt: off
def search_policy[N: Node, **P](fn: SearchPolicyFn[N, P]) -> SearchPolicy[N, P]:
    """
    Wraps a search policy into a callable objects with additional helper
    features (such as the `@` operator for composing with tree
    transformers).
    """
    return SearchPolicy(fn)
# fnt: on
