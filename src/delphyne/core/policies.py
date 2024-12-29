"""
Defining Policies for Delphyne
"""

import math
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from delphyne.core.environment import PolicyEnv
from delphyne.core.trees import AttachedQuery, Node, Tracked, Tree


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


#####
##### Search Policies
#####


class TreeTransformer[N: Node, M: Node](Protocol):
    def __call__[T](self, tree: Tree[N, T]) -> Tree[M, T]: ...


class _ParametricSearchPolicyFn[N: Node, **P](Protocol):
    def __call__[T](
        self,
        tree: Tree[N, T],
        env: PolicyEnv,
        policy: "Policy[Any]",
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> StreamRet[T]: ...


class _SearchPolicyFn[N: Node](Protocol):
    def __call__[T](
        self, tree: Tree[N, T], env: PolicyEnv, policy: "Policy[Any]"
    ) -> StreamRet[T]: ...


@dataclass(frozen=True)
class SearchPolicy[N: Node]:
    fn: _SearchPolicyFn[N]

    def __call__[T](
        self,
        tree: Tree[N, T],
        env: PolicyEnv,
        policy: "Policy[Any]",
    ) -> StreamRet[T]:
        return self.fn(tree, env, policy)

    def __matmul__[M: Node](
        self, tree_transformer: TreeTransformer[M, N]
    ) -> "SearchPolicy[M]":
        def fn[T](
            tree: Tree[M, T],
            env: PolicyEnv,
            policy: "Policy[Any]",
        ) -> StreamRet[T]:
            new_tree = tree_transformer(tree)
            return self.fn(new_tree, env, policy)

        return SearchPolicy(fn)


class _ParametricSearchPolicy[N: Node, **P](Protocol):
    def __call__(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> SearchPolicy[N]: ...


def search_policy[N: Node, **P](
    fn: _ParametricSearchPolicyFn[N, P],
) -> _ParametricSearchPolicy[N, P]:
    """
    Wraps a search policy into a callable objects with additional helper
    features (such as the `@` operator for composing with tree
    transformers).
    """

    def parametric(*args: P.args, **kwargs: P.kwargs) -> SearchPolicy[N]:
        def policy[T](
            tree: Tree[N, T], env: PolicyEnv, policy: "Policy[Any]"
        ) -> StreamRet[T]:
            return fn(tree, env, policy, *args, **kwargs)

        return SearchPolicy(policy)

    return parametric


#####
##### Prompting Policies
#####


class _PromptingPolicyFn(Protocol):
    def __call__[T](
        self, query: AttachedQuery[T], env: PolicyEnv
    ) -> StreamRet[T]: ...


class _ParametricPromptingPolicyFn[**P](Protocol):
    def __call__[T](
        self,
        query: AttachedQuery[T],
        env: PolicyEnv,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> StreamRet[T]: ...


@dataclass(frozen=True)
class PromptingPolicy:
    fn: _PromptingPolicyFn

    def __call__[T](
        self, query: AttachedQuery[T], env: PolicyEnv
    ) -> StreamRet[T]:
        return self.fn(query, env)


class _ParametricPromptingPolicy[**P](Protocol):
    def __call__(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> PromptingPolicy: ...


def prompting_policy[**P](
    fn: _ParametricPromptingPolicyFn[P],
) -> _ParametricPromptingPolicy[P]:
    def parametric(*args: P.args, **kwargs: P.kwargs) -> PromptingPolicy:
        def policy[T](query: AttachedQuery[T], env: PolicyEnv) -> StreamRet[T]:
            return fn(query, env, *args, **kwargs)

        return PromptingPolicy(policy)

    return parametric


#####
##### Policies
#####


class Policy[N: Node](ABC):
    @abstractmethod
    def toplevel(self) -> SearchPolicy[N]:
        pass
