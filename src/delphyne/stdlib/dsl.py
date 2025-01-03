"""
Decorators and shortcuts for the Delphyne DSL.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from delphyne.core.environment import PolicyEnv
from delphyne.core.streams import Stream
from delphyne.core.trees import (
    AttachedQuery,
    Builder,
    Node,
    OpaqueSpace,
    PromptingPolicy,
    SearchPolicy,
    Strategy,
    StrategyComp,
    Tree,
)

#####
##### Strategy Instances `@strategy` Decorator
#####


@dataclass(frozen=True)
class StrategyInstance[N: Node, P, T](StrategyComp[N, P, T]):
    def __getitem__[P2](
        self, get_policy: Callable[[P2], tuple[SearchPolicy[N], P]]
    ) -> Builder[OpaqueSpace[P2, T]]:
        return OpaqueSpace[P2, T].from_strategy(self, get_policy)


def strategy[**A, N: Node, P, T](
    f: Callable[A, Strategy[N, P, T]],
) -> Callable[A, StrategyInstance[N, P, T]]:
    def wrapped(
        *args: A.args, **kwargs: A.kwargs
    ) -> StrategyInstance[N, P, T]:
        return StrategyInstance(f, args, kwargs)

    return wrapped


#####
##### Search Policies
#####


class _ParametricSearchPolicyFn[N: Node, **A](Protocol):
    def __call__[P, T](
        self,
        tree: Tree[N, P, T],
        env: PolicyEnv,
        policy: P,
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> Stream[T]: ...


class _ParametricSearchPolicy[N: Node, **A](Protocol):
    def __call__(
        self, *args: A.args, **kwargs: A.kwargs
    ) -> SearchPolicy[N]: ...


def search_policy[N: Node, **A](
    fn: _ParametricSearchPolicyFn[N, A],
) -> _ParametricSearchPolicy[N, A]:
    """
    Wraps a search policy into a callable objects with additional helper
    features (such as the `@` operator for composing with tree
    transformers).
    """

    def parametric(*args: A.args, **kwargs: A.kwargs) -> SearchPolicy[N]:
        def policy[T](
            tree: Tree[N, Any, T], env: PolicyEnv, policy: Any
        ) -> Stream[T]:
            return fn(tree, env, policy, *args, **kwargs)

        return SearchPolicy(policy)

    return parametric


#####
##### Prompting Policies
#####


class _ParametricPromptingPolicyFn[**A](Protocol):
    def __call__[T](
        self,
        query: AttachedQuery[T],
        env: PolicyEnv,
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> Stream[T]: ...


class _ParametricPromptingPolicy[**A](Protocol):
    def __call__(
        self, *args: A.args, **kwargs: A.kwargs
    ) -> PromptingPolicy: ...


def prompting_policy[**A](
    fn: _ParametricPromptingPolicyFn[A],
) -> _ParametricPromptingPolicy[A]:
    def parametric(*args: A.args, **kwargs: A.kwargs) -> PromptingPolicy:
        def policy[T](query: AttachedQuery[T], env: PolicyEnv) -> Stream[T]:
            return fn(query, env, *args, **kwargs)

        return PromptingPolicy(policy)

    return parametric
