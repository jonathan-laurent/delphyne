"""
Decorators and shortcuts for the Delphyne DSL.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import delphyne.core as dp

#####
##### Strategy Instances `@strategy` Decorator
#####


@dataclass(frozen=True)
class StrategyInstance[N: dp.Node, P, T](dp.StrategyComp[N, P, T]):
    def using[P2](
        self,
        get_policy: Callable[[P2], tuple[dp.SearchPolicy[N], P]],
        inner_policy_type: type[P2] | None = None,
    ) -> dp.Builder[dp.OpaqueSpace[P2, T]]:
        return dp.OpaqueSpace[P2, T].from_strategy(self, get_policy)

    # Pyright seems to be treating __getitem__ differently and does
    # worse inference than for using. Same for operators like &, @...

    # def __getitem__[P2](
    #     self, get_policy: Callable[[P2], tuple[SearchPolicy[N], P]]
    # ) -> Builder[OpaqueSpace[P2, T]]:
    #     return self.using(get_policy)


def strategy[**A, N: dp.Node, P, T](
    f: Callable[A, dp.Strategy[N, P, T]],
) -> Callable[A, StrategyInstance[N, P, T]]:
    def wrapped(
        *args: A.args, **kwargs: A.kwargs
    ) -> StrategyInstance[N, P, T]:
        return StrategyInstance(f, args, kwargs)

    return wrapped


#####
##### Search Policies
#####


class _ParametricSearchPolicyFn[N: dp.Node, **A](Protocol):
    def __call__[P, T](
        self,
        tree: dp.Tree[N, P, T],
        env: dp.PolicyEnv,
        policy: P,
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> dp.Stream[T]: ...


class _ParametricSearchPolicy[N: dp.Node, **A](Protocol):
    def __call__(
        self, *args: A.args, **kwargs: A.kwargs
    ) -> dp.SearchPolicy[N]: ...


def search_policy[N: dp.Node, **A](
    fn: _ParametricSearchPolicyFn[N, A],
) -> _ParametricSearchPolicy[N, A]:
    """
    Wraps a search policy into a callable objects with additional helper
    features (such as the `@` operator for composing with tree
    transformers).
    """

    def parametric(*args: A.args, **kwargs: A.kwargs) -> dp.SearchPolicy[N]:
        def policy[T](
            tree: dp.Tree[N, Any, T], env: dp.PolicyEnv, policy: Any
        ) -> dp.Stream[T]:
            return fn(tree, env, policy, *args, **kwargs)

        return dp.SearchPolicy(policy)

    return parametric


#####
##### Prompting Policies
#####


class _ParametricPromptingPolicyFn[**A](Protocol):
    def __call__[T](
        self,
        query: dp.AttachedQuery[T],
        env: dp.PolicyEnv,
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> dp.Stream[T]: ...


class _ParametricPromptingPolicy[**A](Protocol):
    def __call__(
        self, *args: A.args, **kwargs: A.kwargs
    ) -> dp.PromptingPolicy: ...


def prompting_policy[**A](
    fn: _ParametricPromptingPolicyFn[A],
) -> _ParametricPromptingPolicy[A]:
    def parametric(*args: A.args, **kwargs: A.kwargs) -> dp.PromptingPolicy:
        def policy[T](
            query: dp.AttachedQuery[T], env: dp.PolicyEnv
        ) -> dp.Stream[T]:
            return fn(query, env, *args, **kwargs)

        return dp.PromptingPolicy(policy)

    return parametric
