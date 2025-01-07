"""
Utilities for writing policies.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import delphyne.core as dp

#####
##### Search Policies
#####


@dataclass(frozen=True)
class SearchPolicy[N: dp.Node](dp.AbstractSearchPolicy[N]):
    fn: dp.AbstractSearchPolicy[N]

    def __call__[P, T](
        self,
        tree: "dp.Tree[N, P, T]",
        env: dp.PolicyEnv,
        policy: P,
    ) -> dp.Stream[T]:
        return self.fn(tree, env, policy)

    def __matmul__[M: dp.Node](
        self, tree_transformer: dp.TreeTransformer[M, N]
    ) -> "SearchPolicy[M]":
        def fn[P, T](
            tree: dp.Tree[M, P, T],
            env: dp.PolicyEnv,
            policy: P,
        ) -> dp.Stream[T]:
            new_tree = tree_transformer(tree)
            return self.fn(new_tree, env, policy)

        return SearchPolicy(fn)


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
    ) -> SearchPolicy[N]: ...


def search_policy[N: dp.Node, **A](
    fn: _ParametricSearchPolicyFn[N, A],
) -> _ParametricSearchPolicy[N, A]:
    """
    Wraps a search policy into a callable objects with additional helper
    features (such as the `@` operator for composing with tree
    transformers).
    """

    def parametric(*args: A.args, **kwargs: A.kwargs) -> SearchPolicy[N]:
        def policy[T](
            tree: dp.Tree[N, Any, T], env: dp.PolicyEnv, policy: Any
        ) -> dp.Stream[T]:
            return fn(tree, env, policy, *args, **kwargs)

        return SearchPolicy(policy)

    return parametric


#####
##### Prompting Policies
#####


@dataclass(frozen=True)
class PromptingPolicy(dp.AbstractPromptingPolicy):
    fn: dp.AbstractPromptingPolicy

    def __call__[T](
        self, query: dp.AttachedQuery[T], env: dp.PolicyEnv
    ) -> dp.Stream[T]:
        return self.fn(query, env)


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
    ) -> PromptingPolicy: ...


def prompting_policy[**A](
    fn: _ParametricPromptingPolicyFn[A],
) -> _ParametricPromptingPolicy[A]:
    def parametric(*args: A.args, **kwargs: A.kwargs) -> PromptingPolicy:
        def policy[T](
            query: dp.AttachedQuery[T], env: dp.PolicyEnv
        ) -> dp.Stream[T]:
            return fn(query, env, *args, **kwargs)

        return PromptingPolicy(policy)

    return parametric


#####
##### Logging
#####


def log(
    env: dp.PolicyEnv,
    message: str,
    metadata: dict[str, Any] | None = None,
    loc: dp.Tree[Any, Any, Any] | dp.AttachedQuery[Any] | None = None,
) -> None:
    match loc:
        case None:
            location = None
        case dp.Tree():
            location = dp.Location(loc.ref, None)
        case dp.AttachedQuery(_, ref):
            location = dp.Location(ref[0], ref[1])
    env.tracer.log(message, metadata, location)
