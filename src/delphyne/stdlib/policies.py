"""
Utilities for writing policies.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import delphyne.core as dp

#####
##### Tree Transformers
#####


class PureTreeTransformerFn[A: dp.Node, B: dp.Node](Protocol):
    def __call__[N: dp.Node, P, T](
        self, tree: dp.Tree[A | N, P, T]
    ) -> dp.Tree[B | N, P, T]: ...


class ContextualTreeTransformerFn[A: dp.Node, B: dp.Node](Protocol):
    def __call__[N: dp.Node, P, T](
        self, tree: dp.Tree[A | N, P, T], env: dp.PolicyEnv
    ) -> dp.Tree[B | N, P, T]: ...


@dataclass
class ContextualTreeTransformer[A: dp.Node, B: dp.Node]:
    fn: ContextualTreeTransformerFn[A, B]

    @staticmethod
    def pure(
        fn: PureTreeTransformerFn[A, B],
    ) -> "ContextualTreeTransformer[A, B]":
        def contextual[N: dp.Node, P, T](
            tree: dp.Tree[A | N, P, T], env: dp.PolicyEnv
        ) -> dp.Tree[B | N, P, T]:
            return fn(tree)

        return ContextualTreeTransformer(contextual)

    def __rmatmul__[N: dp.Node](
        self, search_policy: "SearchPolicy[B | N]"
    ) -> "SearchPolicy[A | N]":
        def new_search_policy[P, T](
            tree: dp.Tree[A | N, P, T],
            env: dp.PolicyEnv,
            policy: P,
        ) -> dp.Stream[T]:
            new_tree = self.fn(tree, env)
            return search_policy(new_tree, env, policy)

        return SearchPolicy(new_search_policy)


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
