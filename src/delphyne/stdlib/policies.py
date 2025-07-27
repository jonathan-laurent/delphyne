"""
Utilities for writing policies.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import delphyne.core as dp
from delphyne.core import Node, PolicyEnv

#####
##### Tree Transformers
#####


class PureTreeTransformerFn[A: Node, B: Node](Protocol):
    def __call__[N: Node, P, T](
        self, tree: dp.Tree[A | N, P, T]
    ) -> dp.Tree[B | N, P, T]: ...


class _ContextualTreeTransformerFn[A: Node, B: Node](Protocol):
    def __call__(
        self, env: PolicyEnv, policy: Any
    ) -> PureTreeTransformerFn[A, B]: ...


class _ParametricContextualTreeTransformerFn[A: Node, B: Node, **C](Protocol):
    def __call__(
        self, env: PolicyEnv, policy: Any, *args: C.args, **kwargs: C.kwargs
    ) -> PureTreeTransformerFn[A, B]: ...


@dataclass
class ContextualTreeTransformer[A: Node, B: Node]:
    fn: _ContextualTreeTransformerFn[A, B]

    @staticmethod
    def pure(
        fn: PureTreeTransformerFn[A, B],
    ) -> "ContextualTreeTransformer[A, B]":
        def contextual(env: PolicyEnv, policy: Any):
            return fn

        return ContextualTreeTransformer(contextual)

    def __rmatmul__[N: Node](
        self, search_policy: "SearchPolicy[B | N]"
    ) -> "SearchPolicy[A | N]":
        def new_search_policy[P, T](
            tree: dp.Tree[A | N, P, T],
            env: PolicyEnv,
            policy: P,
        ) -> dp.Stream[T]:
            new_tree = self.fn(env, policy)(tree)
            return search_policy(new_tree, env, policy)

        return SearchPolicy(new_search_policy)


def contextual_tree_transformer[A: Node, B: Node, **C](
    f: _ParametricContextualTreeTransformerFn[A, B, C], /
) -> Callable[C, ContextualTreeTransformer[A, B]]:
    def parametric(*args: C.args, **kwargs: C.kwargs):
        def contextual(env: PolicyEnv, policy: Any):
            return f(env, policy, *args, **kwargs)

        return ContextualTreeTransformer(contextual)

    return parametric


#####
##### Search Policies
#####


@dataclass(frozen=True)
class SearchPolicy[N: Node](dp.AbstractSearchPolicy[N]):
    fn: dp.AbstractSearchPolicy[N]

    def __call__[P, T](
        self,
        tree: "dp.Tree[N, P, T]",
        env: PolicyEnv,
        policy: P,
    ) -> dp.Stream[T]:
        return self.fn(tree, env, policy)


class _ParametricSearchPolicyFn[N: Node, **A](Protocol):
    def __call__[P, T](
        self,
        tree: dp.Tree[N, P, T],
        env: PolicyEnv,
        policy: P,
        *args: A.args,
        **kwargs: A.kwargs,
    ) -> dp.Stream[T]: ...


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
            tree: dp.Tree[N, Any, T], env: PolicyEnv, policy: Any
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
        self, query: dp.AttachedQuery[T], env: PolicyEnv
    ) -> dp.Stream[T]:
        return self.fn(query, env)


class _ParametricPromptingPolicyFn[**A](Protocol):
    def __call__[T](
        self,
        query: dp.AttachedQuery[T],
        env: PolicyEnv,
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
            query: dp.AttachedQuery[T], env: PolicyEnv
        ) -> dp.Stream[T]:
            return fn(query, env, *args, **kwargs)

        return PromptingPolicy(policy)

    return parametric


#####
##### Logging
#####


def log(
    env: PolicyEnv,
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


#####
##### Checking consistency of strategies and policies
#####


type _ParametricPolicy[**A, N: Node, P] = Callable[A, dp.Policy[N, P]]


def ensure_compatible[**A, N: Node, P](
    strategy: Callable[..., dp.StrategyComp[N, P, object]],
) -> Callable[[_ParametricPolicy[A, N, P]], _ParametricPolicy[A, N, P]]:
    """
    A decorator that does nothing but allows type-checkers to ensure
    that the decorated function returns a policy compatible with its
    strategy argument.

    TODO: this decorator does not seem to work with pyright.
    """
    return lambda f: f


#####
##### Inner Policy Dictionaries
#####


type DictIP = dict[str, Any]


def _dict_ip_key_match(key: str, tags: Sequence[dp.Tag]) -> bool:
    key_tags = key.split("&")
    return set(key_tags).issubset(set(tags))


def dict_subpolicy(ip: DictIP, tags: Sequence[dp.Tag]) -> Any:
    """
    Retrieve a sub-policy from a dictionary internal policy, using the
    tags of a particular space.
    """
    matches = [k for k in ip if _dict_ip_key_match(k, tags)]
    if not matches:
        raise ValueError(
            f"Missing sub-policy for space with tags {tags}.\n"
            + f"Provided keys are: {list(ip)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple sub-policies match tags {tags}.\n"
            + f"Matching keys are: {matches}\n"
            + f"Provided keys are: {list(ip)}"
        )
    return ip[matches[0]]


#####
##### Utilities
#####


def query_dependent(
    f: Callable[[dp.AbstractQuery[object]], PromptingPolicy],
) -> PromptingPolicy:
    def policy[T](query: dp.AttachedQuery[T], env: PolicyEnv) -> dp.Stream[T]:
        query_policy = f(query.query)
        yield from query_policy(query, env)

    return PromptingPolicy(policy)
