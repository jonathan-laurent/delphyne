"""
Miscellaneous utilities for Delphyne's standard library.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Never, cast

import delphyne.core_and_base as dp
from delphyne.core_and_base import Opaque
from delphyne.stdlib.computations import Compute, elim_compute
from delphyne.stdlib.environments import PolicyEnv
from delphyne.stdlib.flags import Flag, FlagQuery, elim_flag, get_flag
from delphyne.stdlib.nodes import Branch, branch
from delphyne.stdlib.policies import Policy, PromptingPolicy, SearchPolicy
from delphyne.stdlib.search.dfs import dfs
from delphyne.stdlib.strategies import strategy
from delphyne.stdlib.streams import Stream


@strategy(name="const")
def const_strategy[T](value: T) -> dp.Strategy[Never, object, T]:
    """
    A constant strategy computation that always returns the same value
    and performs no effect.
    """
    return value
    yield


def const_space[T](value: T) -> Opaque[object, T]:
    """
    Build an opaque space containing a single constant value.
    """
    return const_strategy(value).using(lambda p: dfs() & p)


@strategy(name="map")
def map_space_strategy[P, A, B](
    space: Opaque[P, A], f: Callable[[A], B]
) -> dp.Strategy[Branch, P, B]:
    """
    A strategy that selects an element from an opaque space before
    applying a function to it and returning the result.
    """
    res = yield from branch(space)
    return f(res)


def map_space[P, A, B](
    space: Opaque[P, A], f: Callable[[A], B]
) -> Opaque[P, B]:
    """
    Obtain an opaque space from another one by applying a pure function
    to all its elements.
    """
    return map_space_strategy(space, f).using(lambda p: dfs() & p)


def just_compute[P](policy: P) -> Policy[Compute, P]:
    """
    Convenience shortcut to avoid passing lambdas to the `get_policy`
    argument of `using`, in the case of sub-strategies that only feature
    the `Compute` effect.
    """
    return dfs() @ elim_compute() & policy


def ambient_pp(policy: PromptingPolicy) -> PromptingPolicy:
    """
    Convenience shortcut to avoid passing lambdas to the `get_policy`
    argument of `Query.using`, when using the ambient inner policy as a
    prompting policy.
    """
    return policy


def ambient[F](policy: F) -> F:
    """
    Convenience shortcut to avoid passing lambdas to the `get_policy`
    argument of `Query.using`, when using the ambient inner policy as a
    sub-policy (or as a sub-prompting policy).
    """
    return policy


type _AnyPolicy = PromptingPolicy | SearchPolicy[Any] | Policy[Any, Any]


#####
##### Sequencing
#####


def sequence_prompting_policies(
    policies: Iterable[PromptingPolicy], *, stop_on_reject: bool = True
) -> PromptingPolicy:
    def policy[T](
        query: dp.AttachedQuery[T], env: PolicyEnv
    ) -> dp.StreamGen[T]:
        yield from Stream.sequence(
            (pp(query, env) for pp in policies), stop_on_reject=stop_on_reject
        )

    return PromptingPolicy(policy)


def sequence_search_policies[N: dp.Node](
    policies: Iterable[SearchPolicy[N]], *, stop_on_reject: bool = True
) -> SearchPolicy[N]:
    def policy[T](
        tree: dp.Tree[N, Any, T], env: PolicyEnv, policy: Any
    ) -> dp.StreamGen[T]:
        yield from Stream.sequence(
            (sp(tree, env, policy) for sp in policies),
            stop_on_reject=stop_on_reject,
        )

    return SearchPolicy(policy)


def sequence_policies[N: dp.Node, P](
    policies: Iterable[Policy[N, P]], *, stop_on_reject: bool = True
) -> Policy[N, P]:
    def aux[T](tree: dp.Tree[N, Any, T], env: PolicyEnv) -> dp.StreamGen[T]:
        yield from Stream.sequence(
            (p(tree, env) for p in policies),
            stop_on_reject=stop_on_reject,
        )

    return Policy(aux)


#####
##### Parallel combination
#####


def parallel_prompting_policies(
    policies: Sequence[PromptingPolicy],
) -> PromptingPolicy:
    def policy[T](
        query: dp.AttachedQuery[T], env: PolicyEnv
    ) -> dp.StreamGen[T]:
        yield from Stream.parallel([pp(query, env) for pp in policies])

    return PromptingPolicy(policy)


def parallel_search_policies[N: dp.Node](
    policies: Sequence[SearchPolicy[N]],
) -> SearchPolicy[N]:
    def policy[T](
        tree: dp.Tree[N, Any, T], env: PolicyEnv, policy: Any
    ) -> dp.StreamGen[T]:
        yield from Stream.parallel([sp(tree, env, policy) for sp in policies])

    return SearchPolicy(policy)


def parallel_policies[N: dp.Node, P](
    policies: Sequence[Policy[N, P]],
) -> Policy[N, P]:
    @dp.search_policy
    def search[T](
        tree: dp.Tree[N, Any, T], env: PolicyEnv, policy: Any
    ) -> dp.StreamGen[T]:
        assert policy is None
        yield from Stream.parallel([p(tree, env) for p in policies])

    return search() & cast(P, None)


#####
##### Preventing Failures with Defaults
#####


type NoFailFlagValue = Literal["no_fail_try", "no_fail_default"]
"""Possible values for the `NoFailFlag` flag."""


@dataclass
class NoFailFlag(FlagQuery[NoFailFlagValue]):
    """Flag used by the `nofail` space transformer."""

    pass


@strategy(name="nofail")
def nofail_strategy[P, A, B](
    space: Opaque[P, A], *, default: B
) -> dp.Strategy[Flag[NoFailFlag] | Branch, P, A | B]:
    """
    Strategy underlying the `nofail` space transformer.
    """
    flag = yield from get_flag(NoFailFlag)
    match flag:
        case "no_fail_try":
            return (yield from branch(space))
        case "no_fail_default":
            return default


def nofail[P, A, B](space: Opaque[P, A], *, default: B) -> Opaque[P, A | B]:
    """
    Modify an opaque space to that branching over it can never fail.

    If the stream associated with the opaque space gets exhausted and no
    solution is produced, the provided default value is used.

    In demonstrations, the default value can be selected by using the
    `#no_fail_default` hint.
    """
    try_policy = dfs() @ elim_flag(NoFailFlag, "no_fail_try")
    def_policy = dfs() @ elim_flag(NoFailFlag, "no_fail_default")
    return nofail_strategy(space, default=default).using(
        lambda p: try_policy.or_else(def_policy) & p
    )


#####
##### Prompting Policies
#####


def _failing_pp[T](
    query: dp.AttachedQuery[T], env: PolicyEnv
) -> dp.StreamGen[T]:
    return
    yield


failing_pp = PromptingPolicy(_failing_pp)
"""
A prompting policy that always fails.
"""
