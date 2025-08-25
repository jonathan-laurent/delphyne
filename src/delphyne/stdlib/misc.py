"""
Miscellaneous utilities for Delphyne's standard library.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Never, cast, overload

import delphyne.core_and_base as dp
from delphyne.core_and_base import Opaque
from delphyne.stdlib.computations import Compute, elim_compute
from delphyne.stdlib.environments import PolicyEnv
from delphyne.stdlib.flags import Flag, FlagQuery, elim_flag, get_flag
from delphyne.stdlib.nodes import Branch, Fail, branch
from delphyne.stdlib.policies import Policy, PromptingPolicy, SearchPolicy
from delphyne.stdlib.search.dfs import dfs
from delphyne.stdlib.strategies import strategy
from delphyne.stdlib.streams import Stream, stream_or_else


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
    return const_strategy(value).using(just_dfs)


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


def just_dfs[P](policy: P) -> Policy[Branch | Fail, P]:
    """
    Convenience shortcut to avoid passing lambdas to the `get_policy`
    argument of `using`, when using DFS in combination with the ambient
    inner policy.
    """
    return dfs() & policy


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
    sub-policy (or as a sub- prompting policy).
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
        ).gen()

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
        ).gen()

    return SearchPolicy(policy)


def sequence_policies[N: dp.Node, P](
    policies: Iterable[Policy[N, P]], *, stop_on_reject: bool = True
) -> Policy[N, P]:
    @dp.search_policy
    def search[T](
        tree: dp.Tree[N, Any, T], env: PolicyEnv, policy: Any
    ) -> dp.StreamGen[T]:
        assert policy is None
        yield from Stream.sequence(
            (p.search(tree, env, p.inner) for p in policies),
            stop_on_reject=stop_on_reject,
        ).gen()

    return search() & cast(P, None)


@overload
def sequence(
    policies: Iterable[PromptingPolicy],
    *,
    stop_on_reject: bool = True,
) -> PromptingPolicy:
    pass


@overload
def sequence[N: dp.Node](
    policies: Iterable[SearchPolicy[N]],
    *,
    stop_on_reject: bool = True,
) -> SearchPolicy[N]:
    pass


@overload
def sequence[N: dp.Node, P](
    policies: Iterable[Policy[N, P]],
    *,
    stop_on_reject: bool = True,
) -> Policy[N, P]:
    pass


def sequence(
    policies: Iterable[_AnyPolicy],
    *,
    stop_on_reject: bool = True,
) -> _AnyPolicy:
    """
    Try a list of policies, search policies, or prompting policies in
    sequence.

    Attributes:
    - policies: An iterable of policies, search policies, or prompting
          policies to try in sequence.
    - stop_on_reject: If True, stop the sequence as soon as one policy
          sees all its resource requests denied. Note that this is
          necessary for termination when `policies` is an infinite
          iterator.
    """

    it = iter(policies)
    first = next(it)
    if isinstance(first, PromptingPolicy):
        return sequence_prompting_policies(
            cast(Iterable[PromptingPolicy], policies),
            stop_on_reject=stop_on_reject,
        )
    elif isinstance(first, SearchPolicy):
        return sequence_search_policies(
            cast(Iterable[SearchPolicy[Any]], policies),
            stop_on_reject=stop_on_reject,
        )
    else:
        assert isinstance(first, Policy)
        return sequence_policies(
            cast(Iterable[Policy[Any, Any]], policies),
            stop_on_reject=stop_on_reject,
        )


#####
##### Or-Else
#####


def prompting_policy_or_else(
    main: PromptingPolicy, other: PromptingPolicy
) -> PromptingPolicy:
    def policy[T](
        query: dp.AttachedQuery[T], env: PolicyEnv
    ) -> dp.StreamGen[T]:
        yield from stream_or_else(
            lambda: main(query, env).gen(),
            lambda: other(query, env).gen(),
        )

    return PromptingPolicy(policy)


def search_policy_or_else[N: dp.Node](
    main: SearchPolicy[N], other: SearchPolicy[N]
) -> SearchPolicy[N]:
    def policy[T](
        tree: dp.Tree[N, Any, T], env: PolicyEnv, policy: Any
    ) -> dp.StreamGen[T]:
        yield from stream_or_else(
            lambda: main(tree, env, policy).gen(),
            lambda: other(tree, env, policy).gen(),
        )

    return SearchPolicy(policy)


def policy_or_else[N: dp.Node, P](
    main: Policy[N, P], other: Policy[N, P]
) -> Policy[N, P]:
    # TODO: this is not the cleanest implementation since we lie about
    # the return type by returning a dummy internal policy.
    @dp.search_policy
    def sp[T](
        tree: dp.Tree[N, Any, T], env: PolicyEnv, policy: Any
    ) -> dp.StreamGen[T]:
        assert policy is None
        yield from stream_or_else(
            lambda: main.search(tree, env, main.inner).gen(),
            lambda: other.search(tree, env, other.inner).gen(),
        )

    return sp() & cast(P, None)


@overload
def or_else(main: PromptingPolicy, other: PromptingPolicy) -> PromptingPolicy:
    pass


@overload
def or_else[N: dp.Node](
    main: SearchPolicy[N], other: SearchPolicy[N]
) -> SearchPolicy[N]:
    pass


@overload
def or_else[N: dp.Node, P](
    main: Policy[N, P], other: Policy[N, P]
) -> Policy[N, P]:
    pass


def or_else(main: _AnyPolicy, other: _AnyPolicy) -> _AnyPolicy:
    """
    Take two policies, search policies, or prompting policies as
    arguments. Try the first one, and then the second one only if it
    fails (i.e., it does not produce any solution).
    """
    if isinstance(main, PromptingPolicy):
        assert isinstance(other, PromptingPolicy)
        return prompting_policy_or_else(main, other)
    elif isinstance(main, SearchPolicy):
        assert isinstance(other, SearchPolicy)
        return search_policy_or_else(main, other)
    else:
        return policy_or_else(main, other)  # type: ignore


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
def nofail_strategy[P, T](
    space: Opaque[P, T], *, default: T
) -> dp.Strategy[Flag[NoFailFlag] | Branch, P, T]:
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
    search_policy = or_else(try_policy, def_policy)
    return nofail_strategy(space, default=default).using(
        lambda p: search_policy & p
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
