from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Never, cast, overload

import delphyne.core as dp
from delphyne.stdlib.computations import Computation, elim_compute
from delphyne.stdlib.flags import Flag, FlagQuery, elim_flag, get_flag
from delphyne.stdlib.nodes import Branch, Failure, Message, branch
from delphyne.stdlib.policies import (
    ContextualTreeTransformer,
    PromptingPolicy,
    SearchPolicy,
)
from delphyne.stdlib.search.dfs import dfs
from delphyne.stdlib.strategies import strategy
from delphyne.stdlib.streams import stream_or_else


@strategy(name="const")
def const_strategy[T](value: T) -> dp.Strategy[Never, object, T]:
    return value
    yield


def const_space[T](value: T) -> dp.OpaqueSpaceBuilder[object, T]:
    return const_strategy(value)(object, lambda _: (dfs(), None))


@strategy(name="map")
def map_space_strategy[P, A, B](
    space: dp.OpaqueSpaceBuilder[P, A], f: Callable[[A], B]
) -> dp.Strategy[Branch, P, B]:
    res = yield from branch(space)
    return f(res)


def map_space[P, A, B](
    space: dp.OpaqueSpaceBuilder[P, A], f: Callable[[A], B]
) -> dp.OpaqueSpaceBuilder[P, B]:
    return map_space_strategy(space, f).using(lambda p: (dfs(), p))


def just_dfs[P](policy: P) -> dp.Policy[Branch | Failure, P]:
    return (dfs(), policy)


def just_compute[P](policy: P) -> dp.Policy[Computation, P]:
    return (dfs() @ elim_compute, policy)


def ambient_pp(policy: PromptingPolicy) -> PromptingPolicy:
    return policy


def ambient_policy[N: dp.Node, P](policy: dp.Policy[N, P]) -> dp.Policy[N, P]:
    return policy


def ambient[F](policy: F) -> F:
    return policy


type _AnyPolicy = PromptingPolicy | SearchPolicy[Any] | dp.Policy[Any, Any]


#####
##### Sequencing
#####

# TODO: this is way too much boilerplate for a relatively simple feature


def sequence_prompting_policies(
    policies: Iterable[PromptingPolicy],
) -> PromptingPolicy:
    def policy[T](
        query: dp.AttachedQuery[T], env: dp.PolicyEnv
    ) -> dp.Stream[T]:
        for pp in policies:
            yield from pp(query, env)

    return PromptingPolicy(policy)


def sequence_search_policies[N: dp.Node](
    policies: Iterable[SearchPolicy[N]],
) -> SearchPolicy[N]:
    def policy[T](
        tree: dp.Tree[N, Any, T], env: dp.PolicyEnv, policy: Any
    ) -> dp.Stream[T]:
        for sp in policies:
            yield from sp(tree, env, policy)

    return SearchPolicy(policy)


def sequence_policies[N: dp.Node, P](
    policies: Iterable[dp.Policy[N, P]],
) -> dp.Policy[N, P]:
    # TODO: this is not the cleanest implementation since we lie about
    # the return type by returning a dummy internal policy.
    def search_policy[T](
        tree: dp.Tree[N, Any, T], env: dp.PolicyEnv, policy: Any
    ) -> dp.Stream[T]:
        assert policy is None
        for sp, ip in policies:
            yield from sp(tree, env, ip)

    return (search_policy, cast(P, None))


@overload
def sequence(
    policies: Iterable[PromptingPolicy],
) -> PromptingPolicy:
    pass


@overload
def sequence[N: dp.Node](
    policies: Iterable[SearchPolicy[N]],
) -> SearchPolicy[N]:
    pass


@overload
def sequence[N: dp.Node, P](
    policies: Iterable[dp.Policy[N, P]],
) -> dp.Policy[N, P]:
    pass


def sequence(
    policies: Iterable[_AnyPolicy],
) -> _AnyPolicy:
    it = iter(policies)
    first = next(it)
    if isinstance(first, PromptingPolicy):
        return sequence_prompting_policies(
            cast(Iterable[PromptingPolicy], policies)
        )
    elif isinstance(first, SearchPolicy):
        return sequence_search_policies(
            cast(Iterable[SearchPolicy[Any]], policies)
        )
    else:
        return sequence_policies(cast(Iterable[dp.Policy[Any, Any]], policies))


#####
##### Or-Else
#####


def prompting_policy_or_else(
    main: PromptingPolicy, other: PromptingPolicy
) -> PromptingPolicy:
    def policy[T](
        query: dp.AttachedQuery[T], env: dp.PolicyEnv
    ) -> dp.Stream[T]:
        yield from stream_or_else(
            lambda: main(query, env), lambda: other(query, env)
        )

    return PromptingPolicy(policy)


def search_policy_or_else[N: dp.Node](
    main: SearchPolicy[N], other: SearchPolicy[N]
) -> SearchPolicy[N]:
    def policy[T](
        tree: dp.Tree[N, Any, T], env: dp.PolicyEnv, policy: Any
    ) -> dp.Stream[T]:
        yield from stream_or_else(
            lambda: main(tree, env, policy), lambda: other(tree, env, policy)
        )

    return SearchPolicy(policy)


def policy_or_else[N: dp.Node, P](
    main: dp.Policy[N, P], other: dp.Policy[N, P]
) -> dp.Policy[N, P]:
    # TODO: this is not the cleanest implementation since we lie about
    # the return type by returning a dummy internal policy.
    def search_policy[T](
        tree: dp.Tree[N, Any, T], env: dp.PolicyEnv, policy: Any
    ) -> dp.Stream[T]:
        assert policy is None
        main_sp, main_ip = main
        other_sp, other_ip = other
        yield from stream_or_else(
            lambda: main_sp(tree, env, main_ip),
            lambda: other_sp(tree, env, other_ip),
        )

    return (search_policy, cast(P, None))


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
    main: dp.Policy[N, P], other: dp.Policy[N, P]
) -> dp.Policy[N, P]:
    pass


def or_else(main: _AnyPolicy, other: _AnyPolicy) -> _AnyPolicy:
    if isinstance(main, PromptingPolicy):
        assert isinstance(other, PromptingPolicy)
        return prompting_policy_or_else(main, other)
    elif isinstance(main, SearchPolicy):
        assert isinstance(other, SearchPolicy)
        return search_policy_or_else(main, other)
    else:
        return policy_or_else(main, other)  # type: ignore


#####
##### Eliminate message nodes
#####


def elim_messages(
    show_in_log: bool = True,
) -> ContextualTreeTransformer[Message, Never]:
    def transform[N: dp.Node, P, T](
        tree: dp.Tree[Message | N, P, T], env: dp.PolicyEnv, policy: P
    ) -> dp.Tree[N, P, T]:
        if isinstance(tree.node, Message):
            if show_in_log:
                metadata = {"attached": tree.node.data}
                env.tracer.log(tree.node.msg, metadata=metadata)
            return transform(tree.child(None), env, policy)
        return tree.transform(tree.node, lambda n: transform(n, env, policy))  # type: ignore

    return ContextualTreeTransformer(transform)


#####
##### Preventing Failures with Defaults
#####


type NoFailFlagValue = Literal["no_fail_try", "no_fail_default"]


@dataclass
class NoFailFlag(FlagQuery[NoFailFlagValue]):
    pass


@strategy(name="nofail")
def nofail_strategy[P, T](
    space: dp.OpaqueSpaceBuilder[P, T], *, default: T
) -> dp.Strategy[Flag[NoFailFlag] | Branch, P, T]:
    flag = yield from get_flag(NoFailFlag)
    match flag:
        case "no_fail_try":
            yield from branch(space)
        case "no_fail_default":
            return default


def nofail[P, T](
    space: dp.OpaqueSpaceBuilder[P, T], *, default: T
) -> dp.OpaqueSpaceBuilder[P, T]:
    try_policy = dfs() @ elim_flag(NoFailFlag, "no_fail_try")
    def_policy = dfs() @ elim_flag(NoFailFlag, "no_fail_default")
    search_policy = or_else(try_policy, def_policy)
    return nofail_strategy(space, default=default).using(
        lambda p: (search_policy, p)
    )


def _failing_pp[T](
    query: dp.AttachedQuery[T], env: dp.PolicyEnv
) -> dp.Stream[T]:
    return
    yield


failing_pp = PromptingPolicy(_failing_pp)
