from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Never, cast, overload

import delphyne.core as dp
from delphyne.stdlib.computations import Compute, elim_compute
from delphyne.stdlib.flags import Flag, FlagQuery, elim_flag, get_flag
from delphyne.stdlib.nodes import Branch, Fail, Message, branch
from delphyne.stdlib.opaque import Opaque
from delphyne.stdlib.policies import (
    Policy,
    PromptingPolicy,
    PureTreeTransformerFn,
    SearchPolicy,
    contextual_tree_transformer,
    search_policy,
)
from delphyne.stdlib.search.dfs import dfs
from delphyne.stdlib.strategies import strategy
from delphyne.stdlib.streams import stream_or_else


@strategy(name="const")
def const_strategy[T](value: T) -> dp.Strategy[Never, object, T]:
    return value
    yield


def const_space[T](value: T) -> Opaque[object, T]:
    return const_strategy(value).using(just_dfs)


@strategy(name="map")
def map_space_strategy[P, A, B](
    space: Opaque[P, A], f: Callable[[A], B]
) -> dp.Strategy[Branch, P, B]:
    res = yield from branch(space)
    return f(res)


def map_space[P, A, B](
    space: Opaque[P, A], f: Callable[[A], B]
) -> Opaque[P, B]:
    return map_space_strategy(space, f).using(lambda p: dfs() & p)


def just_dfs[P](policy: P) -> Policy[Branch | Fail, P]:
    return dfs() & policy


def just_compute[P](policy: P) -> Policy[Compute, P]:
    return dfs() @ elim_compute() & policy


def ambient_pp(policy: PromptingPolicy) -> PromptingPolicy:
    return policy


def ambient_policy[N: dp.Node, P](policy: Policy[N, P]) -> Policy[N, P]:
    return policy


def ambient[F](policy: F) -> F:
    return policy


type _AnyPolicy = PromptingPolicy | SearchPolicy[Any] | Policy[Any, Any]


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
            yield from pp(query, env).gen()

    return PromptingPolicy(policy)


def sequence_search_policies[N: dp.Node](
    policies: Iterable[SearchPolicy[N]],
) -> SearchPolicy[N]:
    def policy[T](
        tree: dp.Tree[N, Any, T], env: dp.PolicyEnv, policy: Any
    ) -> dp.Stream[T]:
        for sp in policies:
            yield from sp(tree, env, policy).gen()

    return SearchPolicy(policy)


def sequence_policies[N: dp.Node, P](
    policies: Iterable[Policy[N, P]],
) -> Policy[N, P]:
    # TODO: this is not the cleanest implementation since we lie about
    # the return type by returning a dummy internal policy.
    @search_policy
    def search[T](
        tree: dp.Tree[N, Any, T], env: dp.PolicyEnv, policy: Any
    ) -> dp.Stream[T]:
        assert policy is None
        for p in policies:
            yield from p.search(tree, env, p.inner).gen()

    return search() & cast(P, None)


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
    policies: Iterable[Policy[N, P]],
) -> Policy[N, P]:
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
        return sequence_policies(cast(Iterable[Policy[Any, Any]], policies))


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
            lambda: main(query, env).gen(),
            lambda: other(query, env).gen(),
        )

    return PromptingPolicy(policy)


def search_policy_or_else[N: dp.Node](
    main: SearchPolicy[N], other: SearchPolicy[N]
) -> SearchPolicy[N]:
    def policy[T](
        tree: dp.Tree[N, Any, T], env: dp.PolicyEnv, policy: Any
    ) -> dp.Stream[T]:
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
    @search_policy
    def sp[T](
        tree: dp.Tree[N, Any, T], env: dp.PolicyEnv, policy: Any
    ) -> dp.Stream[T]:
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


@contextual_tree_transformer
def elim_messages(
    env: dp.PolicyEnv,
    policy: Any,
    show_in_log: bool = True,
) -> PureTreeTransformerFn[Message, Never]:
    def transform[N: dp.Node, P, T](
        tree: dp.Tree[Message | N, P, T],
    ) -> dp.Tree[N, P, T]:
        if isinstance(tree.node, Message):
            if show_in_log:
                metadata = {"attached": tree.node.data}
                env.tracer.log(tree.node.msg, metadata=metadata)
            return transform(tree.child(None))
        return tree.transform(tree.node, transform)

    return transform


#####
##### Preventing Failures with Defaults
#####


type NoFailFlagValue = Literal["no_fail_try", "no_fail_default"]


@dataclass
class NoFailFlag(FlagQuery[NoFailFlagValue]):
    pass


@strategy(name="nofail")
def nofail_strategy[P, T](
    space: Opaque[P, T], *, default: T
) -> dp.Strategy[Flag[NoFailFlag] | Branch, P, T]:
    flag = yield from get_flag(NoFailFlag)
    match flag:
        case "no_fail_try":
            return (yield from branch(space))
        case "no_fail_default":
            return default


def nofail[P, T](space: Opaque[P, T], *, default: T) -> Opaque[P, T]:
    try_policy = dfs() @ elim_flag(NoFailFlag, "no_fail_try")
    def_policy = dfs() @ elim_flag(NoFailFlag, "no_fail_default")
    search_policy = or_else(try_policy, def_policy)
    return nofail_strategy(space, default=default).using(
        lambda p: search_policy & p
    )


def _failing_pp[T](
    query: dp.AttachedQuery[T], env: dp.PolicyEnv
) -> dp.Stream[T]:
    return
    yield


failing_pp = PromptingPolicy(_failing_pp)
