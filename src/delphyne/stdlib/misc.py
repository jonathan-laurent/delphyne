from collections.abc import Callable
from typing import Any, Iterable, Never, cast, overload

import delphyne.core as dp
from delphyne.stdlib.computations import Computation, elim_compute
from delphyne.stdlib.nodes import Branch, Failure, branch
from delphyne.stdlib.policies import PromptingPolicy, SearchPolicy
from delphyne.stdlib.search.dfs import dfs
from delphyne.stdlib.strategies import strategy


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
    return map_space_strategy(space, f)(P, lambda p: (dfs(), p))  # type: ignore


def just_dfs[P](policy: P) -> dp.Policy[Branch | Failure, P]:
    return (dfs(), policy)


def just_compute[P](policy: P) -> dp.Policy[Computation, P]:
    return (dfs() @ elim_compute, policy)


def ambient_pp(policy: PromptingPolicy) -> PromptingPolicy:
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
