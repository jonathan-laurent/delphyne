from collections.abc import Callable
from typing import Never

import delphyne.core as dp
from delphyne.stdlib.computations import Computation, elim_compute
from delphyne.stdlib.nodes import Branch, Failure, branch
from delphyne.stdlib.policies import PromptingPolicy
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
