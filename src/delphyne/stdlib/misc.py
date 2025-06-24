from collections.abc import Callable
from typing import Never

import delphyne.core as dp
from delphyne.stdlib.nodes import Branch, branch
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
