from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import delphyne as dp
from delphyne.stdlib.nodes import spawn_node
from delphyne.stdlib.policies import search_policy
from delphyne.stdlib.strategies import strategy


@dataclass
class Iterated(dp.Node):
    next: Callable[[tuple[dp.Tracked[Any], ...]], dp.OpaqueSpace[Any, Any]]

    def navigate(self) -> dp.Navigation:
        return (yield self.next(()))


@strategy(name="iterated")
def _iterated[P, T](
    next: Callable[[tuple[T, ...]], dp.Builder[dp.OpaqueSpace[P, T]]],
) -> dp.Strategy[Iterated, P, T]:
    ret = yield spawn_node(Iterated, next=next)
    return cast(Any, ret)


@search_policy
async def search_iterated[P, T](
    tree: dp.Tree[Iterated, P, T],
    env: dp.PolicyEnv,
    policy: P,
) -> dp.Stream[T]:
    assert isinstance(tree.node, Iterated)
    blacklist: tuple[dp.Tracked[T], ...] = ()
    while True:
        async for msg in tree.node.next(blacklist).stream(env, policy):
            if isinstance(msg, dp.Yield):
                # TODO: here, `msg` contains the value we are interested
                # in so it is tempting to just yield it. However, this
                # is not allowed since the attached reference would not
                # properly point to a success node. In our Haskell
                # implementation, such a bug would be caught by the type
                # system. Here, we should at least add some proper
                # dynamic checks to ensure it does not happen.
                child = tree.child(msg.value)
                assert isinstance(child.node, dp.Success)
                yield dp.Yield(child.node.success)
                blacklist += (msg.value,)
                break
            else:
                yield msg


def iterated[P, T](
    next: Callable[[tuple[T, ...]], dp.Builder[dp.OpaqueSpace[P, T]]],
) -> dp.Builder[dp.OpaqueSpace[P, T]]:
    return _iterated(next).using(lambda p: (search_iterated(), p))
