from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import delphyne.core as dp
from delphyne.stdlib.nodes import Failure, fail, spawn_node
from delphyne.stdlib.policies import search_policy
from delphyne.stdlib.strategies import strategy
from delphyne.stdlib.streams import StreamTransformer


@dataclass
class Iteration(dp.Node):
    next: Callable[
        [dp.Tracked[Any] | None],
        dp.OpaqueSpace[Any, tuple[Any | None, Any]],
    ]

    def navigate(self) -> dp.Navigation:
        return (yield self.next(None))


@strategy(name="iterate")
def _iterate[P, S, T](
    next: Callable[[S | None], dp.Opaque[P, tuple[T | None, S]]],
) -> dp.Strategy[Iteration | Failure, P, T]:
    ret = yield spawn_node(Iteration, next=next)
    ret = cast(tuple[T | None, S], ret)
    yielded, _new_state = ret
    if yielded is None:
        yield from fail(label="no_element_yielded")
    else:
        return yielded


@search_policy
def search_iteration[P, T](
    tree: dp.Tree[Iteration | Failure, P, T],
    env: dp.PolicyEnv,
    policy: P,
) -> dp.Stream[T]:
    assert isinstance(tree.node, Iteration)
    state: dp.Tracked[Any] | None = None
    while True:
        for msg in tree.node.next(state).stream(env, policy):
            if isinstance(msg, dp.Yield):
                # TODO: here, `msg` contains the value we are interested
                # in so it is tempting to just yield it. However, this
                # is not allowed since the attached reference would not
                # properly point to a success node. In our Haskell
                # implementation, such a bug would be caught by the type
                # system. Here, we should at least add some proper
                # dynamic checks to ensure it does not happen.
                yielded_and_new_state = msg.value
                state = yielded_and_new_state[1]
                child = tree.child(yielded_and_new_state)
                assert not isinstance(child.node, Iteration)
                if isinstance(child.node, dp.Success):
                    yield dp.Yield(child.node.success)
                break
            else:
                yield msg


def iterate[P, S, T](
    next: Callable[[S | None], dp.Opaque[P, tuple[T | None, S]]],
    transform_stream: Callable[[P], StreamTransformer | None] | None = None,
) -> dp.Opaque[P, T]:
    def iterate_policy(inner_policy: P):
        policy = search_iteration()
        if transform_stream is not None:
            trans = transform_stream(inner_policy)
            if trans is not None:
                policy = trans @ policy
        return (policy, inner_policy)

    return _iterate(next).using(iterate_policy)
