"""
Iteratively calling a strategy or query.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import delphyne.core as dp
from delphyne.stdlib.environments import PolicyEnv
from delphyne.stdlib.nodes import Fail, fail, spawn_node
from delphyne.stdlib.opaque import Opaque, OpaqueSpace
from delphyne.stdlib.policies import search_policy
from delphyne.stdlib.strategies import strategy
from delphyne.stdlib.streams import StreamTransformer


@dataclass
class Iteration(dp.Node):
    """
    Node type for the singleton trees induced by `iterate`.
    """

    next: Callable[
        [dp.Tracked[Any] | None],
        OpaqueSpace[Any, tuple[Any | None, Any]],
    ]

    def navigate(self) -> dp.Navigation:
        return (yield self.next(None))


def _iterate_inherited_tags(next: Any) -> Sequence[dp.SpaceBuilder[Any]]:
    return [next(None)]


@strategy(name="iterate", inherit_tags=_iterate_inherited_tags)
def _iterate[P, S, T](
    next: Callable[[S | None], Opaque[P, tuple[T | None, S]]],
) -> dp.Strategy[Iteration | Fail, P, T]:
    ret = yield spawn_node(Iteration, next=next)
    ret = cast(tuple[T | None, S], ret)
    yielded, _new_state = ret
    if yielded is None:
        yield from fail(label="no_element_yielded")
    else:
        return yielded


@search_policy
def _search_iteration[P, T](
    tree: dp.Tree[Iteration | Fail, P, T],
    env: PolicyEnv,
    policy: P,
) -> dp.StreamGen[T]:
    assert isinstance(tree.node, Iteration)
    state: dp.Tracked[Any] | None = None
    while True:
        for msg in tree.node.next(state).stream(env, policy).gen():
            if isinstance(msg, dp.Solution):
                # Here, `msg` contains the value we are interested
                # in so it is tempting to just yield it. However, this
                # is not allowed since the attached reference would not
                # properly point to a success node. In our Haskell
                # implementation, such a bug would be caught by the type
                # system. Here, we catch it dynamically.
                yielded_and_new_state = msg.tracked
                state = yielded_and_new_state[1]
                child = tree.child(yielded_and_new_state)
                assert not isinstance(child.node, Iteration)
                if isinstance(child.node, dp.Success):
                    yield dp.Solution(child.node.success)
                break
            else:
                yield msg


def iterate[P, S, T](
    next: Callable[[S | None], Opaque[P, tuple[T | None, S]]],
    transform_stream: Callable[[P], StreamTransformer | None] | None = None,
) -> Opaque[P, T]:
    """
    Iteratively call a strategy or query, repeatedly feeding back the
    last call's output state into a new call and yielding values along
    the way.

    A standard use case is to repeatedly call a query or strategy with a
    blacklist of previously generated values, so as to produce diverse
    success values.

    Arguments:
        next: A parametric opaque space, induced by a query or stratey
            that takes a state as an input (or `None` initially) and
            outputs a new state, along with a generated value.
        transform_stream: An optional mapping from the inner policy to a
            stream transformer to be applied to the resulting stream of
            generated values.

    Returns:
        An opaque space enumerating all generated values.
    """

    def iterate_policy(inner_policy: P):
        policy = _search_iteration()
        if transform_stream is not None:
            trans = transform_stream(inner_policy)
            if trans is not None:
                policy = trans @ policy
        return policy & inner_policy

    return _iterate(next).using(iterate_policy)
