"""
Standard Nodes
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Never, cast

import delphyne.core as dp

#####
##### Utilities
#####


def spawn_node[N: dp.Node](
    node_type: type[N], **args: Any
) -> dp.NodeBuilder[Any, Any]:
    return dp.NodeBuilder(lambda spawner: node_type.spawn(spawner, **args))


type FromPolicy[T] = Callable[[Any], T]
"""
For readability, represents a value parametric in the surrounding
policy.
"""


#####
##### Branch Node
#####


@dataclass(frozen=True)
class Branch(dp.Node):
    cands: dp.OpaqueSpace[Any, Any]
    extra_tags: Sequence[dp.Tag]

    def navigate(self) -> dp.Navigation:
        return (yield self.cands)

    def primary_space(self) -> dp.Space[Any]:
        return self.cands


def branch[P, T](
    cands: dp.OpaqueSpaceBuilder[P, T],
    extra_tags: Sequence[dp.Tag] = (),
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Branch, P, T]:
    ret = yield spawn_node(Branch, cands=cands, extra_tags=extra_tags)
    return cast(T, ret)


#####
##### Failure Node
#####


@dataclass(frozen=True)
class Failure(dp.Node):
    message: str

    def valid_action(self, action: object) -> bool:
        return False

    def leaf_node(self) -> bool:
        return True

    def navigate(self) -> dp.Navigation:
        assert False
        yield

    def summary_message(self) -> str:
        return self.message


def fail(message: str) -> dp.Strategy[Failure, object, Never]:
    yield spawn_node(Failure, message=message)
    assert False


def ensure(
    prop: bool, message: str = ""
) -> dp.Strategy[Failure, object, None]:
    if not prop:
        yield from fail(message)


#####
##### Logging Node
#####


@dataclass
class Message(dp.Node):
    msg: str

    def navigate(self) -> dp.Navigation:
        return None
        yield

    def summary_message(self) -> str:
        return self.msg


def message(msg: str) -> dp.Strategy[Message, object, None]:
    yield spawn_node(Message, msg=msg)


#####
##### Factor Node
#####


@dataclass(frozen=True)
class Factor(dp.Node):
    """
    A node that allows computing a confidence score in the [0, 1]
    interval. This confidence can be computed by a query or a dedicated
    strategy but only one element will be generated from the resulting
    space. Instead of having an oracle compute a numerical value
    directly, it computes an evaluation object that is then transformed
    into a number using a policy-provided function. This allows greater
    customization on the policy side.
    """

    eval: dp.OpaqueSpace[Any, Any]
    factor: FromPolicy[Callable[[Any], float]]

    def navigate(self) -> dp.Navigation:
        return ()
        yield

    def primary_space(self):
        return self.eval


def factor[E, P](
    eval: dp.OpaqueSpaceBuilder[P, E],
    factor: Callable[[P], Callable[[E], float]],
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Factor, P, None]:
    yield spawn_node(Factor, eval=eval, factor=factor)


#####
##### Value Node
#####


@dataclass(frozen=True)
class Value(dp.Node):
    """
    Similar to `Factor`, except that the resulting number is used to set
    the whole value of the branch instead of just multiplying it.
    """

    eval: dp.OpaqueSpace[Any, Any]
    value: FromPolicy[Callable[[Any], float]]

    def navigate(self) -> dp.Navigation:
        return ()
        yield

    def primary_space(self):
        return self.eval


def value[E, P](
    eval: dp.OpaqueSpaceBuilder[P, E],
    value: Callable[[P], Callable[[E], float]],
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Value, P, None]:
    yield spawn_node(Value, eval=eval, value=value)
