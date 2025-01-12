"""
Standard Nodes
"""

from collections.abc import Sequence
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
    cands: dp.Builder[dp.OpaqueSpace[P, T]],
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


def message(msg: str) -> dp.Strategy[Message, object, None]:
    yield spawn_node(Message, msg=msg)
