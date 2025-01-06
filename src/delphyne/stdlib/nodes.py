"""
Standard Nodes
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Never, cast

from delphyne.core import trees as tr
from delphyne.core.trees import Navigation, Node, OpaqueSpace, Space, Tag

#####
##### Utilities
#####


def spawn_node[N: Node](
    node_type: type[N], **args: Any
) -> tr.NodeBuilder[Any, Any]:
    return tr.NodeBuilder(lambda spawner: node_type.spawn(spawner, **args))


#####
##### Branch Node
#####


@dataclass(frozen=True)
class Branch(Node):
    cands: OpaqueSpace[Any, Any]
    extra_tags: Sequence[Tag]

    def navigate(self) -> Navigation:
        return (yield self.cands)

    def primary_space(self) -> Space[Any]:
        return self.cands


def branch[P, T](
    cands: tr.Builder[OpaqueSpace[P, T]],
    extra_tags: Sequence[Tag] = (),
    inner_policy_type: type[P] | None = None,
) -> tr.Strategy[Branch, P, T]:
    ret = yield spawn_node(Branch, cands=cands, extra_tags=extra_tags)
    return cast(T, ret)


#####
##### Failure Node
#####


@dataclass(frozen=True)
class Failure(Node):
    message: str

    def valid_action(self, action: object) -> bool:
        return False

    def leaf_node(self) -> bool:
        return True

    def navigate(self) -> Navigation:
        assert False
        yield

    def summary_message(self) -> str:
        return self.message


def fail(message: str) -> tr.Strategy[Failure, object, Never]:
    yield spawn_node(Failure, message=message)
    assert False


def ensure(
    prop: bool, message: str = ""
) -> tr.Strategy[Failure, object, None]:
    if not prop:
        yield from fail(message)
