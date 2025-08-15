"""
Standard Nodes and Effects
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Never, cast

import delphyne.core as dp
from delphyne.stdlib.opaque import Opaque, OpaqueSpace

#####
##### Utilities
#####


def spawn_node[N: dp.Node](
    node_type: type[N], **args: Any
) -> dp.NodeBuilder[Any, Any]:
    """
    A convenience helper to write effect triggering functions.

    Attributes:
        node_type: The type of the node to spawn (e.g., `Branch`).
        args: Arguments to populate the node fields, passed to
            [`Node.spawn`][delphyne.core.trees.Node.spawn].
    """
    return dp.NodeBuilder(lambda spawner: node_type.spawn(spawner, **args))


type FromPolicy[T] = Callable[[Any], T]
"""
Type for an inner-policy-dependent data field.

We use `Any` instead of introducing an inner policy type parameter `P`,
since `Node` is not parametric either. Thus, this alias is mostly meant
for readability and expressing intent.
"""


class NodeMeta:
    """
    Abstract base class for node metadata.

    Nodes can feature fields with arbitrary metadata accessible to
    policies (e.g., `meta` field of `Branch`). Typing those fields with
    `NodeMeta` instead of `object` or `Any` allows for better type
    safety. In particular, it prevents errors that arise from
    accidentally passing uninstantiated parametric inner policy fields.
    """

    pass


#####
##### Branch Node
#####


@dataclass(frozen=True)
class Branch(dp.Node):
    cands: OpaqueSpace[Any, Any]
    extra_tags: Sequence[dp.Tag]
    meta: FromPolicy[NodeMeta] | None

    def navigate(self) -> dp.Navigation:
        return (yield self.cands)

    def primary_space(self) -> dp.Space[Any]:
        return self.cands


def branch[P, T](
    cands: Opaque[P, T],
    extra_tags: Sequence[dp.Tag] = (),
    meta: Callable[[P], NodeMeta] | None = None,
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Branch, P, T]:
    ret = yield spawn_node(
        Branch, cands=cands, extra_tags=extra_tags, meta=meta
    )
    return cast(T, ret)


#####
##### Fail Node
#####


@dataclass(frozen=True)
class Fail(dp.Node):
    error: dp.Error

    def valid_action(self, action: object) -> bool:
        return False

    def leaf_node(self) -> bool:
        return True

    def navigate(self) -> dp.Navigation:
        assert False
        yield

    def summary_message(self) -> str:
        return str(self.error)


def _build_error(
    message: str | None, label: str | None, error: dp.Error | None
) -> dp.Error:
    if not (label or message or error):
        label = "anon-failure-node"
    if error is None:
        error = dp.Error(label=label, description=message)
    else:
        assert message is None and label is None
    return error


def fail(
    label: str | None = None,
    *,
    message: str | None = None,
    error: dp.Error | None = None,
) -> dp.Strategy[Fail, object, Never]:
    yield spawn_node(Fail, error=_build_error(message, label, error))
    assert False


def ensure(
    prop: bool,
    label: str | None = None,
    *,
    message: str | None = None,
    error: dp.Error | None = None,
) -> dp.Strategy[Fail, object, None]:
    if not prop:
        yield from fail(label=label, message=message, error=error)


#####
##### Logging Node
#####


@dataclass
class Message(dp.Node):
    msg: str
    data: object | None

    def navigate(self) -> dp.Navigation:
        return None
        yield

    def summary_message(self) -> str:
        return self.msg


def message(
    msg: str, data: object | None = None
) -> dp.Strategy[Message, object, None]:
    yield spawn_node(Message, msg=msg, data=data)


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
    customization on the policy side. If no such function is given, the
    whole node is ignored.
    """

    eval: OpaqueSpace[Any, Any]
    factor: FromPolicy[Callable[[Any], float] | None]

    def navigate(self) -> dp.Navigation:
        return None
        yield

    def primary_space(self):
        return self.eval


def factor[E, P](
    eval: Opaque[P, E],
    factor: Callable[[P], Callable[[E], float] | None],
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

    eval: OpaqueSpace[Any, Any]
    value: FromPolicy[Callable[[Any], float] | None]

    def navigate(self) -> dp.Navigation:
        return None
        yield

    def primary_space(self):
        return self.eval


def value[E, P](
    eval: Opaque[P, E],
    value: Callable[[P], Callable[[E], float] | None],
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Value, P, None]:
    yield spawn_node(Value, eval=eval, value=value)


#####
##### Join Node
#####


@dataclass(frozen=True)
class Join(dp.Node):
    subs: Sequence[dp.EmbeddedTree[Any, Any, Any]]
    meta: FromPolicy[NodeMeta] | None

    def navigate(self) -> dp.Navigation:
        ret: list[Any] = []
        for sub in self.subs:
            ret.append((yield sub))
        return tuple(ret)


def join[N: dp.Node, P, T](
    subs: Sequence[dp.StrategyComp[N, P, T]],
    meta: Callable[[P], NodeMeta] | None = None,
) -> dp.Strategy[N, P, Sequence[T]]:
    ret = yield spawn_node(Join, subs=subs, meta=meta)
    return cast(Sequence[T], ret)
