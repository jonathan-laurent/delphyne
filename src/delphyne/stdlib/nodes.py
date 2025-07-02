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
    meta: FromPolicy[object] | None

    def navigate(self) -> dp.Navigation:
        return (yield self.cands)

    def primary_space(self) -> dp.Space[Any]:
        return self.cands


def branch[P, T](
    cands: dp.Opaque[P, T],
    extra_tags: Sequence[dp.Tag] = (),
    meta: Callable[[P], object] | None = None,
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Branch, P, T]:
    ret = yield spawn_node(
        Branch, cands=cands, extra_tags=extra_tags, meta=meta
    )
    return cast(T, ret)


#####
##### Failure Node
#####


@dataclass(frozen=True)
class Failure(dp.Node):
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
) -> dp.Strategy[Failure, object, Never]:
    yield spawn_node(Failure, error=_build_error(message, label, error))
    assert False


def ensure(
    prop: bool,
    label: str | None = None,
    *,
    message: str | None = None,
    error: dp.Error | None = None,
) -> dp.Strategy[Failure, object, None]:
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

    eval: dp.OpaqueSpace[Any, Any]
    factor: FromPolicy[Callable[[Any], float] | None]

    def navigate(self) -> dp.Navigation:
        return None
        yield

    def primary_space(self):
        return self.eval


def factor[E, P](
    eval: dp.Opaque[P, E],
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

    eval: dp.OpaqueSpace[Any, Any]
    value: FromPolicy[Callable[[Any], float] | None]

    def navigate(self) -> dp.Navigation:
        return None
        yield

    def primary_space(self):
        return self.eval


def value[E, P](
    eval: dp.Opaque[P, E],
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
    meta: FromPolicy[object] | None

    def navigate(self) -> dp.Navigation:
        ret: list[Any] = []
        for sub in self.subs:
            ret.append((yield sub))
        return tuple(ret)


def join[N: dp.Node, P, T](
    subs: Sequence[dp.StrategyComp[N, P, T]],
    meta: Callable[[P], object] | None = None,
) -> dp.Strategy[N, P, Sequence[T]]:
    ret = yield spawn_node(Join, subs=subs, meta=meta)
    return cast(Sequence[T], ret)
