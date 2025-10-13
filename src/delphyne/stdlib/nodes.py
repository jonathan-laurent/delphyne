"""
Standard Nodes and Effects
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Literal,
    Never,
    NoReturn,
    TypeVar,
    cast,
    overload,
    override,
)

import delphyne.core as dp
import delphyne.stdlib.policies as pol
from delphyne.stdlib.environments import PolicyEnv
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
##### Typed space references
#####


T_inv = TypeVar("T_inv", contravariant=False, covariant=False)


@dataclass(frozen=True)
class TypedSpaceRef(Generic[T_inv]):
    """
    A wrapper around global space references that carries phantom type
    information.

    See `branch` for an example of usage.
    """

    ref: dp.refs.GlobalSpacePath


#####
##### Branch Node
#####


@dataclass(frozen=True)
class Branch(dp.Node):
    """
    The standard `Branch` effect.

    Can be triggered using the `branch` function, which allows branching
    over elements of an opaque space.
    """

    cands: OpaqueSpace[Any, Any]
    meta: FromPolicy[NodeMeta] | None

    @override
    def navigate(self) -> dp.Navigation:
        return (yield self.cands)

    @override
    def primary_space(self) -> dp.Space[Any]:
        return self.cands


@overload
def branch[P, T](
    cands: Opaque[P, T],
    *,
    meta: Callable[[P], NodeMeta] | None = None,
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Branch, P, T]: ...


@overload
def branch[P, T](
    cands: Opaque[P, T],
    *,
    meta: Callable[[P], NodeMeta] | None = None,
    return_space_ref: Literal[True],
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Branch, P, tuple[T, TypedSpaceRef[T]]]: ...


def branch[P, T](
    cands: Opaque[P, T],
    *,
    meta: Callable[[P], NodeMeta] | None = None,
    return_space_ref: bool = False,
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Branch, P, T | tuple[T, TypedSpaceRef[T]]]:
    """
    Branch over the elements of an opaque space.

    Arguments:
        cands: An opaque space, which can be defined from either a query
            or a strategy via the `using` method.
        meta: An optional mapping from the ambient inner policy to
            arbitrary metadata accessible to search policies.
        inner_policy_type: Ambient inner policy type. This information
            is not used at runtime but it can be provided to help type
            inference when necessary.
        return_space_ref: Whether to return a typed reference to the
            space of candidates along with the chosen element.
    """
    ret, nref = yield spawn_node(Branch, cands=cands, meta=meta)
    ret = cast(T, ret)
    if not return_space_ref:
        return ret
    lsref = dp.refs.SpaceRef(dp.refs.SpaceName("cands", ()), ())
    return ret, TypedSpaceRef[T](nref.nested_space(lsref))


#####
##### Fail Node
#####


@dataclass(frozen=True)
class Fail(dp.Node):
    """
    The standard `Fail` effect.

    Can be triggered using the `fail` and `ensure` functions.
    """

    error: dp.Error

    @override
    def leaf_node(self) -> bool:
        return True

    @override
    def navigate(self) -> dp.Navigation:
        assert False
        yield

    @override
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
) -> dp.Strategy[Fail, object, NoReturn]:
    """
    Fail immediately with an error.

    The error can be specified using the `error` keyword argument. As a
    shortcut, the `label` and `message` arguments can also be used to
    directly specify the corresponding fields of the `Error` type. Those
    arguments can only be used if `error` is not provided.

    !!! warning
        Like all effect triggering functions, this function must be
        invoked as:

            yield from fail(...)

        Forgetting `yield from` may not result in a type error but will
        result in a no-op at runtime.
    """
    yield spawn_node(Fail, error=_build_error(message, label, error))
    assert False


def ensure(
    prop: bool,
    label: str | None = None,
    *,
    message: str | None = None,
    error: dp.Error | None = None,
) -> dp.Strategy[Fail, object, None]:
    """
    Ensure that a property holds, otherwise fail with an error.

    See `fail` regarding the `label`, `message` and `error` arguments.

    !!! warning
        Like all effect triggering functions, this function must be
        invoked as:

            yield from ensure(...)

        Forgetting `yield from` may not result in a type error but will
        result in a no-op at runtime.
    """
    if not prop:
        yield from fail(label=label, message=message, error=error)


#####
##### Logging Node
#####


@dataclass
class Message(dp.Node):
    """
    The standard `Message` effect.

    Message nodes are tree nodes carrying a message. They have a unique
    child. They can be eliminated using the `elim_messages` tree
    transformer, which automatically logs their content.

    This effect is useful for debugging strategies. Using `print`
    statements in strategies is discouraged since strategy computations
    are replayed every time a child of the associated tree is computed,
    causing the same message to be repeatedly printed.
    """

    msg: str
    data: object | None
    level: dp.LogLevel | None

    @override
    def navigate(self) -> dp.Navigation:
        return None
        yield

    @override
    def summary_message(self) -> str:
        return self.msg


def message(
    msg: str, data: object | None = None, level: dp.LogLevel | None = None
) -> dp.Strategy[Message, object, None]:
    """
    Log a debugging message. See `Message` for more details.

    Arguments:
        msg: The message to log.
        data: Optional data to attach to the message.
        level: Optional severity level of the message.

    !!! warning
        Like all effect triggering functions, this function must be
        invoked as:

            yield from message(...)

        Forgetting `yield from` may not result in a type error but will
        result in a no-op at runtime.
    """
    yield spawn_node(Message, msg=msg, data=data, level=level)


@pol.contextual_tree_transformer
def elim_messages(
    env: PolicyEnv,
    policy: Any,
    show_in_log: bool = True,
    default_log_level: dp.LogLevel = "info",
) -> pol.PureTreeTransformerFn[Message, Never]:
    """
    Eliminate the `Message` effect.

    Arguments:
        show_in_log: Whether to log the associated content whenever a
            message node is encountered.
        default_log_level: The default severity level to use when no
            severity level is specified in the message node.
    """

    def transform[N: dp.Node, P, T](
        tree: dp.Tree[Message | N, P, T],
    ) -> dp.Tree[N, P, T]:
        if isinstance(tree.node, Message):
            if show_in_log:
                metadata = {"attached": tree.node.data}
                level = tree.node.level or default_log_level
                env.log(level, tree.node.msg, metadata=metadata, loc=tree)
            return transform(tree.child(None))
        return tree.transform(tree.node, transform)

    return transform


#####
##### Factor Node
#####


@dataclass(frozen=True)
class Factor(dp.Node):
    """
    The standard `Factor` effect.

    A `Factor` node allows computing a confidence score in the [0, 1]
    interval. This confidence can be computed by a query or a dedicated
    strategy but only one element will be generated from the resulting
    space. Instead of having an oracle compute a numerical value
    directly, it computes an evaluation object that is then transformed
    into a number using a policy-provided function. This allows greater
    flexibility on the policy side. If no such function is given, the
    whole node is ignored.
    """

    eval: OpaqueSpace[Any, Any]
    factor: FromPolicy[Callable[[Any], float] | None]

    @override
    def navigate(self) -> dp.Navigation:
        return None
        yield

    @override
    def primary_space(self):
        return self.eval


def factor[E, P](
    eval: Opaque[P, E],
    factor: Callable[[P], Callable[[E], float] | None],
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Factor, P, None]:
    """
    Apply a multiplicative penalty to the current search branch.

    Arguments:
        eval: An opaque space, typically induced by a strategy or a
            query whose purpose is to evaluate the current search state,
            returning an evaluation object of type `E`. Policies
            typically extract a single element from this space.
        factor: An inner-policy-dependent function that maps an
            evaluation object of type `E` to an actual numerical
            penalty, as a multiplicative factor in the [0, 1] interval.
            If no function is provided, the whole node is ignored.
            Separating `eval` and `factor` allows greater flexibility
            for policies to tune how multidimensional state evaluations
            are reduced into a single numbers.

    !!! warning
        Like all effect triggering functions, this function must be
        invoked as:

            yield from factor(...)

        Forgetting `yield from` may not result in a type error but will
        result in a no-op at runtime.
    """
    yield spawn_node(Factor, eval=eval, factor=factor)


#####
##### Value Node
#####


@dataclass(frozen=True)
class Value(dp.Node):
    """
    The standard `Value` effect.

    Similar to `Factor`, except that the resulting number is used to set
    the whole value of the branch instead of just multiplying it.
    """

    eval: OpaqueSpace[Any, Any]
    value: FromPolicy[Callable[[Any], float] | None]

    @override
    def navigate(self) -> dp.Navigation:
        return None
        yield

    @override
    def primary_space(self):
        return self.eval


def value[E, P](
    eval: Opaque[P, E],
    value: Callable[[P], Callable[[E], float] | None],
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Value, P, None]:
    """
    Set the value of the current search branch.

    See the similar `factor` function for more details.

    !!! warning
        Like all effect triggering functions, this function must be
        invoked as:

            yield from message(...)

        Forgetting `yield from` may not result in a type error but will
        result in a no-op at runtime.
    """
    yield spawn_node(Value, eval=eval, value=value)


@pol.contextual_tree_transformer
def elim_values(
    env: PolicyEnv,
    policy: Any,
) -> pol.PureTreeTransformerFn[Value, Never]:
    """
    Eliminate the `Value` effect.
    """

    def transform[N: dp.Node, P, T](
        tree: dp.Tree[Value | N, P, T],
    ) -> dp.Tree[N, P, T]:
        if isinstance(tree.node, Value):
            return transform(tree.child(None))
        return tree.transform(tree.node, transform)

    return transform


@pol.contextual_tree_transformer
def binarize_values(
    env: PolicyEnv,
    policy: Any,
    *,
    threshold: float = 0.5,
) -> pol.PureTreeTransformerFn[Value, Branch | Fail]:
    """
    Turn value nodes into assertions based on a threshold.

    Attributes:
        threshold: The threshold above which a value is considered
            acceptable.

    !!! warning:
        `Value` nodes are transformed into `Branch` nodes and so it is
        important that the associated opaque spaces for computing
        strategies only generate one candidate so that no actual
        branching can happen on value estimation.
    """

    def transform[N: dp.Node, P, T](
        tree: dp.Tree[Value | N, P, T],
    ) -> dp.Tree[Branch | Fail | N, P, T]:
        if isinstance(tree.node, Value):
            node = tree.node

            def branch_child(
                eval: dp.Value,
            ) -> dp.Tree[Branch | Fail | N, P, T]:
                value_comp = node.value(policy)
                if (
                    value_comp is None
                    or value_comp(dp.drop_refs(eval)) >= threshold
                ):
                    return transform(tree.child(None))
                else:
                    return dp.Tree[Branch | Fail | N, P, T](
                        Fail(dp.Error()),
                        # What we pass for `child` does not matter since
                        # it will never be accessed.
                        child=(lambda v: transform(tree.child(None))),
                        ref=tree.ref,
                    )

            return dp.Tree[Branch | Fail | N, P, T](
                Branch(node.eval, meta=None), branch_child, tree.ref
            )
        return tree.transform(tree.node, transform)

    return transform


#####
##### Join Node
#####


@dataclass(frozen=True)
class Join(dp.Node):
    """
    The standard `Join` effect.

    This effect can be triggered using the `join` function. A `Join`
    node features a sequence of embedded trees.
    """

    subs: Sequence[dp.EmbeddedTree[Any, Any, Any]]
    meta: FromPolicy[NodeMeta] | None

    @override
    def navigate(self) -> dp.Navigation:
        ret: list[Any] = []
        for sub in self.subs:
            ret.append((yield sub))
        return tuple(ret)


def join[N: dp.Node, P, T](
    subs: Sequence[dp.StrategyComp[N, P, T]],
    meta: Callable[[P], NodeMeta] | None = None,
) -> dp.Strategy[N, P, Sequence[T]]:
    """
    Evaluate a sequence of independent strategy computations, possibly
    in parallel.

    Arguments:
        subs: A sequence of strategy computations to evaluate.
        meta: An optional mapping from the ambient inner policy to
            arbitrary metadata accessible to search policies.

    Returns:
        A sequence featuring all computation results.
    """
    ret, _ = yield spawn_node(Join, subs=subs, meta=meta)
    return cast(Sequence[T], ret)
