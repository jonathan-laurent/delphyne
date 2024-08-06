"""
Concrete implementation of strategy trees.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from delphyne.core import refs, tracing, trees
from delphyne.core.inspect import underlying_strategy_return_type
from delphyne.core.refs import ChoiceOutcomeRef, ChoiceRef, NodeId
from delphyne.core.tracing import Outcome, Tracer, Value
from delphyne.core.trees import Node, StrategyComp, Success, Tree
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


type _Path = Sequence[object]


@dataclass
class StrategyException(Exception):
    """
    Raised when a strategy encounters an internal error (e.g. a failed
    assertion or an index-out-of-bounds error).
    """

    exn: Exception


@dataclass
class StrategyTree[N: Node, T](Tree[N, T]):
    """
    Concrete implementation of the tree interface.
    """

    _tracer: Tracer
    _node_id: NodeId
    _origin: ChoiceRef
    _node: N | Success[T]
    _path: _Path
    _strategy: StrategyComp[N, T]
    _cur_generator: trees.Strategy[N, T] | None  # optimization

    @property
    def node(self) -> N | Success[T]:
        return self._node

    @property
    def node_id(self) -> NodeId:
        return self._node_id

    @property
    def tracer(self) -> Tracer:
        return self._tracer

    def return_type(self) -> TypeAnnot[T] | NoTypeInfo:
        return _strategy_return_type(self._strategy)

    def __repr__(self) -> str:
        return f"StrategyTree({self._node.__repr__()})"

    @staticmethod
    def new[M: Node, U](
        strategy: StrategyComp[M, U]
    ) -> "StrategyTree[M, U]":  # fmt: skip
        tracer = Tracer()
        return StrategyTree.new_from_tracer(strategy, tracer, tracer.ROOT_ID)

    @staticmethod
    def new_from_tracer[M: Node, U](
        strategy: StrategyComp[M, U], tracer: Tracer, node_id: NodeId
    ) -> "StrategyTree[M, U]":  # fmt: skip
        origin = _tree_origin(tracer, node_id)
        return _make_tree(strategy, tracer, node_id, origin)

    def child(self, action: Value) -> "StrategyTree[N, T]":
        action_raw = tracing.drop_refs(action)
        action_ref = tracing.value_ref(action)
        del action
        assert self.node.valid_action(action_raw), (
            "Invalid action for node of type "
            + f"{type(self.node)}: {action_raw}."
        )
        # If `child` has never been called before, the generator that
        # yielded the current node is still valid and can be reused. For
        # subsequent calls, `self._cur_generator` is None and the
        # strategy is resimulated from scratch following `self._path`.
        path = [*self._path, action_raw]
        if self._cur_generator is not None:
            pre_node = _send_action(self._cur_generator, action_raw)
            gen = self._cur_generator
        else:
            pre_node, gen = _recompute(self._strategy, path)
        self._cur_generator = None  # Invalidate the generator
        # Generate an ID for the child
        rel = refs.ChildOf(self._node_id, action_ref)
        id = self._tracer.fresh_or_cached_node_id(rel)
        ret_type = self.return_type()
        node = _finalize_presuccess(pre_node, id, self._origin, ret_type)
        return StrategyTree(
            self._tracer,
            id,
            self._origin,
            node,
            path,
            self._strategy,
            gen,
        )

    def spawn[L: Node, U](
        self, strategy: StrategyComp[L, U], origin: ChoiceRef
    ) -> "StrategyTree[L, U]":  # fmt: skip
        rel = refs.SubtreeOf(self._node_id, origin)
        id = self._tracer.fresh_or_cached_node_id(rel)
        return _make_tree(strategy, self._tracer, id, origin)


@dataclass
class _PreSuccess[T]:
    value: T


def _strategy_return_type[N: Node, T](
    strategy: StrategyComp[N, T]
) -> TypeAnnot[T] | NoTypeInfo:  # fmt: skip
    return underlying_strategy_return_type(strategy)


def _send_action[N: Node, T](
    gen: trees.Strategy[N, T], action: object
) ->  N | _PreSuccess[T]:  # fmt: skip
    try:
        return gen.send(action)
    except StopIteration as e:
        v = cast(T, e.value)
        return _PreSuccess(v)
    except Exception as e:
        raise StrategyException(e)


def _recompute[N: Node, T](
    strategy: StrategyComp[N, T],
    path: _Path,
) -> tuple[N | _PreSuccess[T], trees.Strategy[N, T]]:  # fmt: skip
    gen = strategy()
    mknode = _send_action(gen, None)
    for a in path:
        mknode = _send_action(gen, a)
    return mknode, gen


def _finalize_presuccess[N: Node, T](
    node: N | _PreSuccess[T],
    id: NodeId,
    origin: ChoiceRef,
    type: TypeAnnot[T] | NoTypeInfo
) -> N | Success[T]:  # fmt: skip
    if not isinstance(node, _PreSuccess):
        return node
    ref = ChoiceOutcomeRef(origin, id)
    return Success(Outcome(node.value, ref, type))


def _make_tree[N: Node, T](
    strategy: StrategyComp[N, T],
    tracer: Tracer,
    id: NodeId,
    origin: ChoiceRef,
) -> "StrategyTree[N, T]":  # fmt: skip
    pre_node, gen = _recompute(strategy, ())
    type = _strategy_return_type(strategy)
    node = _finalize_presuccess(pre_node, id, origin, type)
    return StrategyTree[N, T](tracer, id, origin, node, (), strategy, gen)


def _tree_origin(tracer: Tracer, node_id: NodeId) -> ChoiceRef:
    if node_id == tracer.ROOT_ID:
        return tracer.ROOT_ORIGIN
    match tracer.nodes[node_id]:
        case refs.ChildOf(parent, _):
            return _tree_origin(tracer, parent)
        case refs.SubtreeOf(_, origin):
            return origin
