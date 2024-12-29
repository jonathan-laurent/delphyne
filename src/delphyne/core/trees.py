"""
The core tree datastructure.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from delphyne.core import inspect, refs
from delphyne.core.node_fields import NodeFields, detect_node_structure
from delphyne.core.queries import AbstractQuery, ParseError
from delphyne.core.refs import Assembly, GlobalNodePath, ValueRef
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


#####
##### Tracked values
#####


@dataclass
class Tracked[T]:
    """
    A tracked value, which associates a value with a reference.

    The `node` field does not appear in Orakell and is a global path
    (from the global origin) to the node the value is attached too.
    Having this field is useful to check at runtime that a tracked value
    passed as an argument to `child` is attached to the current node.
    """

    value: T
    ref: ValueRef
    node: GlobalNodePath
    type_annot: TypeAnnot[T] | NoTypeInfo


type Value = Assembly[Tracked[Any]]
"""
A dynamic assembly of tracked values.
"""


def value_ref(v: Value) -> refs.ValueRef:
    match v:
        case Tracked(_, ref):
            return ref
        case tuple():
            return tuple(value_ref(o) for o in v)


def drop_refs(v: Value) -> object:
    match v:
        case Tracked(value):
            return value
        case tuple():
            return tuple(drop_refs(o) for o in v)


#####
##### Spaces
#####


class Space[T](ABC):
    """
    Abstract type for a space.
    """

    @abstractmethod
    def tags(self) -> "Sequence[Tag]":
        pass

    @abstractmethod
    def source(self) -> "StrategyComp[Any, T] | AttachedQuery[T]":
        pass


@dataclass(frozen=True)
class AttachedQuery[T]:
    """
    Wrapper for a query attached to a specific space.
    """

    query: AbstractQuery[Any, T]
    ref: refs.GlobalSpaceRef
    answer: Callable[[refs.AnswerModeName, str], Tracked[T] | ParseError]


type Tag = str
"""
Tags for better navigation in traces and in the demo language.
"""


#####
##### Nodes
#####


class Node(ABC):
    """
    Abstract type for a node.
    """

    def extra_tags(self) -> Sequence[Tag]:
        return []

    def tags(self) -> Sequence[Tag]:
        # TODO
        return self.extra_tags()

    def effect_name(self) -> str:
        return self.__class__.__name__

    def fields(self) -> NodeFields:
        cls = self.__class__
        f = detect_node_structure(cls, embedded_class=int, space_class=Space)
        if f is None:
            msg = f"Impossible to autodetect the structure of {cls}"
            assert False, msg
        return f

    def valid_action(self, action: object) -> bool:
        """
        This method can be implemented optionally to validate actions.
        """
        return True

    def leaf_node(self) -> bool:
        return False

    @abstractmethod
    def navigate(self) -> "Navigation":
        """
        See `Navigation` for more details.
        """
        pass

    def primary_choice(self) -> "Space[object] | None":
        """
        Choice that is inferred by default in a choice outcome reference
        (see `SpaceElementRef`). For example, `compare([cands{''},
        cands{'foo bar'}])` can be abbreviated into `compare(['', 'foo
        bar'])` if `cands` is the primary choice for the current node.
        """
        return None

    def base_choices(self) -> "Iterable[Space[object]]":
        """
        Choices that are always displayed in the UI, even if they do not
        appear in the trace. By default, the primary choice is treated
        as the only base choice (when specified).
        """
        if (choice := self.primary_choice()) is not None:
            return (choice,)
        return ()

    def summary_message(self) -> str | None:
        return None

    @classmethod
    def spawn[T](cls: type[T], **kwargs: Any) -> T:
        assert False


type Navigation = Generator[Space[Any], Tracked[Any], Value]
"""
Nodes have a `navigate` method that yields choices and expects
corresponding outcomes until it outputs an action.
"""


####
#### Strategy Type
####


type Strategy[N: Node, P, T] = Generator[NodeBuilder[N], object, T]


type NodeBuilder[N: Node] = Callable[[_GeneralSpawner], N]


@dataclass
class StrategyComp[N: Node, T]:
    """
    A strategy computation also stores metadata for navigation and
    debugging.
    """

    comp: Callable[..., Strategy[N, Any, T]]
    args: list[Any]
    kwargs: dict[str, Any]

    def __call__(self) -> Strategy[N, Any, T]:
        return self.comp(*self.args, **self.kwargs)

    def strategy_name(self) -> str | None:
        return inspect.function_name(self.comp)

    def tags(self) -> Sequence[Tag]:
        return [name] if (name := self.strategy_name()) is not None else []

    def return_type(self) -> TypeAnnot[T] | NoTypeInfo:
        ret_type = inspect.function_return_type(self.comp)
        if isinstance(ret_type, NoTypeInfo):
            return NoTypeInfo()
        return inspect.return_type_of_strategy_type(ret_type)


#####
##### Embedded Trees
#####


@dataclass(frozen=True)
class EmbeddedTree[T](Space[T]):
    _comp: StrategyComp[Any, T]
    spawn_tree: "Callable[[], Tree[Any, T]]"

    def source(self) -> StrategyComp[Any, T]:
        return self._comp

    def tags(self) -> Sequence[Tag]:
        return self._comp.tags()

    @staticmethod
    def builder(comp: StrategyComp[Any, T]) -> "Builder[EmbeddedTree[T]]":
        return lambda spawn, _: EmbeddedTree(comp, lambda: spawn(comp))


#####
##### Builders and Spawners
#####


class _TreeSpawner(Protocol):
    def __call__[N: Node, T](
        self, strategy: "StrategyComp[N, T]"
    ) -> "Tree[N, T]": ...


class _QuerySpawner(Protocol):
    def __call__[T](
        self, query: AbstractQuery[Any, T]
    ) -> "AttachedQuery[T]": ...


type Builder[S] = Callable[[_TreeSpawner, _QuerySpawner], S]


@dataclass(frozen=True)
class _GeneralSpawner:
    """
    Allows spawning arbitrary parametric subspaces at a given node.
    """

    _ref: "_NonEmptyGlobalPath"
    _node_hook: "Callable[[Tree[Any, Any]], None] | None"

    def parametric[S](
        self,
        parametric_builder: Callable[..., Builder[S]],
    ) -> Callable[..., S]:
        assert False

    def parametric_tree[N: Node, T](
        self,
        space_name: refs.SpaceName,
        parametric_strategy: Callable[..., StrategyComp[N, T]],
        args: tuple[Tracked[Any], ...],
    ) -> "Tree[N, T]":
        args_raw = [drop_refs(arg) for arg in args]
        args_ref = tuple(value_ref(arg) for arg in args)
        strategy = parametric_strategy(*args_raw)
        gr = global_path_from_nonempty(self._ref)
        sr = refs.SpaceRef(space_name, args_ref)
        return _reify(strategy, (gr, sr, ()), self._node_hook)

    def parametric_query[T](
        self,
        space_name: refs.SpaceName,
        parametric_query: Callable[..., AbstractQuery[Any, T]],
        args: tuple[Tracked[Any], ...],
    ) -> "AttachedQuery[T]":
        args_raw = [drop_refs(arg) for arg in args]
        args_ref = tuple(value_ref(arg) for arg in args)
        _query = parametric_query(*args_raw)
        _gr = global_path_from_nonempty(self._ref)
        _sr = refs.SpaceRef(space_name, args_ref)
        # TODO: return attached query
        assert False

    def tree[N: Node, T](
        self,
        space_name: refs.SpaceName,
        strategy: StrategyComp[N, T],
    ) -> "Tree[N, T]":
        return self.parametric_tree(space_name, lambda: strategy, ())


#####
##### Tree Type
#####


@dataclass(frozen=True)
class Tree[N: Node, T]:
    node: "N | Success[T]"
    child: "Callable[[Value], Tree[N, T]]"
    ref: GlobalNodePath


@dataclass(frozen=True)
class Success[T](Node):
    success: Tracked[T]

    def leaf_node(self) -> bool:
        return True

    def valid_action(self, action: object) -> bool:
        return False

    def navigate(self) -> Navigation:
        assert False


@dataclass
class StrategyException(Exception):
    """
    Raised when a strategy encounters an internal error (e.g. a failed
    assertion or an index-out-of-bounds error).
    """

    exn: Exception


#####
##### Reifying Strategies into Trees
#####


type _NonEmptyGlobalPath = tuple[GlobalNodePath, refs.SpaceRef, refs.NodePath]
"""
Encodes a non-empty global node path.

More precisely, `(gr, sr, nr)` encodes `(*gr, (sr, nr))`
"""


def global_path_from_nonempty(ref: _NonEmptyGlobalPath) -> GlobalNodePath:
    (gr, sr, nr) = ref
    return (*gr, (sr, nr))


def _reify[N: Node, T](
    strategy: StrategyComp[N, T],
    root_ref: _NonEmptyGlobalPath | None,
    node_hook: Callable[[Tree[Any, Any]], None] | None,
) -> Tree[N, T]:
    def aux(
        strategy: StrategyComp[N, T],
        actions: Sequence[object],
        ref: _NonEmptyGlobalPath,
        node: N | Success[T],
        cur_generator: Strategy[N, Any, T],
    ) -> Tree[N, T]:
        generator_valid = True

        def child(action: Value) -> Tree[N, T]:
            action_raw = drop_refs(action)
            action_ref = value_ref(action)
            del action
            assert node.valid_action(action_raw), (
                "Invalid action for node of type "
                + f"{type(node)}: {action_raw}."
            )
            # If `child` has never been called before, the generator
            # that yielded the current node is still valid and can be
            # reused. For subsequent calls, `self._cur_generator` is
            # None and the strategy is resimulated from scratch
            # following `self._path`.
            nonlocal generator_valid
            new_actions = (*actions, action_raw)
            if generator_valid:
                pre_node = _send_action(cur_generator, action_raw)
                new_gen = cur_generator
                generator_valid = False
                assert False
            else:
                pre_node, new_gen = _recompute(strategy, new_actions)
            gr, cr, nr = ref
            new_ref = (gr, cr, (*nr, action_ref))
            ret_type = strategy.return_type()
            new_node = _finalize_node(pre_node, new_ref, ret_type, node_hook)
            return aux(strategy, new_actions, new_ref, new_node, new_gen)
            assert False

        tree = Tree(node, child, global_path_from_nonempty(ref))
        if node_hook is not None:
            node_hook(tree)
        return tree

    if root_ref is None:
        root_ref = ((), refs.MAIN_SPACE, ())
    pre_node, gen = _recompute(strategy, ())
    ret_type = strategy.return_type()
    node = _finalize_node(pre_node, root_ref, ret_type, node_hook)
    return aux(strategy, (), root_ref, node, gen)


@dataclass
class _PreSuccess[T]:
    value: T


type _PreNode[N: Node, T] = NodeBuilder[N] | _PreSuccess[T]


def _send_action[N: Node, T](
    gen: Strategy[N, Any, T], action: object
) -> _PreNode[N, T]:
    try:
        return gen.send(action)
    except StopIteration as e:
        v = cast(T, e.value)
        return _PreSuccess(v)
    except Exception as e:
        raise StrategyException(e)


def _recompute[N: Node, T](
    strategy: StrategyComp[N, T],
    actions: Sequence[object],
) -> tuple[_PreNode[N, T], Strategy[N, Any, T]]:
    gen = strategy()
    mknode = _send_action(gen, None)
    for a in actions:
        mknode = _send_action(gen, a)
    return mknode, gen


def _finalize_node[N: Node, T](
    pre_node: _PreNode[N, T],
    ref: _NonEmptyGlobalPath,
    type: TypeAnnot[T] | NoTypeInfo,
    node_hook: Callable[[Tree[Any, Any]], None] | None,
) -> N | Success[T]:
    if not isinstance(pre_node, _PreSuccess):
        return pre_node(_GeneralSpawner(ref, node_hook))
    (gr, sr, nr) = ref
    ser = refs.SpaceElementRef(sr, nr)
    return Success(Tracked(pre_node.value, ser, gr, type))
