"""
The core tree datastructure.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

from delphyne.core import environment as en
from delphyne.core import inspect, refs
from delphyne.core import node_fields as nf
from delphyne.core.environment import PolicyEnv
from delphyne.core.node_fields import NodeFields, detect_node_structure
from delphyne.core.queries import AbstractQuery, ParseError
from delphyne.core.refs import GlobalNodePath, SpaceName, Tracked, Value
from delphyne.core.streams import Stream
from delphyne.utils.typing import NoTypeInfo, TypeAnnot

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
    def source(self) -> "NestedTree[Any, Any, T] | AttachedQuery[T]":
        pass


@dataclass(frozen=True)
class AttachedQuery[T]:
    """
    Wrapper for a query attached to a specific space.
    """

    query: AbstractQuery[T]
    ref: refs.GlobalSpaceRef
    answer: Callable[[refs.AnswerModeName, str], Tracked[T] | ParseError]

    def tags(self) -> "Sequence[Tag]":
        return self.query.tags()


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

    # Methods that **must** be overriden

    @abstractmethod
    def navigate(self) -> "Navigation":
        """
        See `Navigation` for more details.
        """
        pass

    # Methods that are _sometimes_ overriden

    def summary_message(self) -> str | None:
        return None

    def leaf_node(self) -> bool:
        return False

    def valid_action(self, action: object) -> bool:
        """
        This method can be implemented optionally to validate actions.
        """
        return True

    # Method indicating the nature of the node's fields, which is
    # typically inferred but can be manually specified when heuristics
    # fail.

    @classmethod
    def fields(cls) -> NodeFields:
        f = detect_node_structure(
            cls, embedded_class=EmbeddedTree, space_class=Space
        )
        if f is None:
            msg = f"Impossible to autodetect the structure of {cls}"
            assert False, msg
        return f

    # Methods with a sensible default behavior that are _rarely_ overriden

    def effect_name(self) -> str:
        return self.__class__.__name__

    def primary_space_name(self) -> refs.SpaceName | None:
        """
        Space that is inferred by default in a space element reference
        (see `SpaceElementRef`). For example, `compare([cands{''},
        cands{'foo bar'}])` can be abbreviated into `compare(['', 'foo
        bar'])` if `cands` is the primary space for the current node.
        """
        if hasattr(self, "__primary__"):
            name = getattr(self, "__primary__")
            if name is None:
                return None
            assert isinstance(name, str)
            return refs.SpaceName(name, ())
        return None

    def get_extra_tags(self) -> Sequence[Tag]:
        if hasattr(self, "extra_tags"):
            return getattr(self, "extra_tags")
        return []

    def get_tags(self) -> Sequence[Tag]:
        if (primary := self.primary_space()) is not None:
            return [*primary.tags(), *self.get_extra_tags()]
        return self.get_extra_tags()

    def base_spaces(self) -> "Iterable[Space[object]]":
        """
        Spaces that are always displayed in the UI, even if they do not
        appear in the trace. By default, the primary space is treated
        as the only base space (when specified).
        """
        if (space := self.primary_space()) is not None:
            return (space,)
        return ()

    # Methods that should not be overriden

    def primary_space(self) -> "Space[object] | None":
        name = self.primary_space_name()
        if name is None:
            return None
        return self.nested_space(name, ())

    def primary_space_ref(self) -> refs.SpaceRef | None:
        name = self.primary_space_name()
        if name is None:
            return None
        return refs.SpaceRef(name, ())

    def nested_space(
        self, name: refs.SpaceName, args: tuple[Value, ...]
    ) -> Space[Any] | None:
        try:
            f: Any = getattr(self, name.name)
            for i in name.indices:
                f = f[i]
            if not args:
                # TODO: we could check that the field is not supposed to be
                # parametric
                assert isinstance(f, Space)
                return cast(Space[Any], f)
            else:
                assert isinstance(f, Callable)
                f = cast(Callable[..., Space[Any]], f)
                return f(*args)
        except TypeError:
            return None

    @classmethod
    def spawn(cls, spawner: "_GeneralSpawner", **args: Any):
        def convert(
            name: refs.SpaceName, field: nf.FieldType, obj: Any
        ) -> Any:
            match field:
                case nf.SpaceF():
                    return spawner.nonparametric(name, obj)
                case nf.ParametricF(nf.SpaceF()):
                    return spawner.parametric(name, obj)
                case nf.EmbeddedF():
                    builder = EmbeddedTree.builder(obj)
                    return spawner.nonparametric(name, builder)
                case nf.ParametricF(nf.EmbeddedF()):
                    parametric_builder = EmbeddedTree.parametric_builder(obj)
                    return spawner.parametric(name, parametric_builder)
                case nf.DataF():
                    return obj
                case nf.SequenceF(f):
                    return [convert(name, f, x)[i] for i, x in enumerate(obj)]
                case nf.OptionalF(f):
                    assert convert(name, f, obj) if obj is not None else None
                case _:
                    assert False

        args_new = {
            fname: convert(refs.SpaceName(fname, ()), fkind, args[fname])
            for fname, fkind in cls.fields().items()
        }
        return cls(**args_new)


type Navigation = Generator[Space[Any], Tracked[Any], Value]
"""
Nodes have a `navigate` method that yields spaces and expects
corresponding elements in return until an action is generated.
"""


####
#### Strategy Type
####


type Strategy[N: Node, P, T] = Generator[NodeBuilder[N, P], object, T]
"""
The return type for strategies.
"""


# We provide manual variance annotations since `P` is a phantom type
# that would be inferred both variant and covariant otherwise.
N = TypeVar("N", bound=Node, covariant=True, contravariant=False)
P = TypeVar("P", covariant=False, contravariant=True)
T = TypeVar("T", covariant=True, contravariant=False)


@dataclass(frozen=True)
class NodeBuilder(Generic[N, P]):
    """
    Wrapping `_NodeBuilder` to add a phantom type `P` for tracking a
    strategy's inner policy type.
    """

    node_builder: "_NodeBuilder[N]"


type _NodeBuilder[N: Node] = Callable[[_GeneralSpawner], N]
"""
Strategies do not directly yield nodes since building a node requires
knowing its reference along with the associated hooks.
"""


@dataclass(frozen=True)
class StrategyComp[N: Node, P, T]:
    """
    A strategy computation also stores metadata for navigation and
    debugging.
    """

    comp: Callable[..., Strategy[N, P, T]]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __call__(self) -> Strategy[N, P, T]:
        return self.comp(*self.args, **self.kwargs)

    def strategy_name(self) -> str | None:
        return inspect.function_name(self.comp)

    def strategy_arguments(self) -> dict[str, Any]:
        return inspect.function_args_dict(self.comp, self.args, self.kwargs)

    def tags(self) -> Sequence[Tag]:
        return [name] if (name := self.strategy_name()) is not None else []

    def return_type(self) -> TypeAnnot[T] | NoTypeInfo:
        ret_type = inspect.function_return_type(self.comp)
        if isinstance(ret_type, NoTypeInfo):
            return NoTypeInfo()
        return inspect.return_type_of_strategy_type(ret_type)


#####
##### Search and Prompting Policies
#####


type Policy[N: Node, P] = tuple[SearchPolicy[N], P]


class TreeTransformer[N: Node, M: Node](Protocol):
    def __call__[T, P](self, tree: "Tree[N, P, T]") -> "Tree[M, P, T]": ...


class _SearchPolicyFn[N: Node](Protocol):
    def __call__[P, T](
        self, tree: "Tree[N, P, T]", env: PolicyEnv, policy: P
    ) -> Stream[T]: ...


@dataclass(frozen=True)
class SearchPolicy[N: Node]:
    fn: _SearchPolicyFn[N]

    def __call__[P, T](
        self,
        tree: "Tree[N, P, T]",
        env: PolicyEnv,
        policy: P,
    ) -> Stream[T]:
        return self.fn(tree, env, policy)

    def __matmul__[M: Node](
        self, tree_transformer: TreeTransformer[M, N]
    ) -> "SearchPolicy[M]":
        def fn[P, T](
            tree: Tree[M, P, T],
            env: PolicyEnv,
            policy: P,
        ) -> Stream[T]:
            new_tree = tree_transformer(tree)
            return self.fn(new_tree, env, policy)

        return SearchPolicy(fn)


class _PromptingPolicyFn(Protocol):
    def __call__[T](
        self, query: AttachedQuery[T], env: PolicyEnv
    ) -> Stream[T]: ...


@dataclass(frozen=True)
class PromptingPolicy:
    fn: _PromptingPolicyFn

    def __call__[T](
        self, query: AttachedQuery[T], env: PolicyEnv
    ) -> Stream[T]:
        return self.fn(query, env)


#####
##### Embedded Trees
#####


@dataclass(frozen=True)
class NestedTree[N: Node, P, T](Space[T]):
    _comp: StrategyComp[N, P, T]
    spawn_tree: "Callable[[], Tree[N, P, T]]"

    def source(self) -> "NestedTree[Any, Any, T]":
        return self

    def tags(self) -> Sequence[Tag]:
        return self._comp.tags()


@dataclass(frozen=True)
class EmbeddedTree[N: Node, P, T](NestedTree[N, P, T]):
    @staticmethod
    def builder[N1: Node, P1, T1](
        strategy: StrategyComp[N1, P1, T1],
    ) -> "Builder[NestedTree[N1, P1, T1]]":
        return lambda spawn, _: NestedTree(strategy, lambda: spawn(strategy))

    @staticmethod
    def parametric_builder[A, N1: Node, P1, T1](
        parametric_strategy: Callable[[A], StrategyComp[N1, P1, T1]],
    ) -> "Callable[[A], Builder[NestedTree[N1, P1, T1]]]":
        return lambda arg: EmbeddedTree.builder(parametric_strategy(arg))


#####
##### Opaque Spaces
#####


@dataclass(frozen=True)
class OpaqueSpace[P, T](Space[T]):
    stream: Callable[[PolicyEnv, P], Stream[T]]
    _source: "NestedTree[Any, Any, T] | AttachedQuery[T]"

    def source(self) -> "NestedTree[Any, Any, T] | AttachedQuery[T]":
        return self._source

    def tags(self) -> Sequence[Tag]:
        return self._source.tags()

    @staticmethod
    def from_query[P1, T1](
        query: AbstractQuery[T1], get_policy: Callable[[P1], PromptingPolicy]
    ) -> "Builder[OpaqueSpace[P1, T1]]":
        def build(spawner: _QuerySpawner) -> OpaqueSpace[P1, T1]:
            attached = spawner(query)
            return OpaqueSpace(
                (lambda env, pol: get_policy(pol)(attached, env)), attached
            )

        return lambda _, spawner: build(spawner)

    @staticmethod
    def from_strategy[N: Node, P1, P2, T1](
        strategy: StrategyComp[N, P2, T1],
        get_policy: Callable[[P1], tuple[SearchPolicy[N], P2]],
    ) -> "Builder[OpaqueSpace[P1, T1]]":
        def build(spawner: _TreeSpawner) -> OpaqueSpace[P1, T1]:
            def stream(env: PolicyEnv, policy: P1) -> Stream[T1]:
                tree = spawner(strategy)
                search_pol, inner_pol = get_policy(policy)
                return search_pol(tree, env, inner_pol)

            nested = NestedTree(strategy, lambda: spawner(strategy))
            return OpaqueSpace(stream, nested)

        return lambda spawner, _: build(spawner)


#####
##### Builders and Spawners
#####


class _TreeSpawner(Protocol):
    def __call__[N: Node, P, T](
        self, strategy: "StrategyComp[N, P, T]"
    ) -> "Tree[N, P, T]": ...


class _QuerySpawner(Protocol):
    def __call__[T](self, query: AbstractQuery[T]) -> "AttachedQuery[T]": ...


type Builder[S] = Callable[[_TreeSpawner, _QuerySpawner], S]


@dataclass(frozen=True)
class _GeneralSpawner:
    """
    Allows spawning arbitrary parametric subspaces at a given node.
    """

    _ref: "_NonEmptyGlobalPath"
    _node_hook: "Callable[[Tree[Any, Any, Any]], None] | None"

    def parametric[S](
        self,
        space_name: SpaceName,
        parametric_builder: Callable[..., Builder[S]],
    ) -> Callable[..., S]:
        def run_builder(*args: Any) -> S:
            args_raw = [refs.drop_refs(arg) for arg in args]
            args_ref = tuple(refs.value_ref(arg) for arg in args)
            builder = parametric_builder(*args_raw)
            gr = _global_path_from_nonempty(self._ref)
            sr = refs.SpaceRef(space_name, args_ref)

            def spawn_tree[N: Node, P, T](
                strategy: StrategyComp[N, P, T],
            ) -> Tree[N, P, T]:
                return _reify(strategy, (gr, sr, ()), self._node_hook)

            def spawn_query[T](
                query: AbstractQuery[T],
            ) -> AttachedQuery[T]:
                def answer(
                    mode: refs.AnswerModeName, text: str
                ) -> Tracked[T] | ParseError:
                    ref = refs.SpaceElementRef(sr, refs.Answer(mode, text))
                    atype = query.answer_type()
                    parsed = query.modes()[mode].parse(atype, text)
                    if isinstance(parsed, ParseError):
                        return parsed
                    return Tracked(parsed, ref, gr, query.answer_type())

                return AttachedQuery(query, (gr, sr), answer)

            return builder(spawn_tree, spawn_query)

        return run_builder

    def nonparametric[S](self, name: SpaceName, builder: Builder[S]) -> S:
        return self.parametric(name, lambda: builder)()


#####
##### Tree Type
#####


@dataclass(frozen=True)
class Tree(Generic[N, P, T]):
    node: "N | Success[T]"
    child: "Callable[[Value], Tree[N, P, T]]"
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


def _global_path_from_nonempty(ref: _NonEmptyGlobalPath) -> GlobalNodePath:
    (gr, sr, nr) = ref
    return (*gr, (sr, nr))


def _reify[N: Node, P, T](
    strategy: StrategyComp[N, P, T],
    root_ref: _NonEmptyGlobalPath | None,
    node_hook: Callable[[Tree[Any, Any, Any]], None] | None,
) -> Tree[N, P, T]:
    """
    Reify a strategy into a tree.

    This version is private because it exposes the `root_ref` argument.
    Outside of these modules, trees can only be reified relative to the
    global origin.
    """

    def aux(
        strategy: StrategyComp[N, P, T],
        actions: Sequence[object],
        ref: _NonEmptyGlobalPath,
        node: N | Success[T],
        cur_generator: Strategy[N, P, T],
    ) -> Tree[N, P, T]:
        generator_valid = True

        def child(action: Value) -> Tree[N, P, T]:
            action_raw = refs.drop_refs(action)
            action_ref = refs.value_ref(action)
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
            else:
                pre_node, new_gen = _recompute(strategy, new_actions)
            gr, cr, nr = ref
            new_ref = (gr, cr, (*nr, action_ref))
            ret_type = strategy.return_type()
            new_node = _finalize_node(pre_node, new_ref, ret_type, node_hook)
            return aux(strategy, new_actions, new_ref, new_node, new_gen)

        tree = Tree(node, child, _global_path_from_nonempty(ref))
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


type _PreNode[N: Node, T] = _NodeBuilder[N] | _PreSuccess[T]


def _send_action[N: Node, P, T](
    gen: Strategy[N, P, T], action: object
) -> _PreNode[N, T]:
    try:
        return gen.send(action).node_builder
    except StopIteration as e:
        v = cast(T, e.value)
        return _PreSuccess(v)
    except Exception as e:
        raise StrategyException(e)


def _recompute[N: Node, P, T](
    strategy: StrategyComp[N, P, T],
    actions: Sequence[object],
) -> tuple[_PreNode[N, T], Strategy[N, P, T]]:
    gen = strategy()
    mknode = _send_action(gen, None)
    for a in actions:
        mknode = _send_action(gen, a)
    return mknode, gen


def _finalize_node[N: Node, T](
    pre_node: _PreNode[N, T],
    ref: _NonEmptyGlobalPath,
    type: TypeAnnot[T] | NoTypeInfo,
    node_hook: Callable[[Tree[Any, Any, Any]], None] | None,
) -> N | Success[T]:
    if not isinstance(pre_node, _PreSuccess):
        return pre_node(_GeneralSpawner(ref, node_hook))
    (gr, sr, nr) = ref
    ser = refs.SpaceElementRef(sr, nr)
    return Success(Tracked(pre_node.value, ser, gr, type))


def reify[N: Node, P, T](
    strategy: StrategyComp[N, P, T],
    node_hook: Callable[[Tree[Any, Any, Any]], None] | None = None,
) -> Tree[N, P, T]:
    """
    Reify a strategy into a tree.
    """
    return _reify(strategy, None, node_hook)


def tracer_hook(tracer: en.Tracer) -> Callable[[Tree[Any, Any, Any]], None]:
    return lambda tree: tracer.trace_node(tree.ref)
