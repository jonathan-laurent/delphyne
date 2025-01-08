"""
The core tree datastructure.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

from delphyne.core import inspect, refs
from delphyne.core import node_fields as nf
from delphyne.core.node_fields import NodeFields, detect_node_structure
from delphyne.core.queries import AbstractQuery, ParseError
from delphyne.core.refs import GlobalNodePath, SpaceName, Tracked, Value
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
    ref: refs.GlobalSpacePath
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

    def primary_space(self) -> "Space[object] | None":
        """
        Space that is inferred by default in a space element reference
        (see `SpaceElementRef`). For example, `compare([cands{''},
        cands{'foo bar'}])` can be abbreviated into `compare(['', 'foo
        bar'])` if `cands` is the primary space for the current node.
        """
        return None

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

    def primary_space_ref(self) -> refs.SpaceRef | None:
        space = self.primary_space()
        if space is None:
            return None
        return space.source().ref[1]

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
        except (TypeError, AttributeError):
            return None

    @classmethod
    def spawn(cls, spawner: "AbstractBuilderExecutor", **args: Any):
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
    Strategies do not directly yield nodes since building a node
    requires knowing its reference along with the associated hooks.

    We add a phantom type `P` for tracking a strategy's inner policy
    type.
    """

    build_node: "Callable[[AbstractBuilderExecutor], N]"


@dataclass(frozen=True)
class StrategyComp[N: Node, P, T]:
    """
    A strategy computation also stores metadata for navigation and
    debugging.
    """

    comp: Callable[..., Strategy[N, P, T]]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    name: str | None = None

    def run_generator(self) -> Strategy[N, P, T]:
        return self.comp(*self.args, **self.kwargs)

    def strategy_name(self) -> str | None:
        if self.name is not None:
            return self.name
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
##### Nested and Embedded Trees
#####


@dataclass(frozen=True)
class NestedTree[N: Node, P, T](Space[T]):
    strategy: StrategyComp[N, P, T]
    ref: refs.GlobalSpacePath
    spawn_tree: "Callable[[], Tree[N, P, T]]"

    def source(self) -> "NestedTree[Any, Any, T]":
        return self

    def tags(self) -> Sequence[Tag]:
        return self.strategy.tags()


@dataclass(frozen=True)
class EmbeddedTree[N: Node, P, T](NestedTree[N, P, T]):
    @staticmethod
    def builder[N1: Node, P1, T1](
        strategy: StrategyComp[N1, P1, T1],
    ) -> "Builder[NestedTree[N1, P1, T1]]":
        return lambda spawn, _: spawn(strategy)

    @staticmethod
    def parametric_builder[A, N1: Node, P1, T1](
        parametric_strategy: Callable[[A], StrategyComp[N1, P1, T1]],
    ) -> "Callable[[A], Builder[NestedTree[N1, P1, T1]]]":
        return lambda arg: EmbeddedTree.builder(parametric_strategy(arg))


#####
##### Builders and Spawners
#####


class NestedTreeSpawner(Protocol):
    def __call__[N: Node, P, T](
        self, strategy: "StrategyComp[N, P, T]"
    ) -> "NestedTree[N, P, T]": ...


class QuerySpawner(Protocol):
    def __call__[T](self, query: AbstractQuery[T]) -> "AttachedQuery[T]": ...


type Builder[S] = Callable[[NestedTreeSpawner, QuerySpawner], S]
"""
On the strategy side, not enough information is typically available to
build spaces so builders are provided instead.
"""


class AbstractBuilderExecutor(ABC):
    """
    Allows spawning arbitrary nested spaces at a given node, given
    builders for them.
    """

    @abstractmethod
    def parametric[S](
        self,
        space_name: SpaceName,
        parametric_builder: Callable[..., Builder[S]],
    ) -> Callable[..., S]: ...

    @abstractmethod
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


type AnyTree = Tree[Node, Any, Any]


class TreeTransformer[N: Node, M: Node](Protocol):
    def __call__[T, P](self, tree: "Tree[N, P, T]") -> "Tree[M, P, T]": ...
