"""
The core tree datastructure.
"""

import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic, Protocol, TypeVar, cast, override

from delphyne.core import inspect, refs
from delphyne.core import node_fields as nf
from delphyne.core.node_fields import NodeFields, detect_node_structure
from delphyne.core.queries import AbstractQuery, ParseError
from delphyne.core.refs import GlobalNodePath, SpaceName, Tracked, Value
from delphyne.utils.typing import NoTypeInfo, TypeAnnot

#####
##### Spaces
#####


type Tag = str
"""
String tags for nodes and spaces.

Nodes and spaces can be assigned string identifiers, which may be
referenced in demonstration tests or when defining inner policies
(e.g., `IPDict`). Tags should contain only alphanumeric characters,
underscores, dots, and dashes.
"""


class Space[T](ABC):
    """
    Abstract type for a space.

    Tree nodes feature local spaces (possibly parametric), that are
    backed by either queries or nested trees. Examples of spaces
    include `EmbeddedTree`, `OpaqueSpace` and `TransparentQuery`.
    """

    @abstractmethod
    def tags(self) -> Sequence[Tag]:
        """
        Returns the tags associated with the space.
        """
        pass

    @abstractmethod
    def source(self) -> "NestedTree[Any, Any, T] | AttachedQuery[T]":
        """
        Returns the source of the space, which is either a nested tree
        or an attached query.

        This method is mostly useful to the demonstration interpreter.
        It is not typically used in policies since it breaks
        abstraction (e.g., whether or not an opaque space is defined via
        a query or a strategy is irrelevant).
        """
        pass


@dataclass(frozen=True)
class AttachedQuery[T]:
    """
    Wrapper for a query attached to a specific space.

    Attributes:
        query: The wrapped query.
        ref: A global reference to the space to which the query is
            attached.
        parse_answer: A wrapper around `self.query.parse_answer`,
            which attaches proper tracking information to answers.
    """

    query: AbstractQuery[T]
    ref: refs.GlobalSpacePath
    parse_answer: Callable[[refs.Answer], Tracked[T] | ParseError]


@dataclass(frozen=True)
class TransparentQuery[T](Space[T]):
    """
    A space that is defined by a single, *transparent* query.

    As opposed to `OpaqueSpace`, the query is meant to be directly
    exposed to policies. This is used to define `Compute` and `Flag`
    nodes for example.
    """

    attached: AttachedQuery[T]
    _tags: "Sequence[Tag]"

    @override
    def source(self) -> AttachedQuery[T]:
        return self.attached

    @override
    def tags(self) -> Sequence[Tag]:
        return self._tags

    @staticmethod
    def build[T1](
        query: AbstractQuery[T1],
    ) -> "SpaceBuilder[TransparentQuery[T1]]":
        return SpaceBuilder(
            build=lambda _, spawner, tags: TransparentQuery(
                spawner(query), tags
            ),
            tags=query.default_tags(),
        )


#####
##### Nodes
#####


@dataclass(frozen=True)
class NavigationError(Exception):
    """
    Exception raised when an error occurs during navigation.

    For internal errors that should not occur within normal use,
    assertions shoule be used instead. This exception is meant to
    represent errors that can occur during normal use, and reported in
    the user interface. For an example, see `Abduction`.

    Attributes:
        message: A human-readable error message.
    """

    message: str


class Node(ABC):
    """
    Abstract type for a tree node.

    New effects can be added to the strategy language by subclassing
    `Node` and then defining a triggering function that calls the
    `spawn` class method (manually defining such a wrapper allows
    providing users with precise type annotations: i.e., `branch` for
    `Branch`).

    **Methods to override:**

    - The `navigate` method must be overriden for all node types.
    - The `leaf_node` method must be overriden for all leaf nodes.
    - The following methods are also frequently overriden:
        `summary_message`, `valid_action` and `primary_space`.

    Other methods are implemented via inspection and are not typically
    overriden.
    """

    # Methods that **must** be overriden

    @abstractmethod
    def navigate(self) -> "Navigation":
        """
        The navigation method, to be defined for each node type.

        It should only be called when `self.leaf_node()` returns False.

        See `Navigation` for details.
        """
        pass

    # Methods that are _sometimes_ overriden

    def summary_message(self) -> str | None:
        """
        Returns an optional summary message for the node, to be
        displayed in the Delphyne extension's Tree View.
        """
        return None

    def leaf_node(self) -> bool:
        """
        Returns True if the node is a leaf node (e.g., `Fail`) and False
        otherwise.

        Leaf nodes do not have to define `navigate` and are treated
        specially by the demonstration interpreter.
        """
        return False

    def valid_action(self, action: object) -> bool:
        """
        Returns whether an action is valid.

        By default, this method always returns `True`. When overriden,
        it is used to dynamically check actions passed to `Tree.child`,
        **after** the `Tracked` wrapper is removed.
        """
        return True

    def primary_space(self) -> "Space[object] | None":
        """
        Optionally returns the node's primary space.

        Primary spaces are useful to shorten space references,
        especially when writing demonstration tests. For example, if
        `cands` is the primary space of the current node, then
        `compare([cands{''}, cands{'foo bar'}])` can be abbreviated into
        `compare(['', 'foo bar'])`.
        """
        return None

    # Method indicating the nature of the node's fields (for inspection)

    @classmethod
    def fields(cls) -> NodeFields:
        """
        Returns a dictionary mapping each field of the node to some
        metadata (e.g., whether the field denotes a local space).

        Such metadata is useful in particular to implement `spawn`.

        The default implementation uses inspection, via
        `detect_node_structure`. See the
        [`node_fields`][delphyne.core.node_fields] module for details.
        """
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
            name: refs.SpaceName, field: nf.FieldKind, obj: Any
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
                    return [convert(name[i], f, x) for i, x in enumerate(obj)]
                case nf.OptionalF(f):
                    assert convert(name, f, obj) if obj is not None else None
                case _:
                    assert False

        args_new = {
            fname: convert(refs.SpaceName(fname, ()), fkind, args[fname])
            for fname, fkind in cls.fields().items()
        }
        return cls(**args_new)

    def map_embedded[N: Node](
        self: N, trans: "AbstractTreeTransformer[Any, Any]"
    ) -> N:
        """
        Apply a tree transformer to all embedded trees and return
        the updated node (the argument is still valid afterwards).
        """

        def convert_embedded(
            emb: EmbeddedTree[Any, Any, Any],
        ) -> EmbeddedTree[Any, Any, Any]:
            assert isinstance(emb, EmbeddedTree), (
                f"Expected an EmbeddedTree, got {type(emb)}"
            )
            nested = NestedTree(
                strategy=emb.nested.strategy,
                ref=emb.nested.ref,
                spawn_tree=lambda: trans(emb.nested.spawn_tree()),
            )
            return EmbeddedTree(nested, _tags=emb.tags())

        def convert_parametric_embedded(obj: Any) -> Callable[[Any], Any]:
            return lambda arg: convert_embedded(obj(arg))

        def convert(field: nf.FieldKind, obj: Any) -> Any:
            match field:
                case nf.SpaceF() | nf.ParametricF(nf.SpaceF()) | nf.DataF():
                    return obj
                case nf.EmbeddedF():
                    return convert_embedded(obj)
                case nf.ParametricF(nf.EmbeddedF()):
                    return convert_parametric_embedded(obj)
                case nf.SequenceF(f):
                    return [convert(f, x) for x in obj]
                case nf.OptionalF(f):
                    return convert(f, obj) if obj is not None else None
                case _:
                    assert False

        args_new = {
            fname: convert(fkind, getattr(self, fname))
            for fname, fkind in self.fields().items()
        }
        return type(self)(**args_new)


type Navigation = Generator[Space[Any], Tracked[Any], Value]
"""
A navigation generator.

A navigation generator is returned by the
[`navigate`][delphyne.Node.navigate] method of non-leaf nodes. It yields
local spaces and receives corresponding elements until an action is
generated and returned.
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

    Phantom type `P` tracks the ambient inner policy type.
    """

    build_node: "Callable[[AbstractBuilderExecutor], N]"


@dataclass(frozen=True)
class StrategyComp(Generic[N, P, T]):
    """
    A *strategy computation* that can be reified into a search tree,
    obtained by instantiating a strategy function.

    Such objects atr usually not created directly but using the
    [`@strategy` decorator][delphyne.stdlib.strategy] from the standard
    library. Metadata information can be provided as: a name for the
    strategy, a return type and a list of tags.

    ??? note
        The name of the `_comp` function is inspected by methods such as
        `strategy_name`. Its signature (i.e., the name of its arguments)
        is inspected by `strategy_arguments`. When provided, type
        annotations are inspected by methods such as `return_type` and
        `inner_policy_type`. Thus, passing an anonymous function as
        `_comp` is not recommended.

    ??? note

    """

    _comp: Callable[..., Strategy[N, P, T]]
    _args: tuple[Any, ...]
    _kwargs: dict[str, Any]
    _name: str | None
    _return_type: TypeAnnot[T] | NoTypeInfo
    _tags: Sequence[Tag]

    ##### External API

    def inline(self) -> Strategy[N, P, T]:
        """
        Inline a strategy computation within another, by executing the
        underlying coroutine.

        Example:

        ```
        # Invoking a sub-strategy by branching over an opaque space.
        y = yield from branch(sub_strategy(foo, bar).using(...))
        # Invoking a sub-strategy via inlining (the signature and inner
        # policy types of the sub strategy must be identical).
        y = yield from sub_strategy(foo, bar).inline()
        ```
        """
        return self.run_generator()

    ##### For internal use in `core` and in the standard library

    def run_generator(self) -> Strategy[N, P, T]:
        """
        Run the coroutine associated with the strategy computation. This
        method is mostly used by [`reify`][delphyne.core.reify] and
        should not be needed outside of Delphyne's internals.
        """
        return self._comp(*self._args, **self._kwargs)

    def default_tags(self) -> Sequence[Tag]:
        """
        Return all default tags associated with the strategy
        computation.
        """
        return self._tags

    ### Inspection methods

    def strategy_name(self) -> str | None:
        """
        Return the name of the instantiated strategy function, using the
        `name` attribute if provided or using `comp.__name__` otherwise.
        """
        if self._name is not None:
            return self._name
        return inspect.function_name(self._comp)

    def strategy_arguments(self) -> dict[str, Any]:
        """
        Return the dictionary of arguments that was used to instantiate
        the undelrying strategy function (using inspection for naming
        positional arguments).
        """
        return inspect.function_args_dict(self._comp, self._args, self._kwargs)

    def strategy_argument_types(
        self,
    ) -> dict[str, TypeAnnot[Any] | NoTypeInfo]:
        """
        Return a dictionary with the same keys as
        [`strategy_arguments`](delphyne.core.StrategyComp.strategy_arguments)
        that maps every strategy argument to its type annotation (or
        `NoTypeInfo` if none is provided).
        """
        hints = typing.get_type_hints(self._comp)
        return {
            a: hints.get(a, NoTypeInfo()) for a in self.strategy_arguments()
        }

    def return_type(self) -> TypeAnnot[T] | NoTypeInfo:
        """
        Return the return type of the strategy computation, using the
        provided metadata or using inspection otherwise (in case a type
        annotation of the form `Strategy[..., ..., T]` is provided).

        This information is useful for serializing success values after
        running the strategy computation (see [`run_strategy`
        command][delphyne.stdlib.commands.run_strategy]) or for
        providing superior printing of values in the Delphyne tree view.
        """
        if not isinstance(self._return_type, NoTypeInfo):
            return self._return_type
        strategy_type = inspect.function_return_type(self._comp)
        if isinstance(strategy_type, NoTypeInfo):
            return NoTypeInfo()
        return inspect.return_type_of_strategy_type(strategy_type)

    def inner_policy_type(self) -> TypeAnnot[T] | NoTypeInfo:
        """
        Return the inner policy type of the strategy computation, using
        inspection (when a type annotation of the form `Strategy[..., P,
        ...]` is provided).

        This information is not currently used but could be leveraged in
        the future to dynamically check the compatibility of an
        associated policy.
        """
        strategy_type = inspect.function_return_type(self._comp)
        if isinstance(strategy_type, NoTypeInfo):
            return NoTypeInfo()
        return inspect.inner_policy_type_of_strategy_type(strategy_type)


#####
##### Nested and Embedded Trees
#####


@dataclass(frozen=True)
class NestedTree[N: Node, P, T]:
    """
    A nested tree is one can back up an arbitrary space (including
    opaque spaces).

    Note: one cannot count on `strategy` having the same node type as
    `spawn_tree` since nested trees can be applied tree transformers
    (see `Node.map_embedded`).
    """

    strategy: StrategyComp[Node, P, T]
    ref: refs.GlobalSpacePath
    spawn_tree: "Callable[[], Tree[N, P, T]]"


@dataclass(frozen=True)
class EmbeddedTree[N: Node, P, T](Space[T]):
    """
    An embedded tree wraps a nested tree, declaring it as such and
    turning it into a space.
    """

    nested: NestedTree[N, P, T]
    _tags: Sequence[Tag]

    @staticmethod
    def builder[N1: Node, P1, T1](
        strategy: StrategyComp[N1, P1, T1],
    ) -> "SpaceBuilder[EmbeddedTree[N1, P1, T1]]":
        return SpaceBuilder[EmbeddedTree[N1, P1, T1]](
            build=lambda spawn, _, tags: EmbeddedTree(spawn(strategy), tags),
            tags=strategy.default_tags(),
        )

    @staticmethod
    def parametric_builder[A, N1: Node, P1, T1](
        parametric_strategy: Callable[[A], StrategyComp[N1, P1, T1]],
    ) -> "Callable[[A], SpaceBuilder[EmbeddedTree[N1, P1, T1]]]":
        return lambda arg: EmbeddedTree.builder(parametric_strategy(arg))

    def source(self) -> "NestedTree[Any, Any, T]":
        return self.nested

    def tags(self) -> Sequence[Tag]:
        return self._tags

    def spawn_tree(self) -> "Tree[N, P, T]":
        return self.nested.spawn_tree()


#####
##### Builders and Spawners
#####


class NestedTreeSpawner(Protocol):
    def __call__[N: Node, P, T](
        self, strategy: "StrategyComp[N, P, T]"
    ) -> "NestedTree[N, P, T]": ...


class QuerySpawner(Protocol):
    def __call__[T](self, query: AbstractQuery[T]) -> "AttachedQuery[T]": ...


@dataclass(frozen=True)
class SpaceBuilder[S]:
    """
    On the strategy side, not enough information is typically available to
    build spaces so builders are provided instead.
    """

    build: Callable[[NestedTreeSpawner, QuerySpawner, Sequence[Tag]], S]
    tags: Sequence[Tag]

    def tagged(self, *tags: Tag) -> "SpaceBuilder[S]":
        return replace(self, tags=(*self.tags, *tags))

    def __call__(
        self, spawner: NestedTreeSpawner, query_spawner: QuerySpawner
    ) -> S:
        return self.build(spawner, query_spawner, self.tags)


class AbstractBuilderExecutor(ABC):
    """
    Allows spawning arbitrary nested spaces at a given node, given
    builders for them.
    """

    @abstractmethod
    def parametric[S](
        self,
        space_name: SpaceName,
        parametric_builder: Callable[..., SpaceBuilder[S]],
    ) -> Callable[..., S]: ...

    @abstractmethod
    def nonparametric[S](self, name: SpaceName, builder: SpaceBuilder[S]) -> S:
        return self.parametric(name, lambda: builder)()


#####
##### Tree Type
#####


@dataclass(frozen=True)
class Tree(Generic[N, P, T]):
    node: "N | Success[T]"
    child: "Callable[[Value], Tree[N, P, T]]"
    ref: GlobalNodePath

    def transform[M: Node](
        self,
        node: "M | Success[T]",
        transformer: "AbstractTreeTransformer[N, M]",
    ) -> "Tree[M, P, T]":
        def child(action: Value) -> Tree[M, P, T]:
            return transformer(self.child(action))

        node = node.map_embedded(transformer)
        return Tree(node, child, self.ref)


@dataclass(frozen=True)
class Success[T]:
    """
    A success leaf, carrying a tracked value.

    Note that although it largely implements the same interface via duck
    typing, `Success` does not inherit `Node`. The reason is that
    `Tree[N, P, T].node` has type `Success[T] | N`. If `Success`
    inherited `Node`, it would be possible for `N` to include a value of
    type `Success[T2]` for some `T2 != T`, which we want to avoid.
    """

    success: Tracked[T]

    def leaf_node(self) -> bool:
        return True

    def valid_action(self, action: object) -> bool:
        return False

    def navigate(self) -> Navigation:
        assert False

    def summary_message(self) -> str | None:
        return None

    def primary_space(self) -> None:
        return None

    def primary_space_ref(self) -> None:
        return None

    def effect_name(self) -> str:
        return self.__class__.__name__

    def get_tags(self) -> Sequence[Tag]:
        return ()

    def nested_space(
        self, name: refs.SpaceName, args: tuple[Value, ...]
    ) -> Space[Any] | None:
        return None

    def map_embedded(
        self, trans: "AbstractTreeTransformer[Any, Any]"
    ) -> "Success[T]":
        return self


@dataclass
class StrategyException(Exception):
    """
    Raised when a strategy encounters an internal error (e.g. a failed
    assertion or an index-out-of-bounds error).
    """

    exn: Exception


type AnyTree = Tree[Node, Any, Any]


class AbstractTreeTransformer[N: Node, M: Node](Protocol):
    def __call__[T, P](self, tree: "Tree[N, P, T]") -> "Tree[M, P, T]": ...


#####
##### Special Nodes
#####


class ComputationNode(Node):
    @abstractmethod
    def run_computation(self) -> str:
        pass
