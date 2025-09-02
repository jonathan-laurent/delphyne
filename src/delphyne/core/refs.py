"""
References to nodes, values, spaces and space elements.

References are serializable, immutable values that can be used to
identify nodes, spaces and values in a tree (possibly deeply nested).
References are useful for tooling and for representing serializable
traces (`Trace`). Also, references are attached to success nodes and
query answers (`Tracked`) so as to allow caching and enforce the
*locality invariant* (see `Tree`).

*Local* references identify a node, space or space element *relative* to
a given tree node. *Global* references are expressed relative to a
single, global origin.

In addition, three kinds of references can be distinguished:

- **Full references**: the default kind of references produced by
      `reify`. Query answers are stored as strings and elements of
      spaces induced by strategies are denoted by sequences of value
      references.
- **Id-based references**: shorter references, where query answers and
      success values are identified by unique identifiers. This concise
      format is used for exporting traces (see `Trace`).
- **Hint-based references**: query answers and success values are
      identified by sequences of *hints*. This format is used in the
      demonstration language (e.g. argument of test instruction `go
      compare(['', 'foo bar'])`) and when visualizing traces resulting
      from demonstrations.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, overload

import delphyne.core.inspect as insp
from delphyne.utils.typing import NoTypeInfo, TypeAnnot

#####
##### Query Answers
#####


@dataclass(frozen=True)
class ToolCall:
    """
    A tool call, usually produced by an LLM oracle.

    Tool calls can be attached to LLM answers (see `Answer`).
    """

    name: str
    args: Mapping[str, Any]

    def _hashable_repr(self) -> str:
        # Tool calls need to be hashable since they are part of answers
        # and references. However, they can feature arbitrary JSON
        # objects.
        import json

        return json.dumps(self.__dict__, sort_keys=True)

    def __hash__(self) -> int:
        return hash(self._hashable_repr())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return NotImplemented
        return self._hashable_repr() == other._hashable_repr()


@dataclass(frozen=True)
class Structured:
    """
    Wrapper for structured LLM answers.

    Many LLM APIs allow producing JSON answers (sometimes following a
    given schema) instead of plain text.
    """

    structured: Any  # JSON object

    def _hashable_repr(self) -> str:
        # See comment in ToolCall._hashable_repr
        import json

        return json.dumps(self.__dict__, sort_keys=True)

    def __hash__(self) -> int:
        return hash(self._hashable_repr())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Structured):
            return NotImplemented
        return self._hashable_repr() == other._hashable_repr()


type AnswerMode = str | None
"""
A name for an answer mode, which can be a string or `None` (the latter
is typically used for naming default modes).

Queries are allowed to define multiple answer modes, each mode being
possibly associated with different settings and with a different parser.
An `Answer` value features the mode that must be used to parse it.
"""


@dataclass(frozen=True)
class Answer:
    """
    An answer to a query.

    It can serve as a _space element reference_ if the space in question
    is a query and the proposed answer correctly parses.

    Attributes:
        mode: The answer mode (see `AnswerMode`).
        content: The answer content, which can be a raw string or a
            structured answer (see `Structured`).
        tool_calls: An optional sequence of tool calls.
        justification: Additional explanations for the answers, which
            are not passed to the parser but can be appended at the end
            of the answer in examples. In particular, this is useful
            when defining queries for which the oracle is not asked to
            produce a justification for its answer, but justifications
            can still be provided in examples for the sake of few-shot
            prompting.
    """

    mode: AnswerMode
    content: str | Structured
    tool_calls: tuple[ToolCall, ...] = ()
    justification: str | None = None


#####
##### References
#####


@dataclass(frozen=True)
class SpaceName:
    """
    A name identifying a parametric space.

    This name can feature integer indices. For example, `subs[0]`
    denotes the first subgoal of a `Join` node.
    """

    name: str
    indices: tuple[int, ...]

    def __getitem__(self, index: int) -> "SpaceName":
        return SpaceName(self.name, (*self.indices, index))


type AtomicValueRef = IndexedRef | SpaceElementRef
"""
An atomic value reference is a space element reference that is indexed
zero or a finite number of times: `space_elt_ref[i1][i2]...[in]`.
"""


@dataclass(frozen=True)
class IndexedRef:
    """
    Indexing an atomic value reference.
    """

    ref: AtomicValueRef
    index: int


type ValueRef = Assembly[AtomicValueRef]
"""
A reference to a local value, which is obtained by combining elements of
(possibly multiple) local spaces.
"""


type Assembly[T] = T | None | tuple[Assembly[T], ...]
"""
An S-expression whose atoms have type `T`.
"""


type ExtAssembly[T] = T | None | Sequence[ExtAssembly[T]]
"""
Generalizing `Assembly` to allow arbitrary sequences (and not just
tuples). The distinction is important because `ValueRef` needs to be
hashable and so cannot contain lists, while `Value` can contain lists. 
"""


type NodePath = tuple[ValueRef, ...]
"""
Encodes a sequence of actions leading to a node with respect to a
given root.
"""


@dataclass(frozen=True)
class NodeId:
    """
    Global identifier of a node within a trace.
    """

    id: int


type NodeRef = NodePath | NodeId
"""
A node reference is either a path or a node identifier.

Only one of these forms may be allowed depending on the context (e.g. in
the id-based references used for exporting traces, only node identifiers
are used, while in the full references attached to trees by `reify`,
only paths are used).
"""


@dataclass(frozen=True)
class SpaceRef:
    """
    A reference to a specific local space.

    The `arg` argument should be `()` for nonparametric spaces and a
    n-uple for spaces parametric in n arguments. This differs from
    Orakell where all parametric spaces have one argument.
    """

    name: SpaceName
    args: tuple[ValueRef, ...]


MAIN_SPACE = SpaceRef(SpaceName("$main", ()), ())
"""
A special space attached to the *global origin* node, and which contains
the main, top-level strategy tree.
"""

MAIN_ROOT: "GlobalNodePath" = ((MAIN_SPACE, ()),)
"""
Global reference to the root of the main, top-level strategy tree.
"""


@dataclass(frozen=True)
class AnswerId:
    """
    The identifier to an `Answer` object stored within a trace.
    """

    id: int


type AnswerRef = Answer | AnswerId
"""
A reference to a query answer.
"""


type HintValue = str
"""
A string that hints at a query answer.
"""


@dataclass(frozen=True)
class Hint:
    """A hint for selecting a query answer.

    A hint can be associated to a qualifier, which is the name of an
    imported demonstration defining the hint.
    """

    qualifier: str | None
    hint: HintValue


@dataclass(frozen=True)
class HintsRef:
    """
    References a local space element via a sequence of hints.
    """

    hints: tuple[Hint, ...]


@dataclass(frozen=True)
class SpaceElementRef:
    """
    A reference to an element of a local space.

    When the `space` field is `None`, the primary field is considered
    instead (if it exists).
    """

    space: SpaceRef | None
    element: AnswerRef | NodeRef | HintsRef


type GlobalNodePath = tuple[tuple[SpaceRef, NodePath], ...]
"""
Path to a node from the global origin, as a sequence of (space to enter,
path to follow) instruction pairs.
"""

type GlobalSpacePath = tuple[GlobalNodePath, SpaceRef]
"""
A path to a global node
"""


#####
##### Node Origins (used in traces)
#####


type NodeOrigin = ChildOf | NestedTreeOf
"""
Origin of a tree.

A tree is either the child of another tree or the root of a nested tree.
Traces can be exported as mappings from node identifiers to node origin
information featuring id-based references (see `Trace`).
"""


@dataclass(frozen=True)
class ChildOf:
    """
    The tree of interest is the child of another one.
    """

    node: NodeId
    action: ValueRef


@dataclass(frozen=True)
class NestedTreeOf:
    """
    The tree of interest is the root of a tree that induces a given
    space.
    """

    node: NodeId
    space: SpaceRef


#####
##### Tracked Values
#####


T = TypeVar("T", covariant=True)


@dataclass(frozen=True)
class Tracked(Generic[T]):
    """
    A tracked value, which pairs a value with a reference.

    Attributes:
        value: The value being tracked.
        ref: A local reference to the value, relative to the node
            reference by the `node` field.
        node: A global reference to the node to which the space that the
            value originates from is attached. In particular, this field
            is useful to check the locality invariant at runtime (e.g.,
            when passing a tracked value to `Tree.child`).
        type_annot: An optional type annotation for the `value` field.
            This is mostly used for improving the rendering of values
            when exporting trace information for external tools.

    Tracked sequences (or pairs) can be indexed using `__getitem__`,
    resulting in tracked values with `IndexedRef` references. Since
    `__getitem__` is defined, tracked values are also iterable.
    """

    value: T
    ref: AtomicValueRef
    node: GlobalNodePath
    type_annot: TypeAnnot[T] | NoTypeInfo

    @overload
    def __getitem__[A, B](
        self: "Tracked[tuple[A, B]]", index: Literal[0]
    ) -> "Tracked[A]": ...

    @overload
    def __getitem__[A, B](
        self: "Tracked[tuple[A, B]]", index: Literal[1]
    ) -> "Tracked[B]": ...

    @overload
    def __getitem__[U](
        self: "Tracked[Sequence[U]]", index: int
    ) -> "Tracked[U]": ...

    def __getitem__[U](
        self: "Tracked[Sequence[U] | tuple[Any, ...]]", index: int
    ) -> "Tracked[U | Any]":
        return Tracked(
            self.value[index],
            IndexedRef(self.ref, index),
            self.node,
            insp.element_type_of_sequence_type(self.type_annot, index),
        )


type Value = ExtAssembly[Tracked[Any]]
"""
An assembly of *local*, tracked values.

Values can serve as actions or space parameters.
"""


@dataclass(frozen=True)
class LocalityError(Exception):
    """
    Exception raised when the locality invariant is violated.

    See `Tree` and `check_local_value`.
    """

    expected_node_ref: GlobalNodePath
    node_ref: GlobalNodePath
    local_ref: AtomicValueRef


def check_local_value(val: Value, node: GlobalNodePath):
    """
    Raise a `LocalityError` exception if a given value is not a local
    value relative to a given node.
    """
    match val:
        case None:
            pass
        case Sequence():
            for v in val:
                check_local_value(v, node)
        case Tracked():
            if val.node != node:
                raise LocalityError(
                    expected_node_ref=node,
                    node_ref=val.node,
                    local_ref=val.ref,
                )
        case _:
            assert False


def _invalid_value(v: object) -> str:
    if isinstance(v, list):
        return f"Lists are not allowed as values, use tuples instead: {v}"
    return f"Invalid value: {v}"


def value_ref(v: Value) -> ValueRef:
    """
    Obtain a reference from a value.
    """
    match v:
        case Tracked(_, ref):
            return ref
        case None:
            return None
        case Sequence():
            return tuple(value_ref(o) for o in v)
    assert False, _invalid_value(v)


def drop_refs(v: Value) -> object:
    """
    Drop the `Tracked` wrappers within a value.
    """
    match v:
        case Tracked(value):
            return value
        case None:
            return None
        case Sequence():
            return tuple(drop_refs(o) for o in v)
    assert False, _invalid_value(v)


def value_type(v: Value) -> TypeAnnot[Any] | NoTypeInfo:
    """
    Obtain a type annotation for a value, assuming all tracked atoms
    also have type annotations.
    """
    match v:
        case Tracked():
            return v.type_annot
        case None:
            return None
        case Sequence():
            types = [value_type(o) for o in v]
            if any(isinstance(t, NoTypeInfo) for t in types):
                return NoTypeInfo()
            return Sequence[*types]  # type: ignore


#####
##### Utilities
#####


def append_node_path(path: NodePath, v: ValueRef) -> NodePath:
    return (*path, v)


def child_ref(path: GlobalNodePath, action: ValueRef) -> GlobalNodePath:
    assert path
    *init, (space, node_path) = path
    return (*init, (space, (*node_path, action)))


def nested_ref(path: GlobalNodePath, ref: SpaceRef) -> GlobalNodePath:
    return (*path, (ref, ()))


def global_path_origin(
    path: GlobalNodePath,
) -> (
    Literal["global_origin"]
    | tuple[Literal["child"], GlobalNodePath, ValueRef]
    | tuple[Literal["nested"], GlobalNodePath, SpaceRef]
):
    if not path:
        return "global_origin"
    *init, (space, node_path) = path
    if not node_path:
        return "nested", tuple(init), space
    return "child", (*init, (space, node_path[:-1])), node_path[-1]
