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

References in this module are *full references*, which are produced by
`reify`. Query answers are stored as strings and elements of spaces
induced by strategies are denoted by sequences of value references.

See modules `irefs` and `hrefs` for two alternative kinds of references:
id-based references and hint-based references.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast, overload

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

    def __str__(self) -> str:
        ret = self.name
        for i in self.indices:
            ret += f"[{i}]"
        return ret


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


@dataclass(frozen=True)
class NodePath:
    """
    Encodes a sequence of actions leading to a node with respect to a
    given root.
    """

    actions: tuple[ValueRef, ...]

    def append(self, action: ValueRef) -> "NodePath":
        return NodePath((*self.actions, action))


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


@dataclass(frozen=True)
class GlobalSpacePath:
    """
    A path to a global space, which alternates between following a path from
    a local root and entering a space within the reached node.
    """

    steps: tuple[tuple[NodePath, SpaceRef], ...]

    def append(self, path: NodePath, space: SpaceRef) -> "GlobalSpacePath":
        return GlobalSpacePath((*self.steps, (path, space)))

    def split(self) -> "tuple[GlobalNodeRef, SpaceRef] | tuple[None, None]":
        if not self.steps:
            return (None, None)
        last_path, last_space = self.steps[-1]
        parent_steps = self.steps[:-1]
        gsref = GlobalNodeRef(GlobalSpacePath(parent_steps), last_path)
        return (gsref, last_space)

    def parent_node(self) -> "GlobalNodeRef | None":
        return self.split()[0]

    def local_ref(self) -> "SpaceRef | None":
        return self.split()[1]


@dataclass(frozen=True)
class SpaceElementRef:
    """
    A reference to an element of a local space.

    Attributes:
        space: The space containing the element, or `None` if this is
            the top-level main space.
        element: The element pointer.
    """

    space: SpaceRef | None
    element: Answer | NodePath


@dataclass(frozen=True)
class GlobalNodeRef:
    """
    Global reference to a node.
    """

    space: GlobalSpacePath
    path: NodePath

    def child(self, action: ValueRef) -> "GlobalNodeRef":
        return GlobalNodeRef(self.space, self.path.append(action))

    def nested_space(self, space: SpaceRef) -> "GlobalSpacePath":
        return GlobalSpacePath((*self.space.steps, (self.path, space)))

    def nested_tree(self, space: SpaceRef) -> "GlobalNodeRef":
        return GlobalNodeRef(self.nested_space(space), NodePath(()))


type GlobalAnswerRef = tuple[GlobalSpacePath, Answer]
"""
A global reference to located answer.
"""


type GlobalActionRef = tuple[GlobalNodeRef, ValueRef]
"""
A global reference to a located action.
"""


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
        ref: A global reference to the space that the value belongs to.
        node: A reference to the node that the value is local to, or
            `None` if the value is a top-level result.
        type_annot: An optional type annotation for the `value` field.
            This is mostly used for improving the rendering of values
            when exporting trace information for external tools.

    Tracked sequences (or pairs) can be indexed using `__getitem__`,
    resulting in tracked values with `IndexedRef` references. Since
    `__getitem__` is defined, tracked values are also iterable.
    """

    value: T
    ref: AtomicValueRef
    node: GlobalNodeRef | None
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
        self: "Tracked[Sequence[U]] | Tracked[tuple[Any, ...]]", index: int
    ) -> "Tracked[U | Any]":
        return Tracked(
            self.value[index],
            IndexedRef(self.ref, index),
            self.node,
            # TODO: will not work well for union of tuples for example
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

    expected_node_ref: GlobalNodeRef | None
    node_ref: GlobalNodeRef | None
    local_ref: AtomicValueRef


def check_local_value(val: Value, node: GlobalNodeRef | None):
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


NONE_REF_REPR = "nil"


def show_assembly[T](show_element: Callable[[T], str], a: Assembly[T]) -> str:
    """
    Print an assembly, assuming that T does not intersect with tuple.
    """
    if isinstance(a, tuple):
        a = cast(tuple[Assembly[T], ...], a)
        return "[" + ", ".join(show_assembly(show_element, x) for x in a) + "]"
    elif a is None:
        return NONE_REF_REPR
    else:
        return show_element(a)
