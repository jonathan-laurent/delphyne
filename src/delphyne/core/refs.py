"""
References to nodes, values, spaces and space elements.
"""

from dataclasses import dataclass
from typing import Any, Literal

from delphyne.utils.typing import NoTypeInfo, TypeAnnot


@dataclass(frozen=True)
class SpaceName:
    """
    A name identifying a parametric space.

    This name can feature indices. For example, `subs[0]` denotes the
    first subgoal of a `Join` node.
    """

    name: str
    indices: tuple[int, ...]

    def __getitem__(self, index: int) -> "SpaceName":
        return SpaceName(self.name, (*self.indices, index))


type AnswerModeName = str | None
"""
A name for an answer mode, or `None` for the default mode.
"""


type ValueRef = Assembly[SpaceElementRef]
"""
A reference to a local value, which is obtained by combining elements of
(possibly multiple) local spaces.
"""


type Assembly[T] = T | tuple[Assembly[T], ...]
"""
An S-expression whose atoms have type `T`.
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
an exportable trace, only identifiers are used while basic strategy
trees do not use identifiers and use paths instead).
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
The global origin's special space that contains the main, top-level
strategy tree.
"""

MAIN_ROOT: "GlobalNodePath" = ((MAIN_SPACE, ()),)


@dataclass(frozen=True)
class Answer:
    """
    An answer to a query. This can serve as a _space element reference_
    if the space in question is a query and the proposed answer
    correctly parses.
    """

    mode: AnswerModeName
    text: str


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

    The hint can be conditioned on a specific query.
    """

    query_name: str | None
    hint: HintValue


@dataclass(frozen=True)
class Hints:
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
    element: AnswerRef | NodeRef | Hints


type GlobalNodePath = tuple[tuple[SpaceRef, NodePath], ...]
"""
Path to a node from the global origin, as a sequence of (space to enter,
path to follow) instruction pairs.
"""


type GlobalNodeRef = NodeId | GlobalNodePath
"""
A global node reference, either as an ID or as a global path.
"""


type GlobalSpaceRef = tuple[GlobalNodeRef, SpaceRef]
"""
A global space reference.
"""


#####
##### Node Origins
#####


type NodeOrigin = ChildOf | NestedTreeOf


@dataclass(frozen=True)
class ChildOf:
    node: NodeId
    action: ValueRef


@dataclass(frozen=True)
class NestedTreeOf:
    node: NodeId
    space: SpaceRef


#####
##### Tracked Values
#####


@dataclass(frozen=True)
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


def value_ref(v: Value) -> ValueRef:
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
        return "nested", (*init, (space, ())), space
    return "child", (*init, (space, node_path[:-1])), node_path[-1]
