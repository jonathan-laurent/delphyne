"""
References to nodes, values, spaces and space elements.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpaceName:
    """
    A name identifying a parametric space.

    This name can feature indices. For example, `subs[0]` denotes the
    first subgoal of a `Join` node.
    """

    name: str
    indices: tuple[int | str, ...]


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
    arg: tuple[ValueRef, ...]


MAIN_SPACE = SpaceRef(SpaceName("$main", ()), ())
"""
The global origin's special space that contains the main, top-level
strategy tree.
"""


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


@dataclass(frozen=True)
class SpaceElementRef:
    """
    A reference to an element of a local space.
    """

    space: SpaceRef
    element: AnswerRef | NodeRef


type GlobalNodePath = tuple[tuple[SpaceRef, NodePath], ...]
"""
Path to a node from the global origin, as a sequence of (space to enter,
path to follow) instruction pairs.
"""
