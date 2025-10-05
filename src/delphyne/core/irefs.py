"""
Id-Based References.

These are shorter references, where query answers and success values are
associated unique identifiers. This concise format is used for exporting
traces (see `Trace`).
"""

from dataclasses import dataclass

from delphyne.core.refs import Assembly, SpaceName, show_assembly

type AtomicValueRef = IndexedRef | SpaceElementRef
"""
An atomic value reference is a space element reference that is indexed
zero or a finite number of times: `space_elt_ref[i1][i2]...[in]`.
"""


@dataclass(frozen=True)
class NodeId:
    """
    Global identifier of a node within a trace.
    """

    id: int

    def __str__(self) -> str:
        return f"%{self.id}"


@dataclass(frozen=True)
class AnswerId:
    """
    The identifier to an `Answer` object stored within a trace.
    """

    id: int

    def __str__(self) -> str:
        return f"@{self.id}"


@dataclass(frozen=True)
class IndexedRef:
    """
    Indexing an atomic value reference.
    """

    ref: AtomicValueRef
    index: int

    def __str__(self) -> str:
        return f"{self.ref}[{self.index}]"


type ValueRef = Assembly[AtomicValueRef]


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

    def __str__(self) -> str:
        name = str(self.name)
        if not self.args:
            return name
        args_str = ", ".join(show_value_ref(a) for a in self.args)
        return f"{name}({args_str})"


@dataclass(frozen=True)
class SpaceElementRef:
    """
    A reference to an element of a local space.
    """

    space: SpaceRef
    element: AnswerId | NodeId

    def __str__(self) -> str:
        return f"{self.space}{{{self.element}}}"


@dataclass(frozen=True)
class GlobalSpaceRef:
    node: NodeId
    ref: SpaceRef


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

    def __str__(self) -> str:
        return f"child({self.node}, {show_value_ref(self.action)})"


@dataclass(frozen=True)
class NestedTreeOf:
    """
    The tree of interest is the root of a tree that induces a given
    space.
    """

    node: NodeId
    space: SpaceRef

    def __str__(self) -> str:
        return f"nested({self.node}, {self.space})"


def show_value_ref(vr: ValueRef) -> str:
    return show_assembly(str, vr)
