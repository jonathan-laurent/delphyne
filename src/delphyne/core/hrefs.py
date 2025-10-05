"""
Hint-Based References.

Query answers and success values are identified by sequences of *hints*.
This format is used in the demonstration language (e.g. argument of test
instruction `go compare(['', 'foo bar'])`) and when visualizing traces
resulting from demonstrations.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import delphyne.core.refs as refs
from delphyne.core.refs import Assembly, SpaceName

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

    def __str__(self) -> str:
        if not self.qualifier:
            return self.hint
        return f"{self.qualifier}:{self.hint}"


@dataclass(frozen=True)
class SpaceElementRef:
    """
    A reference to an element of a local space.

    When the `space` field is `None`, the primary field is considered
    instead (if it exists).
    """

    space: SpaceRef | None
    element: tuple[Hint, ...]

    def __str__(self) -> str:
        hints = "'" + " ".join(str(h) for h in self.element) + "'"
        if self.space is None:
            return hints
        else:
            return f"{self.space}{{{hints}}}"


def show_hints(hs: Sequence[Hint]) -> str:
    return "'" + " ".join(str(h) for h in hs) + "'"


def show_value_ref(vr: ValueRef) -> str:
    return refs.show_assembly(str, vr)


def _convert_trivial_atomic_value_ref(
    avr: refs.AtomicValueRef,
) -> AtomicValueRef:
    if isinstance(avr, refs.IndexedRef):
        ref = _convert_trivial_atomic_value_ref(avr.ref)
        return IndexedRef(ref=ref, index=avr.index)
    else:
        raise ValueError("Attempted to convert non-trivial reference.")


def _convert_trivial_value_ref(vr: refs.ValueRef) -> ValueRef:
    match vr:
        case None:
            return None
        case tuple():
            return tuple(_convert_trivial_value_ref(v) for v in vr)
        case _:
            return _convert_trivial_atomic_value_ref(vr)


def convert_trivial_space_ref(sr: refs.SpaceRef) -> SpaceRef:
    """
    Convert a full space reference that does not contain any space
    element reference into a hint-based space reference.

    This is useful for manipulating primary spaces.

    Raises:
        ValueError: if `sr` contains a space element reference.
    """
    return SpaceRef(
        name=sr.name,
        args=tuple(_convert_trivial_value_ref(a) for a in sr.args),
    )
