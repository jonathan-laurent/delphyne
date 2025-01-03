"""
Internal utilities to detect node fields.
"""

import inspect
import types
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass, is_dataclass
from typing import Any

####
#### Fields
####


@dataclass
class SpaceF:
    """
    A space field that does not correspond to am embedded nested tree.
    """


@dataclass
class EmbeddedF:
    """
    A field corresponding to a nested embedded space.
    """


@dataclass
class DataF:
    """
    A field that corresponds to some other kind of data.
    """


@dataclass
class ParametricF:
    """
    A field corresponding to a parametric space.
    """

    res: SpaceF | EmbeddedF


@dataclass
class SequenceF:
    """
    A field corresponding to a sequence of spaces.
    """

    element: "FieldType"


@dataclass
class OptionalF:
    """
    A field corresponding to a sequence of spaces.
    """

    element: "FieldType"


type LeafFieldType = SpaceF | EmbeddedF | DataF | ParametricF


type FieldType = LeafFieldType | SequenceF | OptionalF


type NodeFields = dict[str, FieldType]


####
#### Inference Utilities
####


def _is_generic_alias(annot: Any) -> bool:
    """
    Determine if an object is a type annotation of the form `A[...]`.
    Then, `typing.get_origin` and `typing.get_args` can be used to
    recover the arguments.
    """
    # TODO: find a more robust solution
    alt = typing._GenericAlias  # type: ignore
    return isinstance(annot, (types.GenericAlias, alt))


def _type_annot_compatible(annot: Any, target: Any) -> bool:
    """
    Determine if a type annotation is equal or an instantiation of
    another (possibly generic) annotation. Subtypes are accepted.

    For example:
        - OpaqueSpace[Any], Space
        - OpaqueSpace, OpaqueSpace

    TODO: handle type aliases such as `type X[T] = OpaqueSpace[T]`.
    """
    if not inspect.isclass(target):
        return False
    if _is_generic_alias(annot):
        annot = typing.get_origin(annot)
    return inspect.isclass(annot) and issubclass(annot, target)


def _decompose_callable_annot(annot: Any) -> None | tuple[Sequence[Any], Any]:
    """
    Decompose a type annotation of the form `Callable[[X1,X2,...], T]`.

    TODO: this function only works with `collections.abc.Callable` and
    not with `typing.Callable`.
    """
    if not _is_generic_alias(annot):
        return None
    if not typing.get_origin(annot) == Callable:  # type: ignore
        return None
    return typing.get_args(annot)  # type: ignore


def _decompose_optional_annot(annot: Any) -> None | tuple[Any]:
    """
    Decompose a type annotation of the form `T | None` or `None | T`.
    TODO: handle `typing.Optional`.
    """
    if not isinstance(annot, types.UnionType):
        return None
    args = typing.get_args(annot)
    if args[0] == types.NoneType:
        return (args[1],)
    if args[1] == types.NoneType:
        return (args[0],)
    return None


def _decompose_sequence_annot(annot: Any) -> None | tuple[Any]:
    """
    Decompose a type annotation of the form `Sequence[T]`.
    """
    if not _is_generic_alias(annot):
        return None
    if not typing.get_origin(annot) == Sequence:  # type: ignore
        return None
    return typing.get_args(annot)  # type: ignore


####
#### Autodetecting field types
####


class _Unrecognized(Exception):
    pass


@dataclass
class _SpecialClasses:
    space: type[Any]
    embedded: type[Any]


def _field_leaf_structure(
    annot: Any, classes: _SpecialClasses
) -> SpaceF | EmbeddedF | DataF:
    if _type_annot_compatible(annot, classes.embedded):
        return EmbeddedF()
    if _type_annot_compatible(annot, classes.space):
        return SpaceF()
    return DataF()


def _field_structure(annot: Any, classes: _SpecialClasses) -> FieldType:
    """
    TODO: detect `Tracked more robustly`.
    """
    if (opt := _decompose_optional_annot(annot)) is not None:
        return OptionalF(_field_leaf_structure(opt[0], classes))
    if (seq := _decompose_sequence_annot(annot)) is not None:
        return SequenceF(_field_leaf_structure(seq[0], classes))
    if (call := _decompose_callable_annot(annot)) is not None:
        elt = _field_leaf_structure(call[1], classes)
        if not isinstance(elt, DataF):
            return ParametricF(elt)
    return DataF()


def detect_node_structure(
    cls: Any, *, space_class: type[Any], embedded_class: type[Any]
) -> NodeFields | None:
    """
    Automatically detect the structure of a node.
    """
    if not is_dataclass(cls):
        return None
    res: dict[str, Any] = {}
    classes = _SpecialClasses(space=space_class, embedded=embedded_class)
    try:
        for f, annot in typing.get_type_hints(cls).items():
            ft = _field_structure(annot, classes)
            res[f] = ft
        return res
    except _Unrecognized:
        return None
