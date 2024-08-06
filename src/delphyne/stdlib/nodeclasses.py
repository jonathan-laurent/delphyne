"""
A decorator to ease the declaration of nodes.
"""

import inspect
import types
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAliasType

from delphyne.core.trees import Node, subchoice
from delphyne.stdlib.embedded import EmbeddedSubtree
from delphyne.stdlib.generators import Generator


LABEL_FIELD = "label"


def _is_generic_alias(annot: Any) -> bool:
    # TODO: find a more robust solution
    alt = typing._GenericAlias  # type: ignore
    return isinstance(annot, (types.GenericAlias, alt))


def _type_annot_compatible(annot: Any, target: Any) -> bool:
    """
    Determine if a type annotation is equal or an instantiation of
    another (possibly generic) annotation.

    For example:
        - Generator[P, T], Generator
        - Generator, Generator
    """
    if not (inspect.isclass(target) or isinstance(target, TypeAliasType)):
        return False
    if _is_generic_alias(annot):
        return typing.get_origin(annot) == target
    return annot == target


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


def _supported_atomic_choice(annot: Any) -> bool:
    supported = [Generator, EmbeddedSubtree]
    return any(_type_annot_compatible(annot, s) for s in supported)


def _subchoice_fun_from_annot(
    name: str, esc_name: str, annot: Any
) -> None | tuple[int, Callable[..., Any]]:
    """
    Create a subchoice function from a type annotation.

    Supported:
        - Instances of `Generator` and `EmbeddedSubtree`.
        - Function with value arguments and a supported return type.
    """
    if (cargs := _decompose_callable_annot(annot)) is not None:
        args, ret = cargs
        if not _supported_atomic_choice(ret):
            return None
        # TODO: check args
        n = len(args)
    elif _supported_atomic_choice(annot):
        n = 0
    else:
        return None
    if n == 0:
        f: Any = lambda self: getattr(self, esc_name)  # type: ignore
        f.__name__ = name
        f = property(subchoice(f))
    else:
        f: Any = lambda self, *args: getattr(self, esc_name)(*args)  # type: ignore
        f.__name__ = name
        f = subchoice(f)
    return (n, f)


@typing.dataclass_transform()
def nodeclass[N: Node](
    *, frozen: bool = True
) -> Callable[[type[N]], type[N]]:  # fmt: skip
    """
    A decorator for declaring new node types with no boilerplate.

    TODO: much more validation is needed.
    """

    assert frozen, "Node classes must be frozen."

    def esc(field_name: str):
        return "_" + field_name

    def create_class(cls: type[N]) -> type[N]:
        hints = typing.get_type_hints(cls)
        annots = {esc(f): cls.__annotations__[f] for f in hints.keys()}
        setattr(cls, "__annotations__", annots)
        setattr(cls, "__match_args__", tuple())
        extra_fields: list[str] = []
        subchoices: list[str] = []
        zeroary_choices: list[str] = []
        for f, a in hints.items():
            res = _subchoice_fun_from_annot(f, esc(f), a)
            if res is None:
                # Important: we do not want `prop` to refer to the
                # surrounding `f`, hence the `f=f` trick.
                prop = property(lambda self, f=f: getattr(self, esc(f)))
                setattr(cls, f, prop)
                extra_fields.append(f)
            else:
                n, subchoice_fun = res
                setattr(cls, f, subchoice_fun)
                subchoices.append(f)
                if n == 0:
                    zeroary_choices.append(f)
        setattr(cls, "__extra_fields__", extra_fields)
        setattr(cls, "__subchoices__", subchoices)
        setattr(cls, "__base_choices__", zeroary_choices)

        def base_choices(self: Any):
            return tuple(getattr(self, f) for f in zeroary_choices)

        setattr(cls, "base_choices", base_choices)

        primary_choice_name = zeroary_choices[0] if zeroary_choices else None
        setattr(cls, "__primary_choice__", primary_choice_name)
        if primary_choice_name is not None:

            def primary_choice(self: Any):
                return getattr(self, primary_choice_name)

            setattr(cls, "primary_choice", primary_choice)

        if LABEL_FIELD in extra_fields:

            def get_label(self: Any):
                return getattr(self, LABEL_FIELD)

            setattr(cls, "get_label", get_label)

        return dataclass(frozen=True)(cls)

    return create_class
