"""
Inspection utilities.
"""

import inspect
import types
import typing
from collections.abc import Callable, Sequence
from typing import Any

from delphyne.utils.typing import NoTypeInfo, TypeAnnot


def function_name(f: Callable[..., Any]) -> str | None:
    """
    Return the name of a function.
    """
    if hasattr(f, "__name__"):
        return f.__name__
    return None


def function_return_type[T](f: Callable[..., T]) -> TypeAnnot[T] | NoTypeInfo:
    """
    Use the type annotations of a function to determine its return type.
    """
    try:
        hints = typing.get_type_hints(f)
        return hints["return"]
    except Exception:
        return NoTypeInfo()


def return_type_of_strategy_type[T](st: Any) -> TypeAnnot[T] | NoTypeInfo:
    """
    It is assumed that the argument is the instantiation of a generic
    type alias whose last argument is named T and is interpreted as the
    strategy return type.
    """
    assert isinstance(st, types.GenericAlias)
    st_origin = typing.get_origin(st)
    assert isinstance(st_origin, typing.TypeAliasType)
    assert len(st_origin.__type_params__) > 0
    assert st_origin.__type_params__[-1].__name__ == "T"
    return typing.get_args(st)[-1]


def inner_policy_type_of_strategy_type[T](
    st: Any,
) -> TypeAnnot[T] | NoTypeInfo:
    """
    If `st` is a type alias with two parameters exactly and whose middle
    one is named `P`, then return the value of `P`.
    """
    assert isinstance(st, types.GenericAlias)
    st_origin = typing.get_origin(st)
    assert isinstance(st_origin, typing.TypeAliasType)
    assert len(st_origin.__type_params__) == 3
    assert st_origin.__type_params__[1].__name__ == "P"
    return typing.get_args(st)[1]


def first_parameter_of_base_class(cls: Any) -> Any:
    base = cls.__orig_bases__[0]  # type: ignore
    return typing.get_args(base)[0]


def function_args_dict(
    f: Callable[..., Any], args: Sequence[Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """
    Force fitting all arguments into a single dictionary, using a
    signature to name positional arguments.
    """
    sig = inspect.signature(f)
    ret: dict[str, Any] = {}
    for a, p in zip(args, sig.parameters.keys()):
        ret[p] = a
    return ret | kwargs


def element_type_of_sequence_type(
    seq_type: Any, i: int = 0
) -> Any | NoTypeInfo:
    """
    Return the type of the element obtained by indexing an element of
    another type.

    If `seq_type` is `list[T]`, `Sequence[T]` or `tuple[T, ...]`, return
    `T`. If `seq_type` is `tuple[T_1, ..., T_n]`, then return `T_i`.
    """
    origin = typing.get_origin(seq_type)
    args = typing.get_args(seq_type)

    # Handle List[T] or Sequence[T]
    if origin in {list, Sequence}:
        if args:
            return args[0]
        return NoTypeInfo

    # Handle tuple[T, ...] or `tuple[T_1, ..., T_n]`
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        elif len(args) > i:
            return args[i]
        return NoTypeInfo

    return NoTypeInfo


def is_sequence_type(typ: Any) -> bool:
    """
    Determine whether a type is of the form `list[T]`, `Sequence[T]` or
    `tuple[T, ...]`.
    """
    origin = typing.get_origin(typ)
    if origin is tuple:
        args = typing.get_args(typ)
        return len(args) == 2 and args[1] is Ellipsis
    return origin in {list, Sequence}


def union_components(typ: Any) -> Sequence[Any]:
    """
    Take a type of the form `Union[T_1, ..., T_n]` or `T_1 | ... | T_n`
    and return the `T_i` sequence. If the type is not a union, returns a
    singleton with the type itself.
    """
    if typ == typing.Never:
        return []
    if typing.get_origin(typ) in [typing.Union, types.UnionType]:
        return typing.get_args(typ)
    return [typ]


def make_union(comps: Sequence[Any]) -> Any:
    """
    Take a sequence of types and return a union type.
    """
    if len(comps) == 0:
        return typing.Never
    if len(comps) == 1:
        return comps[0]
    return typing.Union[*comps]
