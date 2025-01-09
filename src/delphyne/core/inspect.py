"""
Inspection utilities.s
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
