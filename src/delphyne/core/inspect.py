"""
Common inspection utilities.
"""

import functools
import inspect
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from delphyne.core.trees import Strategy
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


class FunctionWrapper[**P, T](ABC):
    """
    Libraries may wrap strategy functions in various way. Having
    wrappers inherit this type allows removing them when needed. Wrapped
    functions must use `functools.wraps` to be compatible with
    `inspect.signature`.

    See `remove_wrappers`.
    """

    @abstractmethod
    def wrapped(self) -> Callable[[], Any]:
        pass

    @abstractmethod
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        pass


def remove_wrappers(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Remove all layers of wrappers from a function.
    """
    while isinstance(f, (FunctionWrapper, functools.partial)):
        f = f.wrapped() if isinstance(f, FunctionWrapper) else f.func
    return f


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


@dataclass
class FunctionCallArgs:
    args: list[Any]
    kwargs: dict[str, Any]

    def to_dict(self, sig: inspect.Signature) -> dict[str, Any]:
        """
        Force fitting all arguments into a single dictionary, using a
        signature to name positional arguments.
        """
        ret: dict[str, Any] = {}
        for a, p in zip(self.args, sig.parameters.keys()):
            ret[p] = a
        return ret | self.kwargs


def instantiated_args(f: Callable[..., Any]) -> FunctionCallArgs:
    """
    Go through a stack of wrappers to gather the arguments of a
    partially instantiated functions, assuming that all instantiated
    arguments are specified using `functools.partial`.
    """
    if isinstance(f, FunctionWrapper):
        return instantiated_args(f.wrapped())
    elif isinstance(f, functools.partial):
        base = instantiated_args(f.func)
        base.args += f.args
        base.kwargs.update(f.keywords)
        return base
    else:
        return FunctionCallArgs([], {})


def instantiated_args_dict(f: Callable[..., Any]) -> dict[str, Any]:
    args = instantiated_args(f)
    sig = inspect.signature(remove_wrappers(f))
    return args.to_dict(sig)


def return_type_of_strategy_type[T](
    st: TypeAnnot[Strategy[Any, T]]
) -> TypeAnnot[T] | NoTypeInfo:  # fmt: skip
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


def param_type_of_strategy_type[T](st: Any) -> TypeAnnot[T] | NoTypeInfo:
    """
    If `st` is a type alias with two parameters exactly and whose first
    one is named `P`, then return the value of `P`.
    """
    assert isinstance(st, types.GenericAlias)
    st_origin = typing.get_origin(st)
    assert isinstance(st_origin, typing.TypeAliasType)
    assert len(st_origin.__type_params__) == 2
    assert st_origin.__type_params__[0].__name__ == "P"
    return typing.get_args(st)[0]


def underlying_strategy_name(f: Callable[..., Any]) -> str | None:
    """
    Return the name of a possibly wrapped strategy function.
    """
    return function_name(remove_wrappers(f))


def underlying_strategy_return_type(
    f: Callable[..., Any]
) -> TypeAnnot[Any] | NoTypeInfo:
    """
    Inspect the underlying return type of a possibly wrapped strategy
    function. The return type is supposed to be annotated, with an
    annotation of the form `Foo[..., T]`, where `Foo` is either
    `Strategy` or a type alias.
    """
    st = function_return_type(remove_wrappers(f))
    if isinstance(st, NoTypeInfo):
        return st
    return return_type_of_strategy_type(cast(Any, st))


def underlying_strategy_param_type(
    f: Callable[..., Any]
) -> TypeAnnot[Any] | NoTypeInfo:
    st = function_return_type(remove_wrappers(f))
    if isinstance(st, NoTypeInfo):
        return st
    return param_type_of_strategy_type(cast(Any, st))
