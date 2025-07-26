import typing
from collections.abc import Callable, Sequence
from typing import Any, cast

import pydantic

type TypeAnnot[T] = type[T] | object
"""
Type for a type annotation denoting type T.

Ideally, `type[T]` should be used instead. However, we want to allow
type union and type aliases.

```
type T = int
U: type[T] = T  # error
V: type[int] = int  # ok
```
"""


class NoTypeInfo:
    """
    Sentinel type for when no type annotation is available.
    """

    pass


def pydantic_dump[T](
    type: TypeAnnot[T] | NoTypeInfo, x: T, *, exclude_defaults: bool = True
) -> object:
    if isinstance(type, NoTypeInfo):
        assert valid_json_object(object), (
            "Unable to serialize a non-JSON object "
            + f"without a type annotation: {object}"
        )
        return object
    adapter = pydantic.TypeAdapter[T](type)
    return adapter.dump_python(
        x, exclude_defaults=exclude_defaults, warnings="error"
    )


ValidationError = pydantic.ValidationError


def pydantic_load[T](type: TypeAnnot[T] | NoTypeInfo, s: object) -> T:
    """
    Raises ValidationError.
    """
    if isinstance(type, NoTypeInfo):
        type = Any
    adapter = pydantic.TypeAdapter[T](type)
    return adapter.validate_python(s)


def parse_function_args(
    f: Callable[..., Any], args: dict[str, Any]
) -> dict[str, Any]:
    hints = typing.get_type_hints(f)
    pargs: dict[str, Any] = {}
    for k in args:
        hint = hints.get(k, NoTypeInfo())
        pargs[k] = pydantic_load(hint, args[k])
    return pargs


def valid_json_object(obj: object) -> bool:
    match obj:
        case int() | float() | str() | bool() | None:
            return True
        case dict():
            obj = cast(dict[object, object], obj)
            return all(
                isinstance(k, str) and valid_json_object(v)
                for k, v in obj.items()
            )
        case tuple() | list():
            obj = cast(Sequence[object], obj)
            return all(valid_json_object(v) for v in obj)
        case _:
            return False
