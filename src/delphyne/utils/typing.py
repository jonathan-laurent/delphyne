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
    type: TypeAnnot[T], x: T, *, exclude_defaults: bool = True
) -> object:  # fmt: skip
    adapter = pydantic.TypeAdapter[T](type)
    return adapter.dump_python(x, exclude_defaults=exclude_defaults)


ValidationError = pydantic.ValidationError


def pydantic_load[T](type: TypeAnnot[T], s: object) -> T:
    """
    Raises ValidationError.
    """
    adapter = pydantic.TypeAdapter[T](type)
    return adapter.validate_python(s)
