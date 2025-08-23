"""
We use pyaml to dump YAML with a more human-readable format.
"""

# pyaml has no stubs
# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false

from collections.abc import Iterable
from typing import Any, cast

import pydantic
import yaml

from delphyne.utils.pretty_yaml import pretty_yaml


def dump_yaml[T](
    type: type[T] | Any,
    obj: T,
    *,
    exclude_defaults: bool = False,
    exclude_none: bool = False,
    exclude_fields: Iterable[str] = (),
) -> str:
    """
    Pretty-print a value in Yaml.

    We allow `type` to be `Any` because pyright does not recognize
    unions as being member of `type`.
    """
    Adapter = pydantic.TypeAdapter(type)
    py = Adapter.dump_python(
        obj,
        exclude_defaults=exclude_defaults,
        exclude_none=exclude_none,
        warnings="error",
    )
    if isinstance(py, dict):
        py = cast(dict[Any, Any], py)
        for f in exclude_fields:
            del py[f]
    return pretty_yaml(py)


def dump_yaml_object(
    obj: object,
    *,
    exclude_defaults: bool = False,
    exclude_none: bool = False,
    exclude_fields: Iterable[str] = (),
) -> str:
    return dump_yaml(
        type(obj),
        obj,
        exclude_defaults=exclude_defaults,
        exclude_none=exclude_none,
        exclude_fields=exclude_fields,
    )


def load_yaml[T](type: type[T], s: str) -> T:
    """
    Raises ValidationError.
    """
    Adapter = pydantic.TypeAdapter(type)
    return Adapter.validate_python(yaml.load(s, yaml.CLoader))
