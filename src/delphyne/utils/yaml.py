"""
We use pyaml to dump YAML with a more human-readable format.
"""

# pyaml has no stubs
# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false

from typing import Any, cast

import pyaml
import pydantic
import yaml


def pretty_yaml(obj: object) -> str:
    return cast(Any, pyaml.dump(obj, sort_dicts=pyaml.PYAMLSort.none))


def dump_yaml[T](type: type[T] | Any, obj: T) -> str:
    """
    Debugging utility to dump a response to YAML.

    We allow `type` to be `Any` because pyright does not recognize
    unions as being member of `type`.
    """
    Adapter = pydantic.TypeAdapter(type)
    py = Adapter.dump_python(obj)
    return pretty_yaml(py)


def load_yaml[T](type: type[T], s: str) -> T:
    """
    Raises ValidationError.
    """
    Adapter = pydantic.TypeAdapter(type)
    return Adapter.validate_python(yaml.load(s, yaml.CLoader))
