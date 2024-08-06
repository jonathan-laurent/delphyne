"""
Structured queries.
"""

import re
import typing
from collections.abc import Callable, Sequence
from typing import Any, Never, Protocol, cast

import yaml
import yaml.parser

from delphyne.core import queries as dq
from delphyne.core.queries import Prompt, Query
from delphyne.stdlib.dsl import convert_to_generator
from delphyne.stdlib.generators import Generator, GeneratorAdaptor
from delphyne.stdlib.search_envs import HasSearchEnv
from delphyne.utils.typing import (
    TypeAnnot,
    ValidationError,
    pydantic_dump,
    pydantic_load,
)
from delphyne.utils.yaml import pretty_yaml


class Parser(Protocol):
    # fmt: off
    def __call__[T](
        self, type: TypeAnnot[T], res: str, /
    ) -> T | dq.ParseError: ...
    # fmt: on


class StructuredQuery[P: HasSearchEnv, T](Query[P, T]):

    # Methods to overload

    def system_message(self, params: P) -> str:
        message = params.env.jinja.prompt("system", self.name(), self)
        assert (
            message is not None
        ), f"Missing system message for {self.name()}."
        return message

    def instance_message(self, params: P) -> str:
        message = params.env.jinja.prompt("instance", self.name(), self)
        if message is None:
            return pretty_yaml(self.serialize_args())
        return message

    def parser(self) -> Parser:
        return raw_yaml

    def options(self, params: P) -> dq.PromptOptions:
        return dq.PromptOptions()

    # Other methods

    def name(self) -> str:
        return type(self).__name__

    def serialize_args(self) -> dict[str, object]:
        return cast(dict[str, object], pydantic_dump(type(self), self))

    def parse_answer(self, res: str) -> T | dq.ParseError:
        return self.parser()(self.return_type(), res)

    def create_prompt[Q](
        self, params: P, examples: Sequence[tuple[Q, str]]
    ) -> Prompt:  # fmt: skip
        # TODO: handle examples
        messages: list[dq.Message] = []
        messages.append(dq.Message("system", self.system_message(params)))
        for q, answer in examples:
            # TODO: should we add an `instance_message` method to `Query`?
            assert isinstance(q, StructuredQuery)
            q = cast(StructuredQuery[Any, Any], q)
            messages.append(dq.Message("user", q.instance_message(params)))
            messages.append(dq.Message("assistant", answer))
        messages.append(dq.Message("user", self.instance_message(params)))
        return Prompt(messages, self.options(params))

    def return_type(self) -> TypeAnnot[T]:
        base = type(self).__orig_bases__[0]  # type: ignore
        return typing.get_args(base)[1]

    def param_type(self) -> TypeAnnot[P]:
        base = type(self).__orig_bases__[0]  # type: ignore
        return typing.get_args(base)[0]

    @classmethod
    def parse(
        cls, args: dict[str, object]
    ) -> "StructuredQuery[Never, object]":
        return pydantic_load(cls, args)

    def using[Q](self, adapter: Callable[[Q], P]) -> Generator[Q, T]:
        gen = convert_to_generator(self)
        return GeneratorAdaptor(gen, adapter)


def raw_yaml[T](type: TypeAnnot[T], res: str) -> T | dq.ParseError:
    try:
        parsed = yaml.safe_load(res)
        return pydantic_load(type, parsed)
    except ValidationError as e:
        return dq.ParseError(str(e))
    except yaml.parser.ParserError as e:
        return dq.ParseError(str(e))


def yaml_from_last_block[T](type: TypeAnnot[T], res: str) -> T | dq.ParseError:
    final = extract_final_block(res)
    if final is None:
        return dq.ParseError("No final code block found.")
    return raw_yaml(type, final)


def raw_string[T](typ: TypeAnnot[T], res: str) -> T | dq.ParseError:
    if isinstance(typ, type):  # if `typ` is a class
        return typ(res)  # type: ignore
    return res  # type: ignore  # TODO: assuming that `typ` is a string alias


def trimmed_raw_string[T](typ: TypeAnnot[T], res: str) -> T | dq.ParseError:
    return raw_string(typ, res.strip())


def string_from_last_block[T](
    typ: TypeAnnot[T], res: str
) -> T | dq.ParseError:  # fmt: skip
    final = extract_final_block(res)
    if final is None:
        return dq.ParseError("No final code block found.")
    return raw_string(typ, final)


def trimmed_string_from_last_block[T](
    typ: TypeAnnot[T], res: str
) -> T | dq.ParseError:  # fmt: skip
    final = extract_final_block(res)
    if final is None:
        return dq.ParseError("No final code block found.")
    return trimmed_raw_string(typ, final)


def extract_final_block(s: str) -> str | None:
    code_blocks = re.findall(r"```[^\n]*\n(.*?)```", s, re.DOTALL)
    return code_blocks[-1] if code_blocks else None
