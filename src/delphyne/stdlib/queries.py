"""
Standard queries and building blocks for prompting policies.
"""

import re
from collections.abc import Callable
from typing import Self, cast

import yaml
import yaml.parser

from delphyne.core.environment import TemplatesManager
from delphyne.core.inspect import first_parameter_of_base_class
from delphyne.core.queries import AbstractQuery, AnswerModeName, ParseError
from delphyne.core.trees import Builder, OpaqueSpace, PromptingPolicy
from delphyne.utils import typing as dpty
from delphyne.utils.typing import TypeAnnot, ValidationError

#####
##### Standard Queries
#####


class Query[T](AbstractQuery[T]):
    def system_prompt(
        self,
        mode: AnswerModeName,
        params: dict[str, object],
        env: TemplatesManager | None = None,
    ) -> str:
        """
        Raises `TemplateNotFound`.
        """
        assert env is not None, _no_prompt_manager_error()
        args = {"query": self, "mode": mode, "params": params}
        return env.prompt("system", self.name(), args)

    def instance_prompt(
        self,
        mode: AnswerModeName,
        params: dict[str, object],
        env: TemplatesManager | None = None,
    ) -> str:
        """
        Raises `TemplateNotFound`.
        """
        assert env is not None, _no_prompt_manager_error()
        args = {"query": self, "mode": mode, "params": params}
        return env.prompt("instance", self.name(), args)

    def serialize_args(self) -> dict[str, object]:
        return cast(dict[str, object], dpty.pydantic_dump(type(self), self))

    @classmethod
    def parse(cls, args: dict[str, object]) -> Self:
        return dpty.pydantic_load(cls, args)

    def answer_type(self) -> TypeAnnot[T]:
        return first_parameter_of_base_class(type(self))

    def __getitem__[P](
        self, get_policy: Callable[[P], PromptingPolicy]
    ) -> Builder[OpaqueSpace[P, T]]:
        return OpaqueSpace[P, T].from_query(self, get_policy)


def _no_prompt_manager_error() -> str:
    return (
        "Please provide an explicit prompt manager "
        + " or override the `system_prompt` and `instance_prompt` functions."
    )


#####
##### Parsers
#####


def raw_yaml[T](type: TypeAnnot[T], res: str) -> T | ParseError:
    try:
        parsed = yaml.safe_load(res)
        return dpty.pydantic_load(type, parsed)
    except ValidationError as e:
        return ParseError(str(e))
    except yaml.parser.ParserError as e:
        return ParseError(str(e))


def yaml_from_last_block[T](type: TypeAnnot[T], res: str) -> T | ParseError:
    final = extract_final_block(res)
    if final is None:
        return ParseError("No final code block found.")
    return raw_yaml(type, final)


def raw_string[T](typ: TypeAnnot[T], res: str) -> T | ParseError:
    if isinstance(typ, type):  # if `typ` is a class
        return typ(res)  # type: ignore
    return res  # type: ignore  # TODO: assuming that `typ` is a string alias


def trimmed_raw_string[T](typ: TypeAnnot[T], res: str) -> T | ParseError:
    return raw_string(typ, res.strip())


def string_from_last_block[T](
    typ: TypeAnnot[T], res: str
) -> T | ParseError:  # fmt: skip
    final = extract_final_block(res)
    if final is None:
        return ParseError("No final code block found.")
    return raw_string(typ, final)


def trimmed_string_from_last_block[T](
    typ: TypeAnnot[T], res: str
) -> T | ParseError:  # fmt: skip
    final = extract_final_block(res)
    if final is None:
        return ParseError("No final code block found.")
    return trimmed_raw_string(typ, final)


def extract_final_block(s: str) -> str | None:
    code_blocks = re.findall(r"```[^\n]*\n(.*?)```", s, re.DOTALL)
    return code_blocks[-1] if code_blocks else None


#####
##### Prompting Policies
#####
