"""
Delphyne Queries.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self, cast

import delphyne.utils.typing as ty
from delphyne.core.chats import AnswerPrefix
from delphyne.core.environments import TemplatesManager
from delphyne.core.errors import Error
from delphyne.core.refs import Answer, AnswerModeName


@dataclass
class ParseError(Error, Exception):
    """
    Can be used as an exception or a returned value.
    """

    def __init__(
        self,
        *,
        label: str | None = None,
        description: str | None = None,
        meta: Any | None = None,
    ):
        if label is None:
            label = "parse_error"
        super().__init__(label=label, description=description, meta=meta)


@dataclass(frozen=True)
class StructuredOutputSettings:
    type: ty.TypeAnnot[Any] | ty.NoTypeInfo


@dataclass(frozen=True)
class ToolSettings:
    tool_types: Sequence[type[Any]]
    force_tool_call: bool


@dataclass(frozen=True)
class QuerySettings:
    structured_output: StructuredOutputSettings | None = None
    tools: ToolSettings | None = None


type AnyQuery = AbstractQuery[Any]


class AbstractQuery[T](ABC):
    @abstractmethod
    def generate_prompt(
        self,
        kind: Literal["system", "instance", "feedback"] | str,
        mode: AnswerModeName,
        params: dict[str, object],
        env: TemplatesManager | None,
    ) -> str:
        pass

    def serialize_args(self) -> dict[str, object]:
        return cast(dict[str, object], ty.pydantic_dump(type(self), self))

    @classmethod
    def parse_instance(cls, args: dict[str, object]) -> Self:
        return ty.pydantic_load(cls, args)

    @abstractmethod
    def answer_type(self) -> ty.TypeAnnot[T] | ty.NoTypeInfo:
        pass

    @abstractmethod
    def query_modes(self) -> Sequence[AnswerModeName]:
        pass

    def query_prefix(self) -> AnswerPrefix | None:
        return None

    def query_settings(self, mode: AnswerModeName) -> QuerySettings:
        return QuerySettings()

    def query_name(self) -> str:
        """
        Return a unique name identifying the query type.

        Currently, we do not use qualified names and so the user must
        ensure the absence of clashes.
        """
        return self.__class__.__name__

    def default_tags(self) -> Sequence[str]:
        return [self.query_name()]

    @abstractmethod
    def parse_answer(self, answer: Answer) -> T | ParseError:
        pass

    def finite_answer_set(self) -> Sequence[Answer] | None:
        """ """
        return None

    def default_answer(self) -> Answer | None:
        return None
