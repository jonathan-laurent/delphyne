"""
Delphyne Queries.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self, cast

import delphyne.utils.typing as ty
from delphyne.core.environments import TemplatesManager
from delphyne.core.refs import Answer, AnswerModeName


@dataclass
class ParseError(Exception):
    """
    Can be used as an exception or a returned value.
    """

    error: str


type AnyQuery = AbstractQuery[Any]


class AbstractQuery[T](ABC):
    @abstractmethod
    def system_prompt(
        self,
        mode: AnswerModeName,
        params: dict[str, object],
        env: TemplatesManager | None = None,
    ) -> str:
        pass

    @abstractmethod
    def instance_prompt(
        self,
        mode: AnswerModeName,
        params: dict[str, object],
        env: TemplatesManager | None = None,
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

    def name(self) -> str:
        return self.__class__.__name__

    def tags(self) -> Sequence[str]:
        return [self.name()]

    @abstractmethod
    def parse_answer(self, answer: Answer) -> T | ParseError:
        pass
