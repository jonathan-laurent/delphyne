"""
Delphyne Queries.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from delphyne.core.environment import TemplatesManager
from delphyne.core.refs import AnswerModeName
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


@dataclass
class ParseError:
    error: str


@dataclass(frozen=True)
class AnswerMode[T]:
    parse: Callable[[str], T | ParseError]


type AnyQuery = AbstractQuery[Any]


class AbstractQuery[T](ABC):
    @property
    @abstractmethod
    def modes(self) -> Mapping[AnswerModeName, AnswerMode[T]]:
        pass

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

    @abstractmethod
    def serialize_args(self) -> dict[str, object]:
        pass

    @classmethod
    @abstractmethod
    def parse(cls, args: dict[str, object]) -> Self:
        pass

    @abstractmethod
    def answer_type(self) -> TypeAnnot[T] | NoTypeInfo:
        pass

    def name(self) -> str:
        return self.__class__.__name__

    def tags(self) -> Sequence[str]:
        return [self.name()]
