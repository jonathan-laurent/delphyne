"""
Delphyne Queries.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Self

from delphyne.core.refs import AnswerModeName
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


@dataclass
class ParseError:
    error: str


@dataclass(frozen=True)
class AnswerMode[T]:
    name: AnswerModeName
    parse: Callable[[str], T | ParseError]


type AnyQuery = AbstractQuery[Any, Any]


class AbstractQuery[P, T](ABC):

    @abstractmethod
    def modes(self) -> Sequence[AnswerMode[T]]:
        pass

    @abstractmethod
    def system_prompt(self, param: P, mode: AnswerModeName) -> str:
        pass

    @abstractmethod
    def instance_prompt(self, param: P, mode: AnswerModeName) -> str:
        pass

    @abstractmethod
    def name(self) -> str:
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

    @abstractmethod
    def param_type(self) -> TypeAnnot[P]:
        pass
