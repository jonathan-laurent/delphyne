"""
Abstract interface for queries.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Never

import pydantic

from delphyne.utils.typing import NoTypeInfo, TypeAnnot


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class PromptOptions:
    model: str | None = None
    max_tokens: int | None = None


@dataclass
class Prompt:
    messages: Sequence[Message]
    options: PromptOptions

    def json(self) -> str:
        adapter = pydantic.TypeAdapter(Prompt)
        return adapter.dump_python(self, exclude_defaults=True)


@dataclass
class ParseError:
    error: str


class Query[P, T](ABC):
    """
    A Delphyne query, as can be inspected in the UI.
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def return_type(self) -> TypeAnnot[T] | NoTypeInfo:
        pass

    @abstractmethod
    def param_type(self) -> TypeAnnot[P]:
        pass

    @abstractmethod
    def serialize_args(self) -> dict[str, object]:
        pass

    @abstractmethod
    def parse_answer(self, res: str) -> T | ParseError:
        pass

    @abstractmethod
    def create_prompt[Q](
        self, params: P, examples: Sequence[tuple[Q, str]]
    ) -> Prompt:  # fmt: skip
        pass

    @classmethod
    @abstractmethod
    def parse(cls, args: dict[str, object]) -> "Query[Never, object]":
        pass


type AnyQuery = Query[Any, Any]
