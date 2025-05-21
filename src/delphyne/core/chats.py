from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from delphyne.core.refs import Answer, Structured, ToolCall


@dataclass(frozen=True)
class OracleMessage:
    kind: Literal["oracle"]
    answer: Answer


@dataclass(frozen=True)
class FeedbackMessage:
    kind: Literal["feedback"]
    category: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    kind: Literal["tool"]
    call: ToolCall
    result: str | Structured


type AnswerPrefixElement = OracleMessage | FeedbackMessage | ToolResult


type AnswerPrefix = Sequence[AnswerPrefixElement]
