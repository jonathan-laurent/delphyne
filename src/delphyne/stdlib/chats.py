from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import delphyne.core as dp


@dataclass(frozen=True)
class OracleMessage:
    kind: Literal["oracle"]
    answer: dp.Answer


@dataclass(frozen=True)
class FeedbackMessage:
    kind: Literal["feedback"]
    category: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    kind: Literal["tool"]
    call: dp.ToolCall
    result: Any  # JSON object


type AnswerPrefixElement = OracleMessage | FeedbackMessage | ToolResult


type AnswerPrefix = Sequence[AnswerPrefixElement]
