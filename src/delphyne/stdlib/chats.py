from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import delphyne.core as dp


@dataclass(frozen=True)
class OracleMessage:
    msg: str


@dataclass(frozen=True)
class FeedbackMessage:
    kind: Literal["message"]
    mode: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    kind: Literal["tool"]
    call: dp.ToolCall
    result: Any  # JSON object


type AnswerPrefixElement = OracleMessage | FeedbackMessage | ToolResult


type AnswerPrefix = Sequence[AnswerPrefixElement]
