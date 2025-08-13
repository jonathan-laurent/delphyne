"""
Types for representing LLM chat histories.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from delphyne.core.refs import Answer, Structured, ToolCall


@dataclass(frozen=True)
class OracleMessage:
    """
    Messge containing an oracle answer.
    """

    kind: Literal["oracle"]
    answer: Answer


@dataclass(frozen=True)
class FeedbackMessage:
    """
    Message containing user feedback.
    """

    kind: Literal["feedback"]
    category: str | None = None
    description: str | None = None
    arg: Any | None = None  # must be serializable


@dataclass(frozen=True)
class ToolResult:
    """
    User message containing the result of a tool call previously
    initiated by an LLM.
    """

    kind: Literal["tool"]
    call: ToolCall
    result: str | Structured


type AnswerPrefixElement = OracleMessage | FeedbackMessage | ToolResult
"""
LLM chat history element.
"""


type AnswerPrefix = Sequence[AnswerPrefixElement]
"""
An LLM chat history, to be passed to a query as an answer prefix (see
`Query.query_prefix`).
"""
