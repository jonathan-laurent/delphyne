"""
Delphyne Demonstrations.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from delphyne.core import refs
from delphyne.core.refs import Hint, SpaceRef

type TestCommandString = str


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]


@dataclass
class Answer:
    answer: str | object
    call: Sequence[ToolCall] = ()
    structured: Literal["auto", True] = "auto"
    mode: str | None = None
    label: str | None = None
    example: bool | None = None
    tags: Sequence[str] = ()
    justification: str | None = None


@dataclass
class QueryDemo:
    query: str
    args: dict[str, Any]
    answers: list[Answer]
    demonstration: str | None = None  # optional label


@dataclass
class StrategyDemo:
    strategy: str
    args: dict[str, Any]
    tests: list[TestCommandString]
    queries: list[QueryDemo]
    demonstration: str | None = None  # optional label


type Demo = QueryDemo | StrategyDemo


type DemoFile = list[Demo]


@dataclass(frozen=True)
class TagSelector:
    tag: str
    num: int | None


type TagSelectors = Sequence[TagSelector]


type NodeSelector = TagSelectors | WithinSpace


@dataclass
class WithinSpace:
    space: TagSelectors
    selector: NodeSelector


@dataclass
class Run:
    hints: Sequence[Hint]
    until: NodeSelector | None


@dataclass
class SelectSpace:
    space: SpaceRef
    expects_query: bool


@dataclass
class IsSuccess:
    pass


@dataclass
class IsFailure:
    pass


@dataclass
class Save:
    name: str


@dataclass
class Load:
    name: str


type TestStep = Run | SelectSpace | IsSuccess | IsFailure | Save | Load


type TestCommand = Sequence[TestStep]


def translate_answer(ans: Answer) -> refs.Answer:
    if isinstance(ans.answer, str) and ans.structured == "auto":
        content = ans.answer
    else:
        content = refs.Structured(ans.answer)
    tool_calls = tuple([refs.ToolCall(c.tool, c.args) for c in ans.call])
    return refs.Answer(ans.mode, content, tool_calls, ans.justification)
