"""
Delphyne Demonstrations.
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from delphyne.core.refs import Hint, SpaceRef

type TestCommandString = str


@dataclass
class Answer:
    answer: str
    mode: str | None = None
    label: str | None = None
    example: bool | None = None


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
    queries: list[QueryDemo] = dataclasses.field(default_factory=list)
    demonstration: str | None = None  # optional label


type Demo = QueryDemo | StrategyDemo


type DemoFile = list[Demo]


type NodeTag = str


@dataclass
class Run:
    hints: Sequence[Hint]
    until: NodeTag | None


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
