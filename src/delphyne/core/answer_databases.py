"""
Answer databases, to be used to fetch implicit answers in demonstrations
and to override oracles in policies.
"""

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from delphyne.core.demos import AnswerSource
from delphyne.core.queries import AbstractQuery
from delphyne.core.refs import Answer


@dataclass
class FromStandaloneQueryDemo:
    """
    Source of an answer located in a standalone query demo.

    Attributes:
        demo_file: Path to the demo file, relative to the
            workspace root.
        demo_name: Name of the demo within the demo file.
    """

    source: Literal["standalone_query_demo"]
    demo_file: str
    demo_name: str


@dataclass
class FromStrategyDemo:
    """
    Source of an answer located in a strategy demo.

    Attributes:
        demo_file: Path to the demo file, relative to the
            workspace root.
        demo_name: Name of the demo within the demo file.
        query_id: Index of the query within the demo (0-based).
        answer_id: Index of the answer within the query (0-based).
    """

    source: Literal["strategy_demo"]
    demo_file: str
    demo_name: str
    query_id: int
    answer_id: int


@dataclass
class FromCommandResult:
    """
    Source of an answer located the result section of a command file.

    Attributes:
        command_file: Path to the command file, relative to the
            workspace root.
        answer_id: Index of the answer within the command result
            (0-based).
    """

    source: Literal["command_result"]
    command_file: str
    answer_id: int


type LocatedAnswerSource = (
    FromStandaloneQueryDemo | FromStrategyDemo | FromCommandResult
)
"""
Provenance information for answers in databases.
"""


def pp_located_answer_source(src: LocatedAnswerSource) -> str:
    match src:
        case FromStandaloneQueryDemo():
            return f"{src.demo_file}:{src.demo_name}"
        case FromStrategyDemo():
            return (
                f"{src.demo_file}:{src.demo_name}:"
                f"{src.query_id}:{src.answer_id}"
            )
        case FromCommandResult():
            return f"{src.command_file}:{src.answer_id}"


@dataclass
class LocatedAnswer:
    """
    An answer with provenance information.
    """

    answer: Answer
    source: LocatedAnswerSource
    # TODO: add `hindsight` flag here.


@dataclass(frozen=True)
class SerializedQuery:
    """
    A hashable representation of a query.

    Attributes:
        name: The name of the query.
        args: The serialized arguments of the query, as a canonical JSON
            string. Object keys are sorted so that equality is defined
            modulo key order.
    """

    name: str
    args_str: str

    @staticmethod
    def _dump_json(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True)

    @staticmethod
    def make(query: AbstractQuery[Any]) -> "SerializedQuery":
        args = SerializedQuery._dump_json(query.serialize_args())
        return SerializedQuery(query.query_name(), args)

    @staticmethod
    def from_json(name: str, args: dict[str, Any]) -> "SerializedQuery":
        return SerializedQuery(name, SerializedQuery._dump_json(args))

    @property
    def args_dict(self):
        return json.loads(self.args_str)

    def parse[T: AbstractQuery[Any]](self, type: type[T]) -> T:
        return type.parse_instance(self.args_dict)


@dataclass
class SeveralAnswerMatches(Exception):
    query: SerializedQuery
    answers: Sequence[LocatedAnswer]

    def __str__(self):
        lines: list[str] = []
        lines.append(f"Several answers match query of type {self.query.name}:")
        for a in self.answers:
            lines.append(f"  - {pp_located_answer_source(a.source)}")
        # TODO: also print query arguments at the end of the message?
        return "\n".join(lines)


type AnswerDatabaseLoader = Callable[
    [AnswerSource], Iterable[tuple[SerializedQuery, LocatedAnswer]]
]


@dataclass
class AnswerDatabase:
    answers: dict[SerializedQuery, list[LocatedAnswer]]

    def __init__(
        self, sources: Sequence[AnswerSource], *, loader: AnswerDatabaseLoader
    ):
        self.answers = defaultdict(list)
        for s in sources:
            for q, a in loader(s):
                self.answers[q].append(a)

    def fetch(self, query: SerializedQuery) -> LocatedAnswer | None:
        cands = self.answers[query]
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        raise SeveralAnswerMatches(query, cands)
