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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class FromCommandResultHindsightFeedback:
    """
    Source of an answer located in the hindsight feedback collected
    during the execution of a command.

    Attributes:
        command_file: Path to the command file, relative to the
            workspace root.
        node_id: Index of the associated hindsight feedback node.
    """

    source: Literal["command_result_hindsight"]
    command_file: str
    node_id: int


type LocatedAnswerSource = (
    FromStandaloneQueryDemo
    | FromStrategyDemo
    | FromCommandResult
    | FromCommandResultHindsightFeedback
)
"""
Provenance information for answers in databases.

Objects of this type must be hashable, so that duplicate answers can be
eliminated in `AnswerDatabase`.
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
            return f"{src.command_file}:trace:{src.answer_id}"
        case FromCommandResultHindsightFeedback():
            return f"{src.command_file}:hindsight_feedback:{src.node_id}"


@dataclass
class LocatedAnswer:
    """
    An answer with provenance information.
    """

    answer: Answer
    source: LocatedAnswerSource

    def is_from_hindsight_feedback(self) -> bool:
        return isinstance(self.source, FromCommandResultHindsightFeedback)


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
"""
Map an answer source to an iterable of query/answer pairs

Duplicates are allowed and are removed by `AnswerDatabase`.
"""


@dataclass
class SourceLoadingError(Exception):
    source: AnswerSource
    exn: Exception

    def __str__(self):
        return f"Failed to load answers from source {self.source}:\n{self.exn}"


@dataclass
class AnswerDatabase:
    """
    A database of answers, to be fetched as implicit answers in a
    demonstration or to override oracles in policies.
    """

    answers: dict[SerializedQuery, list[LocatedAnswer]]

    def __init__(
        self, sources: Sequence[AnswerSource], *, loader: AnswerDatabaseLoader
    ):
        """
        Initialize the database by loading answers from a number of
        sources.

        Raises:
            SourceLoadingError: A source failed to be loaded.
        """
        self.answers = defaultdict(list)
        for s in sources:
            try:
                answers = loader(s)
            except Exception as e:
                raise SourceLoadingError(s, e)
            for q, a in answers:
                self.answers[q].append(a)
        self._remove_duplicates()

    def _remove_duplicates(self):
        for query, answers in self.answers.items():
            seen_sources: set[LocatedAnswerSource] = set()
            unique_answers: list[LocatedAnswer] = []
            for ans in answers:
                src = ans.source
                if src not in seen_sources:
                    seen_sources.add(src)
                    unique_answers.append(ans)
            self.answers[query] = unique_answers

    def fetch(self, query: SerializedQuery) -> LocatedAnswer | None:
        """
        Fetch an answer for a query in the database.

        Return `None` if there is no match and the matching answer if
        there is a unique match. Raise `SeveralAnswerMatches` if there
        are multiple matches. If one match comes from hindsight
        feedback, it is preferred over other matches.
        """
        cands = self.answers[query]
        if not cands:
            return None
        # If a candidate uses hindsight feedback, prefer it.
        if any(c.is_from_hindsight_feedback() for c in cands):
            cands = [c for c in cands if c.is_from_hindsight_feedback()]
        if len(cands) == 1:
            return cands[0]
        raise SeveralAnswerMatches(query, cands)
