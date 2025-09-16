"""
Answer databases, to be used to fetch implicit answers in demonstrations
and to override oracles in policies.
"""

import json
from dataclasses import dataclass
from typing import Any, Literal

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


type QueryName = str


@dataclass
class AnswerDatabase:
    answers: dict[str, int]
    pass
