"""
Abstract Class for Delphyne Queries.

A more featureful subclass is provided in the standard library
(`Query`), which uses reflection for convenience.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self, cast

import delphyne.utils.typing as ty
from delphyne.core.chats import AnswerPrefix
from delphyne.core.environments import TemplatesManager
from delphyne.core.errors import Error
from delphyne.core.refs import Answer, AnswerModeName


@dataclass
class ParseError(Error, Exception):
    """
    Parse Error.

    Can be used as an exception or a returned value.
    """

    def __init__(
        self,
        *,
        label: str | None = None,
        description: str | None = None,
        meta: Any | None = None,
    ):
        if label is None:
            label = "parse_error"
        super().__init__(label=label, description=description, meta=meta)


@dataclass(frozen=True)
class StructuredOutputSettings:
    """
    Settings for LLM structured output.

    Attributes:
        type: Expected type for the output, from which a schema can be
        derived if provided.
    """

    type: ty.TypeAnnot[Any] | ty.NoTypeInfo


@dataclass(frozen=True)
class ToolSettings:
    """
    Tool call settings.

    Attributes:
        tool_types: Nonempty sequence of available tools. All tools must
            be classes, although more constraints can be put by specific
            queries and prompting policies.
        force_tool_call: If True, oracles are informed that a tool call
            **must** be made.
    """

    tool_types: Sequence[type[Any]]
    force_tool_call: bool


@dataclass(frozen=True)
class QuerySettings:
    """
    Settings associated with a query.

    These settings can accessed by prompting policies so as to make
    appropriate requests to LLMs.

    Attributes:
        structured_output: Settings for structured output, or `None` if
            structured output is not enabled.
        tools: Settings for tool calls, or `None` if no tools are
            available.
    """

    structured_output: StructuredOutputSettings | None = None
    tools: ToolSettings | None = None


type AnyQuery = AbstractQuery[Any]
"""
Convenience alias for a query of any type.
"""


class AbstractQuery[T](ABC):
    """
    Abstract Class for Delphyne Queries.

    A more featureful subclass is provided in the standard library
    (`Query`), which uses reflection for convenience.

    ??? "Answer Modes"
        Queries are allowed to define multiple answer modes
        (`AnswerModeName`), each mode being possibly associated with
        different settings and with a different parser.
    """

    @abstractmethod
    def generate_prompt(
        self,
        *,
        kind: Literal["system", "instance", "feedback"] | str,
        mode: AnswerModeName,
        params: dict[str, object],
        env: TemplatesManager | None,
    ) -> str:
        """
        Generate a prompt message for the query.

        Args:
            kind: Kind of prompt to generate. Standard prompt kinds are
                "system", "instance", or "feedback" but others can be
                supported (within or outside the standard library).
            mode: Answer mode selected for the query.
            params: Query hyperparameters.
            env: Template manager used to load Jinja templates.
                Exceptions may be raised when it is needed but not
                provided.
        """
        pass

    def serialize_args(self) -> dict[str, object]:
        """
        Serialize the query arguments as a dictionary of JSON values.
        """
        return cast(dict[str, object], ty.pydantic_dump(type(self), self))

    @classmethod
    def parse_instance(cls, args: dict[str, object]) -> Self:
        """
        Parse a query instance from a dictionary of serialized
        arguments.
        """
        return ty.pydantic_load(cls, args)

    @abstractmethod
    def answer_type(self) -> ty.TypeAnnot[T] | ty.NoTypeInfo:
        """
        Return the answer type for the query, or `NoTypeInfo()` if this
        information is not available.
        """
        pass

    @abstractmethod
    def query_modes(self) -> Sequence[AnswerModeName]:
        """
        Return the sequence of available answer modes.
        """
        pass

    def query_prefix(self) -> AnswerPrefix | None:
        return None

    def query_settings(self, mode: AnswerModeName) -> QuerySettings:
        return QuerySettings()

    def query_name(self) -> str:
        """
        Return a unique name identifying the query type.

        Currently, we do not use qualified names and so the user must
        ensure the absence of clashes.
        """
        return self.__class__.__name__

    def default_tags(self) -> Sequence[str]:
        return [self.query_name()]

    @abstractmethod
    def parse_answer(self, answer: Answer) -> T | ParseError:
        pass

    def finite_answer_set(self) -> Sequence[Answer] | None:
        """ """
        return None

    def default_answer(self) -> Answer | None:
        return None
