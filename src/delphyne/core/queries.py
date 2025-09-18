"""
Abstract Class for Delphyne Queries.

A more featureful subclass is provided in the standard library
(`Query`), which uses reflection for convenience.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self, cast, final

import delphyne.utils.typing as ty
from delphyne.core.chats import AnswerPrefix
from delphyne.core.errors import Error
from delphyne.core.refs import Answer, AnswerMode


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


#####
##### Abstract Templates Manager
#####


class AbstractTemplatesManager(ABC):
    @abstractmethod
    def prompt(
        self,
        *,
        query_name: str,
        prompt_kind: Literal["system", "instance"] | str,
        template_args: dict[str, Any],
        default_template: str | None = None,
    ) -> str:
        """
        Render a prompt message using a template.

        Args:
            query_name: The name of the query for which the prompt is
                built. Used to determine the template file name
                (e.g. "{query_name}.{prompt_kind}.jinja").
            prompt_kind: The kind of prompt (e.g. "system" or "instance")
                that is being rendered, used to determine the name of the
                template file to use.
            template_args: A dictionary of arguments to pass to the
                template. It must not contain key "data", which is
                reserved for the data loaded from the data directories.
            default_template: If provided, this template will be used if
                no template file is found for the given query name and
                kind instead of raising an error.

        Raises:
            TemplateFileMissing: template file not found.
            TemplateError: error raised while rendering the template.
        """

        pass


@dataclass
class TemplateError(Exception):
    """
    Wrapper for template-related exceptions.
    """

    name: str
    exn: Exception


@dataclass
class TemplateFileMissing(Exception):
    """
    Exception raised when a template file is missing.

    This exception should only be raised when a top-level template file
    is missing. If an `include` statement fails within a template, a
    `TemplateError` exception should be raised instead.
    """

    file: str


#####
##### Abstract Queries
#####


class AbstractQuery[T](ABC):
    """
    Abstract Class for Delphyne Queries.

    The type parameter `T` indicates the type of parsed query answers.

    A more featureful subclass is provided in the standard library
    (`Query`), which uses reflection for convenience.

    !!! info "Answer Modes"
        Queries are allowed to define multiple answer modes
        (`AnswerMode`), each mode being possibly associated with
        different settings and with a different parser.
    """

    @abstractmethod
    def generate_prompt(
        self,
        *,
        kind: Literal["system", "instance", "feedback"] | str,
        mode: AnswerMode,
        params: dict[str, object],
        extra_args: dict[str, object] | None = None,
        env: AbstractTemplatesManager | None,
    ) -> str:
        """
        Generate a prompt message for the query.

        Args:
            kind: Kind of prompt to generate. Standard prompt kinds are
                "system", "instance", or "feedback" but others can be
                supported (within or outside the standard library).
            mode: Answer mode selected for the query.
            params: Query hyperparameters.
            extra_args: Additional arguments to pass to the template.
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
    def query_modes(self) -> Sequence[AnswerMode]:
        """
        Return the sequence of available answer modes.
        """
        pass

    def query_prefix(self) -> AnswerPrefix | None:
        """
        Return the chat history featured in the query, if any.

        This is useful to emulate conversational agents by issuing a
        query repeatedly, passing it the whole, updated conversation
        history every time (see `interact`).
        """
        return None

    def query_settings(self, mode: AnswerMode) -> QuerySettings:
        """
        Return the settings associated with the query.
        """
        return QuerySettings()

    @final
    def query_name(self) -> str:
        """
        Return the name of the query (i.e., the name of the associated
        class).
        """
        return self.__class__.__name__

    def default_tags(self) -> Sequence[str]:
        """
        Return a default set of tags to associate with spaces induced by
        the query.

        These tags can be overriden (see `SpaceBuilder`).
        """
        return [self.query_name()]

    @abstractmethod
    def parse_answer(self, answer: Answer) -> T | ParseError:
        """
        Parse a query answer.
        """
        pass

    def finite_answer_set(self) -> Sequence[Answer] | None:
        """
        For queries with a finite set of possible answers, return this
        set. Otherwise, return `None`.

        Demonstration tests can use a special `#<val>` hint to select
        the first answer with content `<val>` from this set.

        Example uses include classification queries (see `classify`)
        where a distribution of answers is extracted from LLM logits,
        flag queries...
        """
        return None

    def default_answer(self) -> Answer | None:
        """
        Return a default answer for the query, if any.

        Default answers are used to answer queries in demonstration
        tests when no answer is provided in the `queries` section and no
        applicable hint is available.
        """
        return None

    def hindsight_answer(self, parsed: T) -> Answer | None:
        """
        Return a hindsight answer that parses back to the given value.
        """
        return None
