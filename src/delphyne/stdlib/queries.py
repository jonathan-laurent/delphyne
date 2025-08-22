"""
Standard queries and building blocks for prompting policies.
"""

import inspect
import random
import re
import textwrap
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import EllipsisType
from typing import Any, ClassVar, Literal, Protocol, cast, overload, override

import numpy as np
import yaml

import delphyne.core as dp
import delphyne.core.chats as ct
import delphyne.core.inspect as dpi
import delphyne.stdlib.models as md
import delphyne.stdlib.policies as pol
from delphyne.core.refs import Answer
from delphyne.stdlib.opaque import Opaque, OpaqueSpace
from delphyne.stdlib.policies import IPDict, log, prompting_policy
from delphyne.stdlib.streams import SpendingDeclined, Stream, spend_on
from delphyne.utils import typing as ty
from delphyne.utils.typing import TypeAnnot, ValidationError

REPAIR_PROMPT = "repair"
"""
Template name suffix for repair prompts.
See `interactive` argument of `few_shot`.
"""

REQUEST_OTHER_PROMPT = "more"
"""
Template name suffix for a requesting different solution
See `interactive` argument of `few_shot`.
"""

ASSISTANT_PRIMING_STR = "!<assistant>"
"""
Single-line separator between the user and assistant messages in a
prompt that uses assistant priming.
"""

DEFAULT_INSTANCE_PROMPT = "{{query | yaml | trim}}"
"""
Default instance prompt template.

See `DEFAULT_INSTANCE_PROMPT_WITH_PREFIX` for the special case where
queries have a special `prefix` attribute.
"""

DEFAULT_INSTANCE_PROMPT_WITH_PREFIX = (
    "{{query | yaml(exclude_fields=['prefix']) | trim}}"
)
"""
Default instance prompt template in the presence of a special `prefix`
attribute.
"""

DEFAULT_FEEDBACK_PROMPT = '''
Error. Please try again.

{% if feedback.description %}
"""
{{feedback.description}}
"""
{% else %}
{{ fail('feedback.description is empty') }}
{% endif %}
'''.lstrip()
"""
Default template for feedback prompts.
"""


#####
##### Response Type
#####


@dataclass(frozen=True)
class FinalAnswer[F]:
    """
    See `Response`.
    """

    final: F


@dataclass(frozen=True)
class ToolRequests[T: md.AbstractTool[Any]]:
    """
    See `Response`.
    """

    tool_calls: Sequence[T]


@dataclass(frozen=True)
class Response[F, T: md.AbstractTool[Any]]:
    """
    Answer type for queries that allow follow-ups.

    `Response` values give access to both the raw LLM response (to be
    passed pass in `AnswerPrefix`) and to eventual tool calls.

    Attributes:
        answer: The raw, unparsed LLM answer.
        parsed: Either the parsed answer wrapped in `FinalAnswer` or
            some tool call requests wrapped in `ToolRequests`.
    """

    answer: dp.Answer
    parsed: FinalAnswer[F] | ToolRequests[T]


@dataclass
class WrappedParseError:
    """
    A wrapped parse error that is returned to a strategy instead of
    causing a failure.

    For queries that declare a return type of the form `Response[... |
    WrappedParseError, ...]`, parse errors do not result in failures but
    are instead wrapped and returned, to be handled explicitly by the
    surrounding strategy. For example, when building conversational
    agents with `interact`, having the query include `WrappedParseError`
    in its return type allows explicitly asking the agent to fix parse
    errors instead of failing (or having the policy retry an identical
    prompt).

    Attributes:
        error: The wrapped parse error.
    """

    error: dp.ParseError


#####
##### Query Configuration
#####


@dataclass(frozen=True)
class QueryConfig:
    """
    Mode-specific query settings.

    More settings could be added in the future.

    Attributes:
        parser: A parser specification.
    """

    parser: "ParserSpec"


type _StandardParserName = Literal["structured", "final_tool_call"]


type ParserSpec = _StandardParserName | GenericTextParser | TextParser[Any]
"""
A parser specification, which can be either:

- **"structured"**: Oracles must answer with structured output, and the
      resulting JSON object is parsed using Pydantic.
- **"final_tool_call"**: The query answer type is presented to oracles
      as a tool, which must be called to produce the final answer. This
      provides an alternative to "structured", which additionally allows
      a chain of thoughts to precede the final answer.
- **A generic text parser**: See `GenericTextParser`. Oracles are then
      expected to produce a string answer, which is parsed using the
      provided function.
- **A text parser**: See `TextParser`. Oracles are then
      expected to produce a string answer, which is parsed using the
      provided function.
"""


class GenericTextParser(Protocol):
    """
    A function that takes a type annotation along with a string as
    arguments and either returns a parsed object of the specified
    type or raises `ParseError`.

    Example: `yaml_from_last_block`.
    """

    def __call__[T](self, type: TypeAnnot[T], answer: str, /) -> T: ...


class TextParser[T](Protocol):
    """
    A function that takes a string as an argument and either returns
    a parsed object or raises `ParseError`.
    """

    def __call__(self, answer: str, /) -> T: ...


type ParserSpecDict = Mapping[dp.AnswerMode, ParserSpec]
"""
A dictionary mapping answer modes to parser specifications.
"""


type QueryConfigDict = Mapping[dp.AnswerMode, QueryConfig]
"""
A dictionary mapping answer modes to query configurations.
"""


@dataclass
class _DecomposedResponseType:
    """
    See _DecomposedAnswerType.
    """

    tools: Sequence[type[md.AbstractTool[Any]]]


@dataclass
class _DecomposedAnswerType:
    """
    Represents an answer type (type parameter of `Query`) of the form
    `F` or `Response[F, T1|...|Tn]`: `final` contains `F` and `resp`
    contains the list of all Ti in the case of a `Response` type and
    `None` otherwise.

    If `F` itself is of the form `F2 | ParseError`, then `F2` is
    assigned to `final_no_error`.
    """

    final: TypeAnnot[Any]
    resp: _DecomposedResponseType | None
    final_no_error: TypeAnnot[Any] | None

    def __init__(self, annot: TypeAnnot[Any]):
        if typing.get_origin(annot) is Response:
            args = typing.get_args(annot)
            assert len(args) == 2
            tools_raw = dpi.union_components(args[1])
            tools: list[type[md.AbstractTool[Any]]] = [
                a for a in tools_raw if issubclass(a, md.AbstractTool)
            ]
            assert len(tools) == len(tools_raw)
            self.resp = _DecomposedResponseType(tools)
            self.final = args[0]
        else:
            self.resp = None
            self.final = annot
        final_comps = dpi.union_components(self.final)
        if len(final_comps) >= 2 and WrappedParseError in final_comps:
            self.final_no_error = dpi.make_union(
                [c for c in final_comps if c != WrappedParseError]
            )
        else:
            self.final_no_error = None

    def wrap_parse_errors(self) -> bool:
        return self.final_no_error is not None

    @property
    def to_parse(self) -> TypeAnnot[Any]:
        return (
            self.final_no_error
            if self.final_no_error is not None
            else self.final
        )


#####
##### New Parser
#####


@dataclass(frozen=True)
class Parser[A]:
    """
    A parser specification.

    In addition to a mapping from answers to answer type `A`, a parser
    also specifies query settings to be passed to oracles. Indeed, these
    two components are typically tied and so specifying them together in
    a single place is clearer.

    Attributes:
        settings: The query settings associated with the parser.
        parse: The parsing function, which is allowed to raise
            the `ParseError` exception.
    """

    settings: dp.QuerySettings
    parse: Callable[[dp.Answer], A]

    def map[B](self, f: Callable[[A], B], /) -> "Parser[B]":
        """
        Apply a function to the parser's output.

        The function `f` is allowed to raise `ParseError`.
        """
        return Parser(
            settings=self.settings,
            parse=lambda ans: f(self.parse(ans)),
        )

    def validate(self, f: Callable[[A], dp.ParseError | None]) -> "Parser[A]":
        """
        Check that the parser's output satisfies a given property.

        If the property is satisfied, function `f` must return `None`.
        Otherwise, it may return or raise a `ParseError`.
        """

        def parse(ans: dp.Answer) -> A:
            res = self.parse(ans)
            if err := f(res):
                raise err
            return res

        return Parser(settings=self.settings, parse=parse)

    @property
    def wrap_errors(self) -> "Parser[A | WrappedParseError]":
        """
        Wrap parse errors into `WrappedParseError`.
        """
        assert False

    def response_with[T: md.AbstractTool[Any]](
        self, tools: TypeAnnot[T]
    ) -> "Parser[Response[A, T]]":
        """
        Wrap answers into full `Response` objects.
        """
        assert False

    @property
    def response(self) -> "GenericParser":
        """
        Wrap answers into full `Response` objects.

        Return a `GenericParser` so that the list of supported tools can
        be extracted from the query's answer type.
        """
        assert False

    @property
    def trim(self: "Parser[str]") -> "Parser[str]":
        """
        Trim the output of a string parser.
        """
        assert False

    @property
    def json(self: "Parser[str]") -> "GenericParser":
        """
        Parse a string as a JSON object.

        Return a `GenericParser` so that the target type can be
        extracted from the query's answer type.
        """
        assert False

    @property
    def yaml(self: "Parser[str]") -> "GenericParser":
        """
        Parse a string as a YAML object.

        Return a `GenericParser` so that the target type can be
        extracted from the query's answer type.
        """
        assert False

    def json_as[U](self, type: TypeAnnot[U]) -> "Parser[U]":
        """
        Parse a string as a JSON object.
        """
        assert False

    def yaml_as[U](self, type: TypeAnnot[U]) -> "Parser[U]":
        """
        Parse a string as a YAML object.
        """
        assert False


@dataclass(frozen=True)
class GenericParser:
    """
    A mapping from a query's answer type to a parser specification.

    This is useful to avoid redundancy when specifying parsers. In
    particular, it allows writing:

    ```python
    @dataclass
    class MyQuery(Query[Response[Out, Tool1 | Tool2]]):
        ...
        __parser__ = last_block.yaml.response
    ```

    instead of:

    ```python
    __parser__ = last_block.yaml_as(Out).response_with(Tool1 | Tool2)
    ```

    Attributes:
        for_type: A function that takes a type annotation and returns a
            `Parser` for this type.
    """

    for_type: "_GenericParserFn"

    @property
    def wrap_errors(self) -> "GenericParser":
        """
        Wrap parse errors into `WrappedParseError`.

        A runtime check is performed to ensure that the answer type
        features `WrappedParseError`.
        """
        assert False

    @property
    def response(self) -> "GenericParser":
        """
        Wrap answers into full `Response` objects.

        Possible tool calls are extracted from the query's answer type
        and an exception is raised if this type does not have the form
        `Response[..., ...]`.
        """
        assert False


class _GenericParserFn(Protocol):
    """
    Type of functions wrapped by `GenericParser`.
    """

    def __call__[T](self, type: TypeAnnot[T], /) -> Parser[T]: ...


def structured_as[T](type: TypeAnnot[T]) -> Parser[T]:
    assert False


# structured: GenericParser = ...

# get_text: Parser[str] = ...

# last_code_block: Parser[str] = ...


#####
##### Standard Queries
#####


class Query[T](dp.AbstractQuery[T]):
    """
    Base class for queries.

    This class adds standard convenience features on top of
    `AbstractQuery`, using reflection to allow queries to be defined
    concisely. Here is a simple example of a query type definition:

    ```python
    @dataclass class MakeSum(Query[list[int]]):
        '''Given a list of allowed numbers and a target number, you
        must find a sub-list whose elements sum up to the target.
        Just answer with a list of numbers as a JSON object and
        nothing else.'''

        allowed: list[int] target: int
    ```

    In general, a query type is a dataclass that inherits `Query[T]`,
    where `T` is the query's answer type. In the example above, no
    parser is specified and so oracles are requested to provide
    structured answers as JSON objects, which are automatically parsed
    into the answer type (`list[int]`) using pydantic. Assuming that no
    Jinja prompt file is provided, the docstring is used as a system
    prompt and instance prompts are generated by simply serializing
    `MakeSum` instances into YAML.

    ## Customizing Prompts

    System and instance prompts can be specified via Jinja templates.
    The templates manager (`TemplatesManager`) looks for templates named
    "<QueryName>.<instance|system>.jinja". Templates can also be
    provided by defining the `__system_prompt__` and/or `__instance_prompt__`
    class attributes. If none of these are provided, the query's
    docstring is used as a system prompt and `DEFAULT_INSTANCE_PROMPT`
    is used as an instance prompt template.

    The following arguments are usually made available to templates
    (although specific prompting policies can add more):

    - `query`: the query instance.
    - `mode`: the requested answer mode.
    - `params`: the query hyperparameters (e.g., as passed to `few_shot`)

    ## Answer Modes and Configurations

    A query can define several answer modes (`AnswerMode`), each of
    which can be associated with a different parser and set of settings.
    By default, the only answer mode is `None`. More answer modes can be
    defined by setting class variable `__modes__`.

    The `query_config` method maps modes to configurations (i.e., a set
    of settings, including a parser specification). Its default behavior
    works by inspecting the `__parser__` and `__config__` class
    attributes and does not typically require overriding.

    ## Allowing Multi-Message Exchanges and Tool Calls

    A common pattern for interacting with LLMs is to have multi-message
    exchanges where the full conversation history is resent repeatedly.
    LLMs are also often allowed to request tool call. This interaction
    pattern is implemented in the `interact` standard strategy. It is
    enabled by several features on the `Query` side.

    ### Answer Prefixes

    If a query type has a prefix attribute with type `AnswerPrefix`,
    this attribute can be used to provide a chat history, to be added to
    the query's prompt.

    ### The `Response` Type

    If the query answer type is `Response`, the query does not only
    return a parsed answer, but also the LLM raw answer (which can be
    appended to a chat history), and possibly a sequence of tool calls.

    ## Manually Overriding the `parse` Method

    For advanced use cases, it is possible to directly override the
    `parse` method that turns an answer into an object of type `T`. When
    this is done, the `__config__` and `__parser__` class attributes
    must not be set. Also, the `query_settings` method always returns
    the default settings instead of relying on `query_config` and must
    be overriden if another behavior is desired.
    """

    __modes__: ClassVar[Sequence[dp.AnswerMode] | None] = None
    __parser__: ClassVar[ParserSpec | ParserSpecDict | None] = None
    __config__: ClassVar[QueryConfig | QueryConfigDict | None] = None

    ### Inspection methods

    def query_config(self, mode: dp.AnswerMode) -> QueryConfig | None:
        """
        Map modes to configurations.

        This method inspects the `__config__` and `__parser__` class
        attributes (none or either can be set but not both).

        - If no attribute is set, the default configuration is used for
              all modes, which uses the `"structured"` parser.
        - If `__config__` is set to a `QueryConfig`, this configuration
              is used for all modes.
        - Alternatively, `__config__` can be set to a dictionary mapping
              modes to configurations.
        - If `__parser__` is set to a parser specification, the default
              configuration is used for all modes, *except* that the
              provided parser is used instead of the default one.
        - Alternatively, `__parser__` can also be set to a dictionary.

        In the special case where the `parse` method is overriden and
        only in this case, return `None` after ensuring that
        `__config__` and `__parser__` are not set.
        """
        cls = type(self)
        parse_overriden = dpi.is_method_overridden(Query, cls, "parse")
        if parse_overriden:
            assert cls.__config__ is None
            assert cls.__parser__ is None
            return None
        if cls.__config__ is not None:
            assert self.__parser__ is None, (
                "Cannot have both __config__ and __parser__ attributes."
            )
            config_attr = cls.__config__
            if isinstance(config_attr, QueryConfig):
                return config_attr
            else:
                assert isinstance(config_attr, Mapping)
                return cast(Any, config_attr)[mode]
        elif cls.__parser__ is not None:
            parser_attr: Any = cls.__parser__
            if isinstance(parser_attr, Mapping):
                parser = cast(Any, parser_attr)[mode]
            else:
                parser = parser_attr
            _check_valid_parser_spec(parser)
            return QueryConfig(parser)
        else:
            return QueryConfig("structured")

    @classmethod
    def _decomposed_answer_type(cls) -> _DecomposedAnswerType:
        return _DecomposedAnswerType(cls._answer_type())

    ### Parsing Answers

    @override
    def parse_answer(self, answer: dp.Answer) -> T | dp.ParseError:
        assert answer.mode in self.query_modes(), (
            f"Unknown mode: {answer.mode}"
        )
        try:
            return self.parse(answer)
        except dp.ParseError as e:
            return e

    def parse(self, answer: Answer) -> T:
        """
        A more convenient method to override instead of `parse_answer`.

        Raises `ParseError`.
        """

        # Decompose the specified answer type
        config = self.query_config(answer.mode)
        assert config is not None
        attr = config.parser
        ans_type = self._decomposed_answer_type()

        # Compute base parsing function `parser`
        parser: _ParsingFunction[Any]
        if attr == "structured":
            parser = _from_structured(ans_type.to_parse)
        elif attr == "final_tool_call":
            assert isinstance(ans_type.to_parse, type)
            parser = _from_final_tool_call(cast(type[Any], ans_type.to_parse))
        else:
            assert callable(attr)
            attr = cast(Callable[..., Any], attr)
            sig = inspect.signature(attr)
            nargs = len(sig.parameters)
            assert nargs == 1 or nargs == 2
            if nargs == 1:
                parser = _from_text_parser(attr)
            else:
                parser = _from_generic_text_parser(attr, ans_type.to_parse)

        if ans_type.wrap_parse_errors():
            parser = _wrap_parse_errors(parser)

        # If the answer type has form `Response[..., ...]`
        if ans_type.resp:
            if _has_final_tool_call(ans_type, answer):
                tcs = []
            else:
                tcs = [
                    _parse_tool_call(ans_type.resp.tools, tc)
                    for tc in answer.tool_calls
                ]
            if tcs:
                return cast(T, Response(answer, ToolRequests(tcs)))
            else:
                return cast(T, Response(answer, FinalAnswer(parser(answer))))

        return parser(answer)

    ### Query Settings

    @override
    def query_settings(self, mode: dp.AnswerMode) -> dp.QuerySettings:
        """
        Return the settings associated with the query.

        By default, this method uses the result of `query_config` to
        determine settings if `parse` is not overriden, and the default
        set of settings otherwise.
        """
        config = self.query_config(mode)
        if config is None:
            # `parse` is overriden
            return dp.QuerySettings(structured_output=None, tools=None)
        structured_output = None
        if config.parser == "structured":
            type = self._decomposed_answer_type().final
            structured_output = dp.StructuredOutputSettings(type)
        tools = None
        if tool_types := self._query_tools(config.parser):
            tools = dp.ToolSettings(
                tool_types, force_tool_call=config.parser == "final_tool_call"
            )
        return dp.QuerySettings(
            structured_output=structured_output,
            tools=tools,
        )

    def _structured_output_type(self) -> TypeAnnot[Any] | ty.NoTypeInfo:
        decomposed = self._decomposed_answer_type()
        return decomposed.final

    def _query_tools(self, parser: ParserSpec) -> Sequence[type[Any]]:
        ans_type = self._decomposed_answer_type()
        tools: list[type[Any]] = []
        if ans_type.resp is not None:
            tools = [*ans_type.resp.tools]
        if parser == "final_tool_call":
            assert isinstance(ans_type.final, type)
            tools.append(ans_type.final)
        return tools

    ### Query Prefixes

    @classmethod
    def _has_special_prefix_attr(cls):
        annots = typing.get_type_hints(cls)
        return "prefix" in annots and annots["prefix"] is ct.AnswerPrefix

    @override
    def query_prefix(self) -> ct.AnswerPrefix | None:
        """
        Return the value of the `prefix` attribute if it has type
        annotation `AnswerPrefix` or return `None`.
        """
        if self._has_special_prefix_attr():
            return getattr(self, "prefix")
        return None

    ### Producing Prompts

    @override
    def generate_prompt(
        self,
        *,
        kind: Literal["system", "instance"] | str,
        mode: dp.AnswerMode,
        params: dict[str, object],
        extra_args: dict[str, object] | None = None,
        env: dp.TemplatesManager | None = None,
    ) -> str:
        assert env is not None, _no_prompt_manager_error()
        args: dict[str, object] = {
            "query": self,
            "mode": mode,
            "params": params,
        }
        if extra_args:
            args.update(extra_args)
        if (glob := self.globals()) is not None:
            args["globals"] = glob
        return env.prompt(
            query_name=self.query_name(),
            prompt_kind=kind,
            template_args=args,
            default_template=self._default_prompt(kind),
        )

    @classmethod
    def _default_prompt(
        cls, kind: Literal["system", "instance"] | str
    ) -> str | None:
        attr_name = f"__{kind}_prompt__"
        if hasattr(cls, attr_name):
            res = getattr(cls, attr_name)
            assert isinstance(res, str)
            return textwrap.dedent(res).strip()
        if kind == "instance":
            if cls._has_special_prefix_attr():
                return DEFAULT_INSTANCE_PROMPT_WITH_PREFIX
            else:
                return DEFAULT_INSTANCE_PROMPT
        if kind == "system" and (doc := inspect.getdoc(cls)) is not None:
            return doc
        if kind == "feedback":
            return DEFAULT_FEEDBACK_PROMPT
        return None

    def globals(self) -> dict[str, object] | None:
        """
        Return global objects accessible in prompts via the `globals`
        attribute.
        """
        return None

    ### Other Simple Overrides

    @override
    def serialize_args(self) -> dict[str, object]:
        return cast(dict[str, object], ty.pydantic_dump(type(self), self))

    @classmethod
    def _answer_type(cls) -> TypeAnnot[T]:
        return dpi.first_parameter_of_base_class(cls)

    @override
    def answer_type(self) -> TypeAnnot[T]:
        return self._answer_type()

    @override
    def finite_answer_set(self) -> Sequence[dp.Answer] | None:
        # We handle the special case where the return type is a literal
        # type that is a subtype of str.
        ans = self.answer_type()
        if (res := _match_string_literal_type(ans)) is not None:
            return [dp.Answer(None, v) for v in res]
        return None

    @override
    def query_modes(self) -> Sequence[dp.AnswerMode]:
        if self.__modes__ is not None:
            return self.__modes__
        return [None]

    ### Generating Opaque Spaces

    @overload
    def using(self, get_policy: EllipsisType, /) -> Opaque[IPDict, T]: ...

    @overload
    def using[P](
        self,
        get_policy: Callable[[P], pol.PromptingPolicy] | EllipsisType,
        /,
        inner_policy_type: type[P] | None = None,
    ) -> Opaque[P, T]: ...

    def using[P](
        self,
        get_policy: Callable[[P], pol.PromptingPolicy] | EllipsisType,
        /,
        inner_policy_type: type[P] | None = None,
    ) -> Opaque[P, T]:
        """
        Turn a query into an opaque space by providing a mapping from
        the ambient inner policy to a prompting policy.

        Attributes:
            get_policy: A function that maps the ambient inner policy to
                a prompting policy to use for answering the query.
                Alternatively, if the ellipsis value `...` is passed, the
                inner policy type is assumed to be `IPDict`, and
                prompting policies are automatically selected using tags
                (see `IPDict` documentation).
            inner_policy_type: Ambient inner policy type. This information
                is not used at runtime but it can be provided to help type
                inference when necessary.

        The optional `inner_policy_type` argument is ignored at runtime
        and can be used to help type checkers infer the type of the
        ambient inner policy.
        """
        if isinstance(get_policy, EllipsisType):
            return OpaqueSpace[P, T].from_query(
                self, cast(Any, pol.dict_subpolicy)
            )
        return OpaqueSpace[P, T].from_query(self, lambda p, _: get_policy(p))

    def run_toplevel(
        self,
        env: dp.PolicyEnv,
        policy: pol.PromptingPolicy,
    ) -> Stream[T]:
        """
        Obtain a search stream of query answers, given a prompting
        policy.
        """
        attached = dp.spawn_standalone_query(self)
        return policy(attached, env)


def _no_prompt_manager_error() -> str:
    return (
        "Please provide an explicit prompt manager "
        + " or override the `system_prompt` and `instance_prompt` functions."
    )


def _parse_tool_call(
    tools: Sequence[type[md.AbstractTool[Any]]], tc: dp.ToolCall
) -> md.AbstractTool[Any]:
    for t in tools:
        if tc.name == t.tool_name():
            return _parse_or_raise(t, tc.args)
    raise dp.ParseError(description=f"Unknown tool: {tc.name}.")


def _has_final_tool_call(ans_type: _DecomposedAnswerType, answer: dp.Answer):
    if not isinstance(ans_type.final, type):
        return False
    return any(
        tc.name == md.tool_name_of_class_name(ans_type.final.__name__)
        for tc in answer.tool_calls
    )


def _check_valid_parser_spec(obj: Any):
    assert isinstance(obj, str) or callable(obj)


def _match_string_literal_type(t: Any) -> Sequence[str] | None:
    if (
        (vals := dpi.literal_type_args(t)) is not None
        and len(vals) > 0
        and all(isinstance(v, str) for v in vals)
    ):
        return vals
    return None


#####
##### Parsers
#####


class _ParsingFunction[T](Protocol):
    def __call__(self, answer: dp.Answer, /) -> T: ...


def _get_text_answer(ans: Answer) -> str:
    if ans.tool_calls:
        raise dp.ParseError(
            description="Trying to parse answer with tool calls."
        )
    if not isinstance(ans.content, str):
        raise dp.ParseError(description="Unexpected structured answer.")
    return ans.content


def _get_single_tool_call(ans: Answer) -> dp.ToolCall:
    n = len(ans.tool_calls)
    if n == 0:
        msg = "No tool call was made."
        raise dp.ParseError(description=msg)
    if n > 1:
        msg = "Too many tool calls."
        raise dp.ParseError(description=msg)
    return ans.tool_calls[0]


def _parse_or_raise[T](type: TypeAnnot[T], obj: Any) -> T:
    try:
        return ty.pydantic_load(type, obj)
    except ValidationError as e:
        raise dp.ParseError(description=str(e))


def _parse_structured_output[T](type: TypeAnnot[T], answer: dp.Answer) -> T:
    if not isinstance(answer.content, dp.Structured):
        raise dp.ParseError(
            description="A structured output was expected.",
        )
    return _parse_or_raise(type, answer.content.structured)


def _from_generic_text_parser[T](
    parser: GenericTextParser, type: TypeAnnot[T]
) -> _ParsingFunction[T]:
    return lambda answer: parser(type, _get_text_answer(answer))


def _from_text_parser[T](parser: TextParser[T]) -> _ParsingFunction[T]:
    return lambda answer: parser(_get_text_answer(answer))


def _from_structured[T](type: TypeAnnot[T]) -> _ParsingFunction[T]:
    return lambda answer: _parse_structured_output(type, answer)


def _from_final_tool_call[T](type: type[T]) -> _ParsingFunction[T]:
    return lambda answer: _parse_or_raise(
        type, _get_single_tool_call(answer).args
    )


def _wrap_parse_errors(parser: _ParsingFunction[Any]) -> _ParsingFunction[Any]:
    def parse(answer: dp.Answer) -> Any:
        try:
            return parser(answer)
        except dp.ParseError as e:
            return WrappedParseError(e)

    return parse


#####
##### Standard Parsers
#####


def raw_yaml[T](type: TypeAnnot[T], res: str) -> T:
    """
    Parse a text answer that consists in a single YAML object and
    nothing else.
    """
    try:
        parsed = yaml.safe_load(res)
        return ty.pydantic_load(type, parsed)
    except ValidationError as e:
        raise dp.ParseError(description=str(e))
    except Exception as e:
        raise dp.ParseError(description=str(e))


def yaml_from_last_block[T](type: TypeAnnot[T], res: str) -> T:
    """
    Parse the YAML object defined in the last code block of a text
    answer (between triple back quotes). In particular, this parser
    allows chain of thoughts.
    """
    final = extract_final_block(res)
    if final is None:
        raise dp.ParseError(description="No final code block found.")
    return raw_yaml(type, final)


def raw_string[T](type_annot: TypeAnnot[T], res: str) -> T:
    """
    Do not perform any parsing and return the answer as a raw string.
    """
    try:
        type_annot_resolved = dpi.resolve_aliases_in_type(type_annot)
        if isinstance(type_annot_resolved, type):  # if `type` is a class
            return type_annot_resolved(res)  # type: ignore
        if type_annot_resolved is str:
            return cast(T, res)
        assert False, f"Not a string-convertible type: {type_annot}."
    except Exception as e:
        raise dp.ParseError(description=f"raw_string parsed failed: {str(e)}")


def trimmed_raw_string[T](type: TypeAnnot[T], res: str) -> T:
    """
    Do not perform any parsing and return the answer as a raw string,
    after trimming leading and trailing whitespace.
    """
    return raw_string(type, res.strip())


def first_word[T](type: TypeAnnot[T], res: str) -> T:
    """
    Parse the first word of the answer and turn it into an object of
    type T=Literal[s1,...,sn].
    """
    vals = _match_string_literal_type(type)
    if vals is None:
        msg = f"Not recognized as a string literal type: {type}."
        raise dp.ParseError(description=msg)
    try:
        assert res, "Cannot parse an empty string."
        first = res.split()[0]
        assert first in vals, "Unallowed value: " + first
        return cast(T, first)
    except Exception as e:
        raise dp.ParseError(description=str(e))


def string_from_last_block[T](type: TypeAnnot[T], res: str) -> T:
    """
    Extract the string content of the last code block from the answer
    (surrounded by triple back quotes).
    """
    final = extract_final_block(res)
    if final is None:
        raise dp.ParseError(description="No final code block found.")
    return raw_string(type, final)


def trimmed_string_from_last_block[T](type: TypeAnnot[T], res: str) -> T:
    """
    Extract the string content of the last code block from the answer
    (surrounded by triple back quotes) and trim leading and trailing
    whitespace.
    """
    final = extract_final_block(res)
    if final is None:
        raise dp.ParseError(description="No final code block found.")
    return trimmed_raw_string(type, final)


def extract_final_block(s: str) -> str | None:
    # In case the output is ill-formed, the quotes may not be balanced.
    # This is why we use a lookahead here.
    # See tests in `test_stdlib.py`
    code_blocks = re.findall(r"(?=(```[^\n]*\n(.*?)```))", s, re.DOTALL)
    return code_blocks[-1][1] if code_blocks else None


#####
##### Example Selectors
#####


type ExampleSelector = Callable[[Sequence[dp.Example]], Sequence[dp.Example]]
"""
A function for selecting a subset of examples from a given sequence.
"""


def select_all_examples(
    examples: Sequence[dp.Example],
) -> Sequence[dp.Example]:
    """
    Example selector that returns all available examples.
    """
    return examples


def select_random_examples(num_examples: int) -> ExampleSelector:
    """
    Example selector that randomly selects a given number of examples.

    All examples are selected if less examples are available than
    requested.
    """

    def select(
        examples: Sequence[dp.Example],
    ) -> Sequence[dp.Example]:
        if num_examples >= len(examples):
            return examples
        selected = random.sample(examples, num_examples)
        return selected

    return select


def select_with_either_tags(tags: Sequence[str]):
    """
    Select examples that are annotated with at least one of a provided
    set of tags.
    """

    def select(
        examples: Sequence[dp.Example],
    ) -> Sequence[dp.Example]:
        return [ex for ex in examples if any(t in ex.tags for t in tags)]

    return select


#####
##### Prompting Policies
#####


def fetch_examples(
    database: dp.ExampleDatabase,
    query: dp.AbstractQuery[Any],
    selectors: Sequence[ExampleSelector],
) -> Sequence[tuple[dp.AbstractQuery[Any], dp.Answer]]:
    raw = list(database.examples(query.query_name(), query.serialize_args()))
    for sel in selectors:
        raw = sel(raw)
    return [(query.parse_instance(ex.args), ex.answer) for ex in raw]


def _priming_split(prompt: str) -> tuple[str, str | None]:
    # Split a prompt so as to allow assistant priming. If the prompt
    # contains `ASSISTANT_PRIMING_STR`` on a single line, then split it
    # into a `(before, after)`` pair. Otherwise, return `(prompt,
    # None)`.
    if ASSISTANT_PRIMING_STR not in prompt:  # Optimization
        return prompt, None
    lines = prompt.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == ASSISTANT_PRIMING_STR:
            before = "\n".join(lines[:i]).rstrip()
            after = "\n".join(lines[i + 1 :]).lstrip()
            if after:
                return before, after
            else:
                break
    return prompt, None


def _instance_prompt(
    query: dp.AbstractQuery[Any],
    env: dp.TemplatesManager | None,
    params: dict[str, object],
    mode: dp.AnswerMode,
):
    msgs: list[md.ChatMessage] = []
    prompt = query.generate_prompt(
        kind="instance", mode=mode, params=params, env=env
    )
    # Handle assistant priming
    prompt, priming = _priming_split(prompt)
    msgs.append(md.UserMessage(prompt))
    if priming is not None:
        # What mode we pick does not really matter here.
        msgs.append(md.AssistantMessage(Answer(None, priming)))
    # Add prefix if needed
    if (prefix := query.query_prefix()) is not None:
        for elt in prefix:
            if isinstance(elt, dp.OracleMessage):
                msgs.append(md.AssistantMessage(elt.answer))
            elif isinstance(elt, dp.FeedbackMessage):
                fmsg = query.generate_prompt(
                    kind="feedback",
                    mode=mode,
                    params=params,
                    extra_args={"feedback": elt},
                    env=env,
                )
                msgs.append(md.UserMessage(fmsg))
            else:
                assert isinstance(elt, dp.ToolResult)
                msgs.append(md.ToolMessage(elt.call, elt.result))
    return msgs


def create_prompt(
    query: dp.AbstractQuery[Any],
    examples: Sequence[tuple[dp.AbstractQuery[Any], dp.Answer]],
    params: dict[str, object],
    mode: dp.AnswerMode,
    env: dp.TemplatesManager | None,
) -> md.Chat:
    msgs: list[md.ChatMessage] = []
    sys = query.generate_prompt(
        kind="system", mode=mode, params=params, env=env
    )
    msgs.append(md.SystemMessage(sys))
    for q, ans in examples:
        msgs.extend(_instance_prompt(q, env, params, ans.mode))
        msgs.append(md.AssistantMessage(ans))
    # TODO: handle different modes
    msgs.extend(_instance_prompt(query, env, params, mode))
    return msgs


def log_oracle_response(
    env: dp.PolicyEnv,
    query: dp.AttachedQuery[Any],
    req: md.LLMRequest,
    resp: md.LLMResponse,
    *,
    verbose: bool,
):
    with env.tracer.lock:  # to avoid interleaving logs with other threads
        if verbose:
            info = {
                "request": ty.pydantic_dump(md.LLMRequest, req),
                "response": ty.pydantic_dump(md.LLMResponse, resp),
            }
            log(env, "llm_response", info, loc=query)
        # TODO: severity
        for extra in resp.log_items:
            log(env, extra.message, extra.metadata, loc=query)
        if resp.usage_info is not None:
            usage = {"model": resp.model_name, "usage": resp.usage_info}
            log(env, "llm_usage", usage, loc=query)


def _compute_value_distribution(
    values: Sequence[str], info: md.TokenInfo
) -> dict[str, float]:
    """
    Return a dictionary (possibly empty) mapping values to their
    logprob, as deduced from the metadata attached to the first token of
    the answer.
    """
    assert len(values) > 0
    logdistr: dict[str, float] = {}
    assert info.top_logprobs is not None
    for tok, lp in info.top_logprobs:
        cands = [v for v in values if v.startswith(tok.token)]
        assert len(cands) <= 1
        if not cands:
            continue
        v = cands[0]
        if v not in logdistr:
            # if values=["common", "rare"], there could be a
            # non-negligeable logprob on token "comm" (in addition to
            # "common") We only count the first instance.
            logdistr[v] = lp
    return logdistr


@dataclass
class ProbInfo(dp.SearchMeta):
    """
    Distribution probability, guaranteed to be nonempty and to sum to 1.
    """

    distr: Sequence[tuple[dp.Tracked[Any], float]]


def _parse_or_log_and_raise[T](
    answer: dp.Answer, query: dp.AttachedQuery[T], env: dp.PolicyEnv
) -> dp.Tracked[T]:
    parsed = query.parse_answer(answer)
    if isinstance(parsed, dp.ParseError):
        log(env, "parse_error", {"error": parsed}, loc=query)
        raise parsed
    return parsed


def get_request_cache(env: dp.PolicyEnv) -> md.LLMCache | None:
    cache = env.requests_cache
    assert cache is None or isinstance(cache, md.LLMCache)
    return cache


def _send_request(
    model: md.LLM, req: md.LLMRequest, env: dp.PolicyEnv
) -> dp.StreamContext[md.LLMResponse | SpendingDeclined]:
    res = yield from spend_on(
        lambda: (
            resp := model.send_request(req, get_request_cache(env)),
            resp.budget,
        ),
        estimate=model.estimate_budget(req),
    )
    return res


@prompting_policy
def classify[T](
    query: dp.AttachedQuery[T],
    env: dp.PolicyEnv,
    model: md.LLM,
    params: dict[str, object] | None = None,
    select_examples: Sequence[ExampleSelector] = (),
    mode: dp.AnswerMode = None,
    enable_logging: bool = True,
    top_logprobs: int = 20,
    temperature: float = 1.0,
    bias: tuple[str, float] | None = None,
) -> dp.StreamGen[T]:
    """
    Execute a classification query, attaching a probability distribution
    to the attached answer.

    Arguments:
        query: The query to answer.
        env: The global policy environment.
        model: The LLM to use for answering the query.
        params: Prompt hyperparameters.
        select_examples: Example selector.
        mode: The answer mode to use for parsing the query answer.
        enable_logging: Whether to log raw oracle responses.
        top_logprobs: The number of top logprobs to request from the
            LLM, putting an upper bound on the support size of the
            classifier's output distributions.
        temperature: A temperature to apply to the classifier's output
            distribution (a temperature of 0 means that only top
            elements are assigned a nonzero probability).
        bias: When `bias=(e, p)` is provided, the final classifier
            distribution `D` is transformed into `(1-p)*D + p*dirac(e)`

    See `few_shot` for details on some of the arguments above.
    """
    env.tracer.trace_query(query.ref)
    examples = fetch_examples(env.examples, query.query, select_examples)
    mngr = env.templates
    if params is None:
        params = {}
    prompt = create_prompt(query.query, examples, params, mode, mngr)
    aset = query.query.finite_answer_set()
    assert aset is not None
    vals: list[str] = []
    for a in aset:
        assert isinstance(a.content, str)
        vals.append(a.content)
    options: md.RequestOptions = {
        "logprobs": True,
        "top_logprobs": top_logprobs,
        # TODO: somehow, there seems to be a problem with this, where
        # one can get an empty answer with "finish_reason: length":
        # "max_completion_tokens": 1,
        "temperature": 0.0,
    }
    req = md.LLMRequest(
        prompt,
        num_completions=1,
        options=options,
    )
    resp = yield from _send_request(model, req, env)
    if isinstance(resp, SpendingDeclined):
        return
    log_oracle_response(env, query, req, resp, verbose=enable_logging)
    if not resp.outputs:
        return
    output = resp.outputs[0]
    answer = dp.Answer(mode, output.content)
    env.tracer.trace_answer(query.ref, answer)
    parse = partial(_parse_or_log_and_raise, query=query, env=env)
    try:
        element = parse(answer)
        lpinfo = output.logprobs
        assert lpinfo is not None
        ldistr = _compute_value_distribution(vals, lpinfo[0])
        if not ldistr:
            assert isinstance(output.content, str)
            ldistr = {output.content: 0.0}
        distr = _apply_temperature(ldistr, temperature)
        if bias is not None:
            distr = _apply_bias(distr, bias)
        distr_tup = [(parse(dp.Answer(mode, k)), p) for k, p in distr.items()]
        meta = ProbInfo(distr_tup)
        yield dp.Solution(element, meta)
    except dp.ParseError:
        return


def _apply_temperature(
    logprobs_dict: dict[str, float],
    temperature: float,
) -> dict[str, float]:
    """
    Turn log-probabilities into probabilities. We assume that
    `logprobs_dict` is not empty.
    """
    logprobs = np.array(list(logprobs_dict.values()))
    if temperature == 0:
        probs = np.zeros(len(logprobs))
        probs[np.argmax(logprobs)] = 1.0
    else:
        logprobs /= temperature
        probs = np.exp(logprobs - np.max(logprobs))
        probs = probs / probs.sum()
    return {k: p for k, p in zip(logprobs_dict, probs)}


def _apply_bias(
    probs_dict: dict[str, float], bias: tuple[str, float]
) -> dict[str, float]:
    elt, p = bias
    if elt not in probs_dict:
        probs_dict[elt] = 0
    probs_dict = {k: (1 - p) * v for k, v in probs_dict.items()}
    probs_dict[elt] += p
    return probs_dict


def _unwrap_parse_error[T](
    element: dp.Tracked[T] | dp.ParseError,
) -> dp.Tracked[T] | dp.ParseError:
    if isinstance(element, dp.Tracked):
        if isinstance(element.value, WrappedParseError):
            return element.value.error
        if isinstance(resp := element.value, Response):
            resp = cast(Response[Any, Any], resp)
            if isinstance(resp.parsed, FinalAnswer):
                if isinstance(resp.parsed.final, WrappedParseError):
                    return resp.parsed.final.error
    return element


@prompting_policy
def few_shot[T](
    query: dp.AttachedQuery[T],
    env: dp.PolicyEnv,
    model: md.LLM,
    *,
    params: dict[str, object] | None = None,
    select_examples: Sequence[ExampleSelector] = (),
    mode: dp.AnswerMode = None,
    enable_logging: bool = True,
    temperature: float | None = None,
    num_concurrent: int = 1,
    max_requests: int | None = None,
    no_wrap_parse_errors: bool = False,
    iterative_mode: bool = False,
) -> dp.StreamGen[T]:
    """
    The standard few-shot prompting policy.

    A prompt is formed by concatenating a system prompt, a series of
    examples (each of which consists in an instance prompt followed by
    an answer), and a final answer prompt. Then, answers are repeatedly
    sampled and parsed, until a spending request is declined.

    Arguments:
        query: The query to answer. env: The policy environment.
        model: The LLM to use for answering the query
        params: Prompt hyperparameters, which are passed to prompt
            templates as a `params` dictionary.
        select_examples: A series of filters for selecting examples, to
            be applied in sequence. By default, no filter is used and so
            all available examples are fetched.
        mode: The answer mode to use for parsing the query answer.
        enable_logging: Whether to log raw oracle responses.
        temperature: The temperature parameter to use with the LLM, as a
            number from 0 to 2.
        num_concurrent: The number of completions to request for each
            LLM call. Note that most LLM providers only bill input
            tokens once, regardless of the number of completions.
        max_requests: The maximum number of LLM requests to perform
            before the resulting seach stream terminates, if any.
        no_wrap_parse_errors: If set to `True`, then parser results of
            type `WrappedParseError` are unwrapped and treated as normal
            parse errors.
        iterative_mode: If set to `False` (default), answers are
            repeatedly and independently sampled. If set to `True`, a
            single chat conversation occurs instead: Whenever a parse
            error occurs, a message is issued by rendering the
            `<QueryName>.repair.jinja` template, asking for a new
            attempt to be made (the `ParseError` object is available as
            an `error` template variable). After an answer is
            successfully generated and parsed, a message is issued by
            rendering the `<QueryName>.more.jinja` template, asking for
            another different answer to be generated.

            This special mode allows creating simple conversational
            agents with very little effort, by only defining a single
            query. However, it does not support tool calls, and the
            demonstration language cannot be used to illustrate how
            `repair` and `more` messages should be handled. For
            implementing more advanced conversational agents, see
            the standard `interact` strategy.
    """
    assert not iterative_mode or num_concurrent == 1
    assert max_requests is None or max_requests > 0
    env.tracer.trace_query(query.ref)
    examples = fetch_examples(env.examples, query.query, select_examples)
    mngr = env.templates
    if params is None:
        params = {}
    prompt = create_prompt(query.query, examples, params, mode, mngr)
    settings = query.query.query_settings(mode)
    options: md.RequestOptions = {}
    if temperature is not None:
        options["temperature"] = temperature
    structured_output = None
    if settings.structured_output is not None:
        out_type = settings.structured_output.type
        structured_output = md.Schema.make(out_type)
    tools = []
    if settings.tools is not None:
        if settings.tools.force_tool_call:
            options["tool_choice"] = "required"
        tools = [md.Schema.make(t) for t in settings.tools.tool_types]
    num_reqs = 0
    while max_requests is None or num_reqs < max_requests:
        num_reqs += 1
        req = md.LLMRequest(
            prompt,
            num_completions=num_concurrent,
            options=options,
            tools=tools,
            structured_output=structured_output,
        )
        resp = yield from _send_request(model, req, env)
        if isinstance(resp, SpendingDeclined):
            return
        log_oracle_response(env, query, req, resp, verbose=enable_logging)
        if not resp.outputs:
            log(env, "llm_no_output", loc=query)
            continue
        elements: list[dp.Tracked[T] | dp.ParseError] = []
        answers: list[dp.Answer] = []
        for output in resp.outputs:
            answer = dp.Answer(mode, output.content, tuple(output.tool_calls))
            answers.append(answer)
            element = query.parse_answer(answer)
            if no_wrap_parse_errors:
                element = _unwrap_parse_error(element)
            env.tracer.trace_answer(query.ref, answer)
            if isinstance(element, dp.ParseError):
                log(env, "parse_error", {"error": element}, loc=query)
            elements.append(element)
        for element in elements:
            if not isinstance(element, dp.ParseError):
                yield dp.Solution(element)
        # In iterative mode, we want to keep the conversation going
        if iterative_mode:
            assert len(elements) == 1 and len(answers) == 1
            element = elements[0]
            if isinstance(element, dp.ParseError):
                try:
                    repair = query.query.generate_prompt(
                        kind=REPAIR_PROMPT,
                        mode=mode,
                        params={"params": params, "error": element},
                        env=mngr,
                    )
                except dp.TemplateFileMissing:
                    repair = (
                        "Invalid answer. Please consider the following"
                        + f" feedback and try again:\n\n{element}"
                    )
                new_message = md.UserMessage(repair)
            else:
                try:
                    gen_new = query.query.generate_prompt(
                        kind=REQUEST_OTHER_PROMPT,
                        mode=mode,
                        params={"params": params},
                        env=mngr,
                    )
                except dp.TemplateFileMissing:
                    gen_new = "Good! Can you generate a different answer now?"
                new_message = md.UserMessage(gen_new)

            prompt = (*prompt, md.AssistantMessage(answers[0]), new_message)


#####
##### Constant Answers
#####


@prompting_policy
def answer_with[T](
    query: dp.AttachedQuery[T],
    env: dp.PolicyEnv,
    answers: Sequence[str],
    probs: Sequence[float] | None = None,
    mode: dp.AnswerMode = None,
) -> dp.StreamGen[T]:
    """
    A prompting policy that returns a hardcoded set of answers without
    looking at the query. If `probs` is not provided, then all elements
    are yielded in sequence. If it is, the top element is yielded once,
    with a `ProbInfo` annotation featuring the provided distribution.
    """
    assert answers
    parse = partial(_parse_or_log_and_raise, query=query, env=env)
    try:
        tracked = [parse(dp.Answer(mode, a)) for a in answers]
        if probs is not None:
            assert len(tracked) == len(probs)
            assert all(0 <= p <= 1 for p in probs)
            max_prob = max(probs)
            max_idx = probs.index(max_prob)
            yield dp.Solution(
                tracked[max_idx],
                meta=ProbInfo(
                    [(tracked[i], probs[i]) for i in range(len(tracked))]
                ),
            )

        else:
            for elt in tracked:
                yield dp.Solution(elt)
    except dp.ParseError as e:
        assert False, f"Failed to parse hardcoded answer: {e}"
