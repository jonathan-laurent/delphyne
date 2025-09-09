"""
Standard queries and building blocks for prompting policies.
"""

import inspect
import random
import re
import textwrap
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from types import EllipsisType
from typing import Any, ClassVar, Literal, Protocol, cast, overload, override

import numpy as np

import delphyne.core as dp
import delphyne.core.chats as ct
import delphyne.core.inspect as dpi
import delphyne.stdlib.models as md
import delphyne.stdlib.policies as pol
from delphyne.core.refs import Answer
from delphyne.stdlib.environments import Example, ExampleDatabase, PolicyEnv
from delphyne.stdlib.opaque import Opaque, OpaqueSpace
from delphyne.stdlib.policies import IPDict, prompting_policy
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
    passed pass in `AnswerPrefix`) and to eventual tool calls. See the
    `Parser.response`, `Parser.response_with`, and
    `GenericParser.response` methods for creating parsers that produce
    `Response` values.

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

    See the `Parser.wrap_errors` and `GenericParser.wrap_errors` methods
    for creating parsers that produce `WrappedParseError` values.

    Attributes:
        error: The wrapped parse error.
    """

    error: dp.ParseError


#####
##### Formatting Metadata
#####


@dataclass
class FormattingMetadata:
    """
    Metadata passed to prompt templates for generating formatting
    instructions. Such metadata can be produced by parsers.
    """

    where: (
        Literal["last_code_block", "full_answer", "tool_call"] | str | None
    ) = None
    what: Literal["yaml", "json", "text", "one_word"] | str | None = None
    schema: md.Schema | None = None


#####
##### Parsers
#####


@dataclass(frozen=True)
class Parser[A]:
    """
    A parser specification.

    In addition to a mapping from answers to answer type `A`, a parser
    also specifies query settings to be passed to oracles, along with
    special formatting instructions to be rendered into the prompt.
    Indeed, these components are typically tied and so specifying them
    together in a single place is clearer.

    Attributes:
        settings: The query settings associated with the parser.
        formatting: Formatting metadata.
        parse: The parsing function, which is allowed to raise
            the `ParseError` exception.
    """

    settings: dp.QuerySettings
    formatting: FormattingMetadata
    parse: Callable[[dp.Answer], A]

    def update_formatting(
        self, f: Callable[[FormattingMetadata], FormattingMetadata], /
    ) -> "Parser[A]":
        return replace(self, formatting=f(self.formatting))

    def map[B](
        self,
        f: Callable[[A], B | dp.ParseError],
        /,
        *,
        catch_exn: bool = False,
    ) -> "Parser[B]":
        """
        Apply a function to the parser's output.

        Arguments:
            f: The function to apply, which is allowed to raise or
                return `ParseError`.
            catch_exn: If `True`, any other exception raised by `f` is
                caught and wrapped into a `ParseError`.
        """

        def parse(ans: dp.Answer) -> B:
            res = self.parse(ans)
            try:
                ret = f(res)
            except dp.ParseError as e:
                raise e
            except Exception as e:
                if catch_exn:
                    raise dp.ParseError(description=str(e))
                else:
                    raise e
            if isinstance(ret, dp.ParseError):
                raise ret
            return ret

        return Parser(
            settings=self.settings, formatting=self.formatting, parse=parse
        )

    def validate(
        self,
        f: Callable[[A], dp.ParseError | None],
        /,
        *,
        catch_exn: bool = False,
    ) -> "Parser[A]":
        """
        Check that the parser's output satisfies a given property.

        If the property is satisfied, function `f` must return `None`.
        Otherwise, it may return or raise a `ParseError`.
        """

        def parse(ans: dp.Answer) -> A:
            res = self.parse(ans)
            try:
                opt_err = f(res)
            except dp.ParseError as e:
                raise e
            except Exception as e:
                if catch_exn:
                    raise dp.ParseError(description=str(e))
                else:
                    raise e
            if opt_err:
                raise opt_err
            return res

        return Parser(
            settings=self.settings, formatting=self.formatting, parse=parse
        )

    @property
    def wrap_errors(self) -> "Parser[A | WrappedParseError]":
        """
        Wrap parse errors into `WrappedParseError`.
        """

        def parse(ans: dp.Answer) -> A | WrappedParseError:
            try:
                return self.parse(ans)
            except dp.ParseError as e:
                return WrappedParseError(e)

        return Parser(
            settings=self.settings, formatting=self.formatting, parse=parse
        )

    def response_with[T: md.AbstractTool[Any]](
        self, tools: TypeAnnot[T]
    ) -> "Parser[Response[A, T]]":
        """
        Wrap answers into full `Response` objects.
        """

        tools_raw = dpi.union_components(tools)
        tools_types: list[type[md.AbstractTool[Any]]] = [
            a for a in tools_raw if issubclass(a, md.AbstractTool)
        ]
        assert len(tools_types) == len(tools_raw), (
            f"Invalid tools union: {tools}"
        )
        if self.settings.tools is None:
            tools_settings = dp.ToolSettings(
                tool_types=tools_types, force_tool_call=False
            )
        else:
            tools_settings = dp.ToolSettings(
                tool_types=[*tools_types, *self.settings.tools.tool_types],
                force_tool_call=self.settings.tools.force_tool_call,
            )
        settings = replace(self.settings, tools=tools_settings)

        def parse(ans: dp.Answer) -> Response[A, T]:
            # If the answer is one of the provided tool types, we
            # return. Otherwise, we call the parser recursively.
            tcs: list[T] = []
            for tc in ans.tool_calls:
                for t in tools_types:
                    if tc.name == t.tool_name():
                        tcs.append(_parse_or_raise(t, tc.args))
                        break
            if tcs:
                return Response[A, T](ans, ToolRequests(tcs))
            else:
                parsed = self.parse(ans)
                return Response[A, T](ans, FinalAnswer(parsed))

        return Parser(
            settings=settings, formatting=self.formatting, parse=parse
        )

    @property
    def response(self) -> "GenericParser":
        """
        Wrap answers into full `Response` objects.

        Return a `GenericParser` so that the list of supported tools can
        be extracted from the query's answer type.
        """

        def parser(annot: TypeAnnot[Any], /) -> Parser[Any]:
            assert typing.get_origin(annot) is Response, (
                f"Response type expected: {annot}"
            )
            args = typing.get_args(annot)
            assert len(args) == 2
            return self.response_with(args[1])

        return GenericParser(parser)

    @property
    def trim(self: "Parser[str]") -> "Parser[str]":
        """
        Trim the output of a string parser.
        """
        return self.map(str.strip)

    @property
    def json(self: "Parser[str]") -> "GenericParser":
        """
        Parse a string as a JSON object.

        Return a `GenericParser` so that the target type can be
        extracted from the query's answer type.
        """

        return GenericParser(self.json_as)

    @property
    def yaml(self: "Parser[str]") -> "GenericParser":
        """
        Parse a string as a YAML object.

        Return a `GenericParser` so that the target type can be
        extracted from the query's answer type.
        """

        return GenericParser(self.yaml_as)

    def json_as[U](self: "Parser[str]", type: TypeAnnot[U]) -> "Parser[U]":
        """
        Parse a string as a JSON object.

        !!! info
            This method currently does not work very well with type
            inference since its arguments do not allow inferring the
            type of `U`. This should work better once `TypeAnnot` can be
            replaced with `TypeExpr` (incoming in Python 3.14).
        """

        _assert_not_response_type(type, where="json_as")
        schema = md.Schema.make(type)
        return self.map(partial(_parse_json_as, type)).update_formatting(
            lambda f: replace(f, what="json", schema=schema)
        )

    def yaml_as[U](self: "Parser[str]", type: TypeAnnot[U]) -> "Parser[U]":
        """
        Parse a string as a YAML object.

        !!! info
            This method currently does not work very well with type
            inference since its arguments do not allow inferring the
            type of `U`. This should work better once `TypeAnnot` can be
            replaced with `TypeExpr` (incoming in Python 3.14).
        """
        _assert_not_response_type(type, where="yaml_as")
        schema = md.Schema.make(type)
        return self.map(partial(_parse_yaml_as, type)).update_formatting(
            lambda f: replace(f, what="yaml", schema=schema)
        )


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

        def parser(annot: TypeAnnot[Any], /) -> Parser[Any]:
            comps = dpi.union_components(annot)
            assert len(comps) >= 2 and WrappedParseError in comps, (
                "Answer type does not have shape `... | WrappedParseError`: "
                + f"{annot}"
            )
            annot = dpi.make_union(
                [c for c in comps if c != WrappedParseError]
            )
            return self.for_type(annot).wrap_errors

        return GenericParser(parser)

    @property
    def response(self) -> "GenericParser":
        """
        Wrap answers into full `Response` objects.

        Possible tool calls are extracted from the query's answer type
        and an exception is raised if this type does not have the form
        `Response[..., ...]`.
        """

        def parser(annot: TypeAnnot[Any], /) -> Parser[Any]:
            assert typing.get_origin(annot) is Response, (
                f"Response type expected: {annot}"
            )
            args = typing.get_args(annot)
            assert len(args) == 2
            return self.for_type(args[0]).response_with(args[1])

        return GenericParser(parser)


class _GenericParserFn(Protocol):
    """
    Type of functions wrapped by `GenericParser`.
    """

    def __call__[T](self, type: TypeAnnot[T], /) -> Parser[T]: ...


def structured_as[T](type: TypeAnnot[T], /) -> Parser[T]:
    """
    Parse an LLM structured answer into a given target type.

    !!! warning
        Only dataclass types are supported, since most LLM providers
        only support structured output and tool calls for JSON objects.
    """
    _assert_not_response_type(type, where="structured_as")
    _check_valid_structured_output_type(type)
    settings = dp.QuerySettings(dp.StructuredOutputSettings(type))
    formatting = FormattingMetadata(
        where="full_answer", what="json", schema=md.Schema.make(type)
    )
    return Parser(
        settings, formatting, lambda ans: _parse_structured_output(type, ans)
    )


def _assert_not_response_type(annot: TypeAnnot[Any], *, where: str) -> None:
    if annot is Response or typing.get_origin(annot) is Response:
        raise ValueError(
            f"Unexpected target type for `{where}`: {annot}.\n"
            + "Did you forget to append `.response` to your parser definition?"
        )


def final_tool_call_as[T](annot: TypeAnnot[T], /) -> Parser[T]:
    """
    Variant of `structured_as`, where the query answer type is presented
    to oracles as a tool, which must be called to produce the final
    answer. This provides an alternative to "structured", which
    additionally allows a chain of thoughts to precede the final answer.

    !!! warning
        Only dataclass types are supported, since most LLM providers
        only support structured output and tool calls for JSON objects.
    """
    _check_valid_structured_output_type(annot)
    assert isinstance(annot, type)  # redundant with previous check
    tool = cast(type[Any], annot)
    tool_settings = dp.ToolSettings(tool_types=[tool], force_tool_call=True)
    settings = dp.QuerySettings(None, tool_settings)
    formatting = FormattingMetadata(
        where="tool_call", what="json", schema=md.Schema.make(annot)
    )

    def parse(ans: dp.Answer) -> T:
        assert len(ans.tool_calls) == 1, (
            f"Expected one final tool call, got answer: {ans}"
        )
        return _parse_or_raise(tool, ans.tool_calls[0].args)

    return Parser(settings, formatting, parse)


structured = GenericParser(structured_as)
"""
Generic parser associated with `structured_as`.
"""


final_tool_call = GenericParser(final_tool_call_as)
"""
Generic parser associated with `final_tool_call_as`.
"""


def _get_text_answer(ans: Answer) -> str:
    _assert_no_tool_calls(ans)
    if not isinstance(ans.content, str):
        raise dp.ParseError(description="Unexpected structured answer.")
    return ans.content


get_text = Parser[str](
    dp.QuerySettings(),
    FormattingMetadata(where="full_answer", what="text"),
    _get_text_answer,
)
"""
Parser that extracts the text content of an answer.

A runtime error is raised if the answer contains structured content.
"""


last_code_block: Parser[str] = get_text.map(
    lambda s: block
    if (block := extract_final_block(s)) is not None
    else dp.ParseError(description="No code block found.", meta={"source": s})
).update_formatting(lambda f: replace(f, where="last_code_block"))
"""
Parser that extracts the last code block from a text answer.
"""


type ParserDict = dict[dp.AnswerMode, Parser[Any] | GenericParser]
"""
A mapping from answer modes to parser specifications.

Can be used as a value for the `__parser__` class attribute of queries.
"""


def _check_valid_structured_output_type(annot: TypeAnnot[Any]) -> None:
    if orig := typing.get_origin(annot):
        annot = orig
    forbidden = [str, int, float, bool, dict, tuple]
    if not isinstance(annot, type) or annot in forbidden:
        raise ValueError(
            f"Structured output not supported for type: {annot}.\n"
            "Most LLM providers only support structured output for JSON "
            "objects. Consider wrapping your output type in a custom class."
        )


##### Parser Utilities


def _parse_yaml_as[T](type: TypeAnnot[T], ans: str, /) -> T:
    import yaml

    try:
        parsed = yaml.safe_load(ans)
        return ty.pydantic_load(type, parsed)
    except ValidationError as e:
        raise dp.ParseError(description=str(e))
    except Exception as e:
        raise dp.ParseError(description=str(e))


def _parse_json_as[T](type: TypeAnnot[T], ans: str, /) -> T:
    import json

    try:
        parsed = json.loads(ans)
        return ty.pydantic_load(type, parsed)
    except ValidationError as e:
        raise dp.ParseError(description=str(e))
    except Exception as e:
        raise dp.ParseError(description=str(e))


def _parse_structured_output[T](type: TypeAnnot[T], answer: dp.Answer) -> T:
    _assert_no_tool_calls(answer)
    if not isinstance(answer.content, dp.Structured):
        raise dp.ParseError(
            description="A structured output was expected.",
        )
    return _parse_or_raise(type, answer.content.structured)


def _parse_or_raise[T](type: TypeAnnot[T], obj: Any) -> T:
    try:
        return ty.pydantic_load(type, obj)
    except ValidationError as e:
        raise dp.ParseError(description=str(e))


def _assert_no_tool_calls(ans: dp.Answer) -> None:
    if ans.tool_calls:
        raise dp.ParseError(
            description="Unexpected tool calls.",
            meta={"tool_calls": ans.tool_calls},
        )


def extract_final_block(s: str) -> str | None:
    # In case the output is ill-formed, the quotes may not be balanced.
    # This is why we use a lookahead here.
    # See tests in `test_stdlib.py`
    code_blocks = re.findall(r"(?=(```[^\n]*\n(.*?)```))", s, re.DOTALL)
    return code_blocks[-1][1] if code_blocks else None


def _first_word[T](type: TypeAnnot[T]) -> Parser[T]:
    """
    See `first_word`.
    """

    def process(type: TypeAnnot[T], res: str) -> T:
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

    return get_text.map(partial(process, type)).update_formatting(
        lambda f: replace(f, what="one_word")
    )


first_word = GenericParser(_first_word)
"""
Parse the first word of the answer and turn it into an object of
type `T = Literal[s1,...,sn]`.
"""


class QueryTemplateArgs(typing.TypedDict):
    """
    Template arguments passed to all query templates.

    For particular kinds of templates, additional arguments may be
    provided (e.g., `feedback` for feedback prompts).

    Attributes:
        query: The query instance.
        mode: The requested answer mode. In a multi-message chat with
            few-shot examples, this variable can have different values
            across examples. Also, it has the same value for the system
            prompt and the final instance prompt.
        available_modes: The sequence of all available answer modes for
            the query type.
        params: The query hyperparameters (e.g., as passed to `few_shot`)
        format: Formatting metadata, as derived from `mode` (and whose
            value may therefore differ across examples).
    """

    # TODO: in future Python versions, use `extra_items=Any` (PEP 728)

    query: "Query[Any]"
    mode: dp.AnswerMode
    available_modes: Sequence[dp.AnswerMode]
    params: dict[str, Any]
    format: FormattingMetadata


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

    All attributes of a query must be serializable by pydantic. They can
    be builtin types (int, list, dict...), custom dataclasses...

    ## Customizing Prompts

    System and instance prompts can be specified via Jinja templates.
    The templates manager (`TemplatesManager`) looks for templates named
    "<QueryName>.<instance|system>.jinja". Templates can also be
    provided by defining the `__system_prompt__` and/or
    `__instance_prompt__` class attributes. If none of these are
    provided, the query's docstring is used as a system prompt and
    `DEFAULT_INSTANCE_PROMPT` is used as an instance prompt template.
    All attributes from `QueryTemplateArgs` are made available to
    templates, with possibly extra ones.

    ## Answer Modes and Configurations

    A query can define several answer modes (`AnswerMode`), each of
    which can be associated with a different parser and set of settings.
    By default, the only answer mode is `None`. More answer modes can be
    defined by setting class variable `__modes__`.

    The `parser_for` method maps modes to parser specifications. Its
    default implementation first checks whether the `parser` method is
    overriden, in which case it is used. Otherwise, the `__parser__`
    attribute is checked. If none of these conditions hold, `structured`
    is used as a default parser.

    ## Allowing Multi-Message Exchanges and Tool Calls

    A common pattern for interacting with LLMs is to have multi-message
    exchanges where the full conversation history is resent repeatedly.
    LLMs are also often allowed to request tool calls. This interaction
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
    """

    __modes__: ClassVar[Sequence[dp.AnswerMode] | None] = None
    __parser__: ClassVar[Parser[Any] | GenericParser | ParserDict | None] = (
        None
    )

    ### Parsing Answers

    def parser(self) -> Parser[T] | GenericParser:
        """
        Method to override to provide a parser specification common to
        all modes. Alternatively, the `__parser__` class attribute can
        be set. The first method allows more flexibility since parser
        specifications can then depend on query attributes.
        """
        assert False, (
            "Please provide `__parser__`, `parser` or "
            + f"`parser_for` for query type {type(self)}"
        )

    def parser_for(self, mode: dp.AnswerMode) -> Parser[T] | GenericParser:
        """
        Obtain a parser speficiation for a given answer mode.

        This method can be overriden. By default, it does the following:

        1. If the `parser` method is overriden, it uses it.
        2. If `__parser__` is set as a parser, it is used.
        2. If `__parser__` is set as a dictionary, the mode is used as a
           key to obtain a parser.
        3. Otherwise, `structured` is used as a default parser.
        """
        if dpi.is_method_overridden(Query, type(self), "parser"):
            assert self.__parser__ is None, (
                f"Both `__parser__` and `parser` are provided for {type(self)}."
            )
            return self.parser()
        elif self.__parser__ is None:
            return structured  # default parser
        else:
            assert not dpi.is_method_overridden(
                Query, type(self), "parser_for"
            ), (
                "Both `__parser__` and `parser_for` are "
                + f"provided for {type(self)}."
            )
            parser_attr = self.__parser__
            if isinstance(parser_attr, dict):
                parser = parser_attr[mode]
            else:
                parser = parser_attr
            assert isinstance(parser, (Parser, GenericParser)), (
                "Expected parser type, got: " + f"{type(parser)}."
            )
            return cast(Any, parser)

    def _instantiated_parser_for(self, mode: dp.AnswerMode) -> Parser[T]:
        parser = self.parser_for(mode)
        if isinstance(parser, GenericParser):
            return parser.for_type(self._answer_type())
        else:
            return parser

    @override
    def parse_answer(self, answer: dp.Answer) -> T | dp.ParseError:
        assert answer.mode in self.query_modes(), (
            f"Unknown mode: {answer.mode}"
        )
        try:
            parser = self._instantiated_parser_for(answer.mode)
            return parser.parse(answer)
        except dp.ParseError as e:
            return e

    @override
    def query_settings(self, mode: dp.AnswerMode) -> dp.QuerySettings:
        parser = self._instantiated_parser_for(mode)
        return parser.settings

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
        env: dp.AbstractTemplatesManager | None = None,
    ) -> str:
        assert env is not None, _no_prompt_manager_error()
        args_min: QueryTemplateArgs = {
            "query": self,
            "mode": mode,
            "available_modes": self.query_modes(),
            "params": params,
            "format": self._instantiated_parser_for(mode).formatting,
        }
        args: dict[str, object] = {**args_min}
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
        env: PolicyEnv,
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


def _match_string_literal_type(t: Any) -> Sequence[str] | None:
    if (
        (vals := dpi.literal_type_args(t)) is not None
        and len(vals) > 0
        and all(isinstance(v, str) for v in vals)
    ):
        return vals
    return None


#####
##### Example Selectors
#####


type ExampleSelector = Callable[[Sequence[Example]], Sequence[Example]]
"""
A function for selecting a subset of examples from a given sequence.
"""


def select_all_examples(
    examples: Sequence[Example],
) -> Sequence[Example]:
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
        examples: Sequence[Example],
    ) -> Sequence[Example]:
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
        examples: Sequence[Example],
    ) -> Sequence[Example]:
        return [ex for ex in examples if any(t in ex.tags for t in tags)]

    return select


#####
##### Prompting Policies
#####


def fetch_examples(
    database: ExampleDatabase,
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
    env: dp.AbstractTemplatesManager | None,
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
        # What mode we pick does not really matter here since it does
        # not influence hot the assistant message is rendered for LLMs,
        # but we still provide the right mode.
        msgs.append(md.AssistantMessage(Answer(mode, priming)))
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
    env: dp.AbstractTemplatesManager | None,
) -> md.Chat:
    msgs: list[md.ChatMessage] = []
    sys = query.generate_prompt(
        kind="system", mode=mode, params=params, env=env
    )
    msgs.append(md.SystemMessage(sys))
    for q, ans in examples:
        msgs.extend(_instance_prompt(q, env, params, ans.mode))
        msgs.append(md.AssistantMessage(ans))
    msgs.extend(_instance_prompt(query, env, params, mode))
    return tuple(msgs)


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
    answer: dp.Answer, query: dp.AttachedQuery[T], env: PolicyEnv
) -> dp.Tracked[T]:
    parsed = query.parse_answer(answer)
    if isinstance(parsed, dp.ParseError):
        env.info("parse_error", {"error": parsed}, loc=query)
        raise parsed
    return parsed


type _RequestDigest = str


def json_object_digest(obj: Any) -> str:
    import hashlib
    import json

    obj_str = json.dumps(obj).encode("utf-8")
    return hashlib.md5(obj_str).hexdigest()[:8]


def _log_request(
    env: PolicyEnv,
    *,
    query: dp.AttachedQuery[Any],
    request: md.LLMRequest,
):
    req_json = ty.pydantic_dump(md.LLMRequest, request)
    req_digest = json_object_digest(req_json)
    info = {
        "hash": req_digest,
        "query": query.query.query_name(),
        "request": req_json,
    }
    env.info("llm_request", info, loc=query)
    return req_digest


def _log_response(
    env: PolicyEnv,
    *,
    query: dp.AttachedQuery[Any],
    request: md.LLMRequest,
    response: md.LLMResponse,
):
    req_json = ty.pydantic_dump(md.LLMRequest, request)
    req_digest = json_object_digest(req_json)
    info = {
        "request": req_digest,
        "response": ty.pydantic_dump(md.LLMResponse, response),
    }
    if response.usage_info is not None:
        usage = {
            "model": response.model_name,
            "usage": response.usage_info,
        }
        info["usage"] = usage
    env.info("llm_response", info, loc=query)
    for extra in response.log_items:
        meta = {"request": req_digest, "details": extra.metadata}
        env.log(extra.level, extra.message, meta, loc=query)


def _send_request(
    env: PolicyEnv,
    *,
    model: md.LLM,
    query: dp.AttachedQuery[Any],
    request: md.LLMRequest,
) -> dp.StreamContext[md.LLMResponse | SpendingDeclined]:
    _log_request(env, query=query, request=request)
    response = yield from spend_on(
        lambda: (resp := model.send_request(request, env.cache), resp.budget),
        estimate=model.estimate_budget(request),
    )
    if not isinstance(response, SpendingDeclined):
        _log_response(env, query=query, request=request, response=response)
    return response


@prompting_policy
def classify[T](
    query: dp.AttachedQuery[T],
    env: PolicyEnv,
    model: md.LLM,
    params: dict[str, object] | None = None,
    select_examples: Sequence[ExampleSelector] = (),
    mode: dp.AnswerMode = None,
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
    resp = yield from _send_request(env, model=model, request=req, query=query)
    if isinstance(resp, SpendingDeclined):
        return
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
    env: PolicyEnv,
    model: md.LLM,
    *,
    params: dict[str, object] | None = None,
    select_examples: Sequence[ExampleSelector] = (),
    mode: dp.AnswerMode = None,
    temperature: float | None = None,
    num_completions: int = 1,
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
        temperature: The temperature parameter to use with the LLM, as a
            number from 0 to 2.
        num_completions: The number of completions to request for each
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
    assert not iterative_mode or num_completions == 1
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
            num_completions=num_completions,
            options=options,
            tools=tuple(tools),
            structured_output=structured_output,
        )
        resp = yield from _send_request(
            env, model=model, request=req, query=query
        )
        if isinstance(resp, SpendingDeclined):
            return
        if not resp.outputs:
            env.warn("llm_no_output", loc=query)
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
                env.info("parse_error", {"error": element}, loc=query)
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
    env: PolicyEnv,
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
