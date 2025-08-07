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
from delphyne.stdlib.streams import SearchStream, SpendingDeclined, spend_on
from delphyne.utils import typing as ty
from delphyne.utils.typing import TypeAnnot, ValidationError

REPAIR_PROMPT = "repair"
REQUEST_OTHER_PROMPT = "more"
ASSISTANT_PRIMING_STR = "!<assistant>"


#####
##### Response Type
#####


@dataclass
class FinalAnswer[F]:
    final: F


@dataclass
class ToolRequests[T: md.AbstractTool[Any]]:
    tool_calls: Sequence[T]


@dataclass
class Response[F, T: md.AbstractTool[Any]]:
    """
    Answer type for queries that allow follow-ups, giving access to both
    the raw LLM response (to be passed pass in `AnswerPrefix`) and to
    tool eventual tool calls.
    """

    answer: dp.Answer
    parsed: FinalAnswer[F] | ToolRequests[T]


#####
##### Query Configuration
#####


@dataclass(frozen=True)
class QueryConfig:
    parser: "ParserSpec"


type _StandardParserName = Literal["structured", "final_tool_call"]


type ParserSpec = (
    _StandardParserName | GenericTextParser | Callable[[str], Any]
)

type ParserSpecDict = Mapping[dp.AnswerModeName, ParserSpec]


type QueryConfigDict = Mapping[dp.AnswerModeName, QueryConfig]


@dataclass
class _DecomposedResponseType:
    tools: Sequence[type[md.AbstractTool[Any]]]


@dataclass
class _DecomposedAnswerType:
    """
    Represents an answer type (type parameter of `Query`) of the form
    `F` or `Response[F, T1|...|Tn]`: `final` contains `F` and `resp`
    contains the list of all Ti.
    """

    final: TypeAnnot[Any]
    resp: _DecomposedResponseType | None

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


#####
##### Standard Queries
#####


class Query[T](dp.AbstractQuery[T]):
    """
    Base class for queries, which adds convenience features on top of
    `AbstractQuery`, including inferring a lot of information from type
    hints and from the special __parser__ class attribute.
    """

    __modes__: ClassVar[Sequence[dp.AnswerModeName] | None] = None
    __parser__: ClassVar[ParserSpec | ParserSpecDict | None] = None
    __config__: ClassVar[QueryConfig | QueryConfigDict | None] = None

    ### Inspection methods

    def query_config(self, mode: dp.AnswerModeName) -> QueryConfig | None:
        """
        By default, we obtain the configuration by looking for a
        __config__ field, which might be either a single configuration
        or a dictionary mapping modes to configurations. Instead, if
        only the parser is specified, we can use __parser__.
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
        try:
            return self.parse(answer)
        except dp.ParseError as e:
            return e

    def parse(self, answer: Answer) -> T:
        """
        A more convenient method to override instead of `parse_answer`.

        Raises dp.ParseError
        """

        # Decompose the specified answer type
        config = self.query_config(answer.mode)
        assert config is not None
        attr = config.parser
        ans_type = self._decomposed_answer_type()

        # Compute base parsing function `parser`
        parser: _ParsingFunction[Any]
        if attr == "structured":
            parser = _from_structured(ans_type.final)
        elif attr == "final_tool_call":
            assert isinstance(ans_type.final, type)
            parser = _from_final_tool_call(cast(type[Any], ans_type.final))
        else:
            assert callable(attr)
            attr = cast(Callable[..., Any], attr)
            sig = inspect.signature(attr)
            nargs = len(sig.parameters)
            assert nargs == 1 or nargs == 2
            if nargs == 1:
                parser = attr
            else:
                parser = _from_generic_text_parser(attr, ans_type.final)

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
    def query_settings(self, mode: dp.AnswerModeName) -> dp.QuerySettings:
        config = self.query_config(mode)
        assert config is not None
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
        kind: Literal["system", "instance"] | str,
        mode: dp.AnswerModeName,
        params: dict[str, object],
        env: dp.TemplatesManager | None = None,
    ) -> str:
        assert env is not None, _no_prompt_manager_error()
        args: dict[str, object] = {
            "query": self,
            "mode": mode,
            "params": params,
        }
        if (glob := self.globals()) is not None:
            args["globals"] = glob
        return env.prompt(
            kind, self.query_name(), args, self._default_prompt(kind)
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
                return "{{query | yaml(exclude_fields=['prefix']) | trim}}"
            else:
                return "{{query | yaml | trim}}"
        if kind == "system" and (doc := inspect.getdoc(cls)) is not None:
            return doc
        return None

    def globals(self) -> dict[str, object] | None:
        return None

    ### Other Simple Overloads

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

    ### Generating Opaque Spaces

    @overload
    def using(self, get_policy: EllipsisType, /) -> Opaque[IPDict, T]: ...

    @overload
    def using[P2](
        self,
        get_policy: Callable[[P2], pol.PromptingPolicy] | EllipsisType,
        /,
        inner_policy_type: type[P2] | None = None,
    ) -> Opaque[P2, T]: ...

    def using[P](
        self,
        get_policy: Callable[[P], pol.PromptingPolicy] | EllipsisType,
        /,
        inner_policy_type: type[P] | None = None,
    ) -> Opaque[P, T]:
        """
        Turn a strategy instance into an opaque space by providing a
        mapping from the ambient inner policy to a prompting policy.

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
    ) -> SearchStream[T]:
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
    def __call__(self, answer: dp.Answer) -> T: ...


class GenericTextParser(Protocol):
    def __call__[T](self, type: TypeAnnot[T], res: str) -> T: ...


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


def _from_structured[T](type: TypeAnnot[T]) -> _ParsingFunction[T]:
    return lambda answer: _parse_structured_output(type, answer)


def _from_final_tool_call[T](type: type[T]) -> _ParsingFunction[T]:
    return lambda answer: _parse_or_raise(
        type, _get_single_tool_call(answer).args
    )


#####
##### Standard Parsers
#####


def raw_yaml[T](type: TypeAnnot[T], res: str) -> T:
    try:
        parsed = yaml.safe_load(res)
        return ty.pydantic_load(type, parsed)
    except ValidationError as e:
        raise dp.ParseError(description=str(e))
    except Exception as e:
        raise dp.ParseError(description=str(e))


def yaml_from_last_block[T](type: TypeAnnot[T], res: str) -> T:
    final = extract_final_block(res)
    if final is None:
        raise dp.ParseError(description="No final code block found.")
    return raw_yaml(type, final)


def raw_string[T](type_annot: TypeAnnot[T], res: str) -> T:
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
    return raw_string(type, res.strip())


def first_word[T](type: TypeAnnot[T], res: str) -> T:
    """
    Parse the first word of the answer and turn it into an object of
    type T=Literal[s1,...,sn]
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
    final = extract_final_block(res)
    if final is None:
        raise dp.ParseError(description="No final code block found.")
    return raw_string(type, final)


def trimmed_string_from_last_block[T](type: TypeAnnot[T], res: str) -> T:
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


def select_all_examples(
    examples: Sequence[dp.Example],
) -> Sequence[dp.Example]:
    return examples


def select_random_examples(num_examples: int) -> ExampleSelector:
    def select(
        examples: Sequence[dp.Example],
    ) -> Sequence[dp.Example]:
        if num_examples >= len(examples):
            return examples
        selected = random.sample(examples, num_examples)
        return selected

    return select


def select_with_either_tags(tags: Sequence[str]):
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
    mode: dp.AnswerModeName,
):
    msgs: list[md.ChatMessage] = []
    prompt = query.generate_prompt("instance", mode, params, env)
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
                ps = params | {"feedback": elt}
                fmsg = query.generate_prompt("feedback", mode, ps, env)
                msgs.append(md.UserMessage(fmsg))
            else:
                assert isinstance(elt, dp.ToolResult)
                msgs.append(md.ToolMessage(elt.call, elt.result))
    return msgs


def create_prompt(
    query: dp.AbstractQuery[Any],
    examples: Sequence[tuple[dp.AbstractQuery[Any], dp.Answer]],
    params: dict[str, object],
    mode: dp.AnswerModeName,
    env: dp.TemplatesManager | None,
) -> md.Chat:
    msgs: list[md.ChatMessage] = []
    sys = query.generate_prompt("system", mode, params, env)
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
) -> dp.StreamGen[md.LLMResponse | SpendingDeclined]:
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
    mode: dp.AnswerModeName = None,
    enable_logging: bool = True,
    top_logprobs: int = 20,
    temperature: float = 1.0,
    bias: tuple[str, float] | None = None,
) -> dp.Stream[T]:
    """
    Execute a classification query, attaching a probability distribution
    to the attached answer.

    When `bias=(e, p)` is provided, then the final distribution `D` is
    transformed into `(1-p)*D + p*dirac(e)`
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


@prompting_policy
def few_shot[T](
    query: dp.AttachedQuery[T],
    env: dp.PolicyEnv,
    model: md.LLM,
    params: dict[str, object] | None = None,
    select_examples: Sequence[ExampleSelector] = (),
    mode: dp.AnswerModeName = None,
    enable_logging: bool = True,
    temperature: float | None = None,
    num_concurrent: int = 1,
    iterative_mode: bool = False,
    max_requests: int | None = None,
) -> dp.Stream[T]:
    """
    The standard few-shot prompting sequential prompting policy.

    Arguments:
        query: The query to answer.
        env: The policy environment.

    If `iterative_mode` is `False`, then the prompt is always the same
    and different answers are sampled. If `iterative_mode` is `True`,
    everything happens within a single big chat. Every parse error leads
    to some feedback while every correctly parsed answer leads to a
    message inviting the system to generate another different solution.

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
    mode: dp.AnswerModeName = None,
) -> dp.Stream[T]:
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
