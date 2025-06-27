"""
Standard queries and building blocks for prompting policies.
"""

import inspect
import re
import textwrap
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import yaml

import delphyne.core as dp
import delphyne.core.chats as ct
import delphyne.core.inspect as dpi
import delphyne.stdlib.models as md
from delphyne.core.refs import Answer
from delphyne.stdlib.policies import log, prompting_policy
from delphyne.utils import typing as ty
from delphyne.utils.typing import TypeAnnot, ValidationError

REPAIR_PROMPT = "repair"
REQUEST_OTHER_PROMPT = "more"
ASSISTANT_PRIMING_STR = "!<assistant>"


#####
##### Standard Queries
#####


type _StandardParserName = Literal["structured", "final_tool_call"]


type ParserSpec = (
    _StandardParserName | GenericTextParser | Callable[[str], Any]
)
"""
Queries can have a `__parser__` attribute, from which the `parse`,
`query_tools` and `query_config` methods are deduced.
"""


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


@dataclass
class _DecomposedResponseType:
    tools: Sequence[type[md.AbstractTool[Any]]]


@dataclass
class _DecomposedAnswerType:
    resp: _DecomposedResponseType | None
    final: TypeAnnot[Any]

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


class Query[T](dp.AbstractQuery[T]):
    """
    Base class for queries, which adds convenience features on top of
    `AbstractQuery`, including inferring a lot of information from type
    hints and from the special __parser__ class attribute.
    """

    @classmethod
    def _parser_attribute(cls) -> ParserSpec | None:
        parse_overriden = dpi.is_method_overridden(Query, cls, "parse")
        if hasattr(cls, "__parser__"):
            assert not parse_overriden
            return getattr(cls, "__parser__")
        return "structured" if not parse_overriden else None

    @classmethod
    def _decomposed_answer_type(cls) -> _DecomposedAnswerType:
        return _DecomposedAnswerType(cls._answer_type())

    def query_tools(self) -> Sequence[type[Any]]:
        ans_type = self._decomposed_answer_type()
        tools: list[type[Any]] = []
        if ans_type.resp is not None:
            tools = [*ans_type.resp.tools]
        if self._parser_attribute() == "final_tool_call":
            assert isinstance(ans_type.final, type)
            tools.append(ans_type.final)
        return tools

    def parse(self, answer: Answer) -> T:
        """
        A more convenient method to override instead of `parse_answer`.

        Raises dp.ParseError
        """

        # Decompose the specified answer type
        attr = self._parser_attribute()
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

    def query_config(self) -> dp.QueryConfig:
        attr = self._parser_attribute()
        return dp.QueryConfig(
            force_structured_output=(attr == "structured"),
            force_tool_call=(attr == "final_tool_call"),
        )

    @classmethod
    def _has_special_prefix_attr(cls):
        annots = typing.get_type_hints(cls)
        return "prefix" in annots and annots["prefix"] is ct.AnswerPrefix

    def finite_answer_set(self) -> Sequence[dp.Answer] | None:
        # We handle the special case where the return type is a literal
        # type that is a subtype of str.
        ans = self.answer_type()
        if (res := _match_string_literal_type(ans)) is not None:
            return [dp.Answer(None, v) for v in res]
        return None

    def query_prefix(self) -> ct.AnswerPrefix | None:
        """
        Return the value of the `prefix` attribute if it has type
        annotation `AnswerPrefix` or return `None`.
        """
        if self._has_special_prefix_attr():
            return getattr(self, "prefix")
        return None

    def parse_answer(self, answer: dp.Answer) -> T | dp.ParseError:
        try:
            return self.parse(answer)
        except dp.ParseError as e:
            return e

    def globals(self) -> dict[str, object] | None:
        return None

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
        return env.prompt(kind, self.name(), args, self._default_prompt(kind))

    def serialize_args(self) -> dict[str, object]:
        return cast(dict[str, object], ty.pydantic_dump(type(self), self))

    @classmethod
    def _answer_type(cls) -> TypeAnnot[T]:
        return dpi.first_parameter_of_base_class(cls)

    def answer_type(self) -> TypeAnnot[T]:
        return self._answer_type()

    def using[P](
        self,
        get_policy: Callable[[P], dp.AbstractPromptingPolicy],
        inner_policy_type: type[P] | None = None,
    ) -> dp.OpaqueSpaceBuilder[P, T]:
        return dp.OpaqueSpace[P, T].from_query(self, get_policy)

    # EXPERIMENTAL: a shorter, call-based syntax.
    def __call__[P](
        self,
        inner_policy_type: type[P],
        get_policy: Callable[[P], dp.AbstractPromptingPolicy],
    ) -> dp.OpaqueSpaceBuilder[P, T]:
        return self.using(get_policy, inner_policy_type)

    def run_toplevel(
        self,
        env: dp.PolicyEnv,
        policy: dp.AbstractPromptingPolicy,
    ) -> dp.Stream[T]:
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


def raw_string[T](type: TypeAnnot[T], res: str) -> T:
    try:
        if isinstance(type, __builtins__.type):  # if `type` is a class
            return type(res)  # type: ignore
        # TODO: check that `type` is a string alias
        return res  # type: ignore
    except Exception as e:
        raise dp.ParseError(description=str(e))


def trimmed_raw_string[T](type: TypeAnnot[T], res: str) -> T:
    return raw_string(type, res.strip())


def first_word[T](type: TypeAnnot[T], res: str) -> T:
    """
    Parse the first word of the answer and turn it into an object of
    type T=Literal[s1,...,sn]
    """
    vals = _match_string_literal_type(type)
    assert vals is not None
    try:
        assert res
        first = res.split()[0]
        assert first in vals
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
    code_blocks = re.findall(r"```[^\n]*\n(.*?)```", s, re.DOTALL)
    return code_blocks[-1] if code_blocks else None


#####
##### Prompting Policies
#####


def find_all_examples(
    database: dp.ExampleDatabase,
    query: dp.AbstractQuery[Any],
) -> Sequence[tuple[dp.AbstractQuery[Any], dp.Answer]]:
    raw = database.examples(query.name(), query.serialize_args())
    return [(query.parse_instance(args), ans) for args, ans in raw]


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
class LogProbInfo:
    """
    Note: The distribution may not sum up to 1 and even be empty. The
    user can normalize it if needed.
    """

    logprobs: dict[dp.Answer, float]


@prompting_policy
def classify[T](
    query: dp.AttachedQuery[T],
    env: dp.PolicyEnv,
    model: md.LLM,
    mode: dp.AnswerModeName = None,
    enable_logging: bool = True,
    top_logprobs: int = 20,
) -> dp.Stream[T]:
    env.tracer.trace_query(query.ref)
    examples = find_all_examples(env.examples, query.query)
    mngr = env.templates
    prompt = create_prompt(query.query, examples, {}, mode, mngr)
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
    cost_estimate = model.estimate_budget(req)
    yield dp.Barrier(cost_estimate)
    resp = model.send_request(req)
    log_oracle_response(env, query, req, resp, verbose=enable_logging)
    yield dp.Spent(resp.budget)
    if not resp.outputs:
        return
    output = resp.outputs[0]
    answer = dp.Answer(mode, output.content)
    element = query.parse_answer(answer)
    env.tracer.trace_answer(query.ref, answer)
    if isinstance(element, dp.ParseError):
        log(env, "parse_error", {"error": element}, loc=query)
    else:
        lpinfo = output.logprobs
        assert lpinfo is not None
        ldistr = _compute_value_distribution(vals, lpinfo[0])
        meta = LogProbInfo(
            {dp.Answer(mode, v): lp for v, lp in ldistr.items()}
        )
        # TODO: add metadata
        yield dp.Yield(element, meta=meta)


@prompting_policy
def few_shot[T](
    query: dp.AttachedQuery[T],
    env: dp.PolicyEnv,
    model: md.LLM,
    mode: dp.AnswerModeName = None,
    enable_logging: bool = True,
    temperature: float | None = None,
    num_concurrent: int = 1,
    iterative_mode: bool = False,
    max_requests: int | None = None,
) -> dp.Stream[T]:
    """
    The standard few-shot prompting sequential prompting policy.

    If `iterative_mode` is `False`, then the prompt is always the same
    and different answers are sampled. If `iterative_mode` is `True`,
    everything happens within a single big chat. Every parse error leads
    to some feedback while every correctly parsed answer leads to a
    message inviting the system to generate another different solution.

    TODO: Have an example limit and randomly sample examples. TODO: We
    are currently not using prompt params.
    """
    assert not iterative_mode or num_concurrent == 1
    assert max_requests is None or max_requests > 0
    env.tracer.trace_query(query.ref)
    examples = find_all_examples(env.examples, query.query)
    mngr = env.templates
    prompt = create_prompt(query.query, examples, {}, mode, mngr)
    config = query.query.query_config()
    options: md.RequestOptions = {}
    if temperature is not None:
        options["temperature"] = temperature
    structured_output = None
    if config.force_structured_output:
        ans_type = query.query.answer_type()
        structured_output = md.Schema.make(ans_type)
    if config.force_tool_call:
        options["tool_choice"] = "required"
    tools = [md.Schema.make(t) for t in query.query.query_tools()]
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
        cost_estimate = model.estimate_budget(req)
        yield dp.Barrier(cost_estimate)
        resp = model.send_request(req)
        log_oracle_response(env, query, req, resp, verbose=enable_logging)
        yield dp.Spent(resp.budget)
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
                yield dp.Yield(element)
        # In iterative mode, we want to keep the conversation going
        if iterative_mode:
            assert len(elements) == 1 and len(answers) == 1
            element = elements[0]
            if isinstance(element, dp.ParseError):
                try:
                    repair = query.query.generate_prompt(
                        REPAIR_PROMPT,
                        query.query.name(),
                        {"error": element},
                        mngr,
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
                        REQUEST_OTHER_PROMPT, query.query.name(), {}, mngr
                    )
                except dp.TemplateFileMissing:
                    gen_new = "Good! Can you generate a different answer now?"
                new_message = md.UserMessage(gen_new)

            prompt = (*prompt, md.AssistantMessage(answers[0]), new_message)
