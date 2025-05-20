"""
Standard queries and building blocks for prompting policies.
"""

import inspect
import re
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import yaml

import delphyne.core as dp
import delphyne.core.inspect as dpi
import delphyne.stdlib.models as md
from delphyne.core.refs import Answer
from delphyne.stdlib.policies import log, prompting_policy
from delphyne.utils import typing as ty
from delphyne.utils.typing import TypeAnnot, ValidationError

REPAIR_PROMPT = "repair"
REQUEST_OTHER_PROMPT = "more"


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
class FollowUpRequest[T: md.AbstractTool]:
    """
    Object returned by a query when a follow-up is requested for
    answering tool calls.

    Tool calls are provided in `tool_calls` in the order in which they
    are provided in `answer`.
    """

    answer: dp.Answer
    tool_calls: Sequence[T]


@dataclass
class _DecomposedAnswerType:
    """
    Decomposed view of an answer type of the form `A | FollowUpRequest[T]`.
    """

    follow_up_request: bool
    tools: Sequence[type[md.AbstractTool]]
    final: TypeAnnot[Any]

    def __init__(self, annot: TypeAnnot[Any]):
        follow_up_request = False
        tools: list[type[md.AbstractTool]] | None = None
        final_comps: list[TypeAnnot[Any]] = []
        for comp in dpi.union_components(annot):
            if typing.get_origin(comp) is FollowUpRequest:
                assert tools is None
                follow_up_request = True
                args = typing.get_args(comp)
                assert len(args) == 1
                tools_raw = dpi.union_components(args[0])
                tools = [
                    a for a in tools_raw if issubclass(a, md.AbstractTool)
                ]
                assert len(tools) == len(tools_raw)
            else:
                final_comps.append(comp)
        self.follow_up_request = follow_up_request
        self.tools = tools or []
        self.final = dpi.make_union(final_comps)


class Query[T](dp.AbstractQuery[T]):
    """
    Base class for queries, which adds convenience features on top of
    `AbstractQuery`, including inferring a lot of information from type
    hints and from the special __parser__ class attribute.
    """

    @classmethod
    def _parser_attribute(cls) -> ParserSpec:
        if hasattr(cls, "__parser__"):
            return getattr(cls, "__parser__")
        return "structured"

    @classmethod
    def _decomposed_answer_type(cls) -> _DecomposedAnswerType:
        return _DecomposedAnswerType(cls._answer_type())

    def query_tools(self) -> Sequence[type[Any]]:
        ans_type = self._decomposed_answer_type()
        tools = [*ans_type.tools]
        if self._parser_attribute() == "final_tool_call":
            final = ans_type.final
            assert isinstance(final, type)
            tools.append(final)
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

        # Add tool supports if needed
        if ans_type.follow_up_request:
            tcs = [
                _parse_tool_call(ans_type.tools, tc)
                for tc in answer.tool_calls
            ]
            if tcs:
                return cast(T, FollowUpRequest(answer, tcs))
        return parser(answer)

    def query_config(self) -> dp.QueryConfig:
        attr = self._parser_attribute()
        return dp.QueryConfig(
            force_structured_output=(attr == "structured"),
            force_tool_call=(attr == "final_tool_call"),
        )

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
            return res
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
    tools: Sequence[type[md.AbstractTool]], tc: dp.ToolCall
) -> md.AbstractTool:
    for t in tools:
        if tc.name == t.tool_name():
            return _parse_or_raise(t, tc.args)
    raise dp.ParseError(description=f"Unknown tool: {tc.name}.")


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
    __builtins__.type
    try:
        if isinstance(type, __builtins__.type):  # if `type` is a class
            return type(res)  # type: ignore
        # TODO: check that `type` is a string alias
        return res  # type: ignore
    except Exception as e:
        raise dp.ParseError(description=str(e))


def trimmed_raw_string[T](type: TypeAnnot[T], res: str) -> T:
    return raw_string(type, res.strip())


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


def create_prompt(
    query: dp.AbstractQuery[Any],
    examples: Sequence[tuple[dp.AbstractQuery[Any], dp.Answer]],
    params: dict[str, object],
    env: dp.TemplatesManager | None,
) -> md.Chat:
    msgs: list[md.ChatMessage] = []
    msgs.append(
        md.SystemMessage(query.generate_prompt("system", None, params, env))
    )
    for q, ans in examples:
        msgs.append(
            md.UserMessage(
                q.generate_prompt("instance", ans.mode, params, env)
            )
        )
        msgs.append(md.AssistantMessage(ans))
    msgs.append(
        md.UserMessage(query.generate_prompt("instance", None, params, env))
    )
    return msgs


def log_oracle_response(
    env: dp.PolicyEnv,
    query: dp.AttachedQuery[Any],
    req: md.LLMRequest,
    resp: md.LLMResponse,
    *,
    verbose: bool,
):
    if verbose:
        info = {
            "request": ty.pydantic_dump(md.LLMRequest, req),
            "response": ty.pydantic_dump(md.LLMResponse, resp),
        }
        log(env, "llm_response", info, loc=query)
    # TODO: severity
    for extra in resp.log_items:
        log(env, extra.message, extra.metadata, loc=query)


@prompting_policy
def few_shot[T](
    query: dp.AttachedQuery[T],
    env: dp.PolicyEnv,
    model: md.LLM,
    enable_logging: bool = True,
    iterative_mode: bool = False,
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
    examples = find_all_examples(env.examples, query.query)
    mngr = env.templates
    prompt = create_prompt(query.query, examples, {}, mngr)
    config = query.query.query_config()
    structured_output = None
    if config.force_structured_output:
        ans_type = query.query.answer_type()
        assert isinstance(ans_type, type)
        structured_output = md.Schema.make(ans_type)
    req = md.LLMRequest(prompt, 1, {}, structured_output=structured_output)
    cost_estimate = model.estimate_budget(req)
    while True:
        yield dp.Barrier(cost_estimate)
        resp = model.send_request(req)
        log_oracle_response(env, query, req, resp, verbose=enable_logging)
        yield dp.Spent(resp.budget)
        if not resp.outputs:
            log(env, "llm_no_output", loc=query)
            continue
        output = resp.outputs[0]
        answer = dp.Answer(None, output.content, tuple(output.tool_calls))
        element = query.parse_answer(answer)
        env.tracer.trace_answer(query.ref, answer)
        if isinstance(element, dp.ParseError):
            log(env, "parse_error", {"error": element}, loc=query)
        else:
            yield dp.Yield(element)
        # In iterative mode, we want to keep the conversation going
        if iterative_mode:
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
            prompt = (*prompt, md.AssistantMessage(answer), new_message)
