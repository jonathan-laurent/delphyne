"""
Standard queries and building blocks for prompting policies.
"""

import re
from collections.abc import Callable, Sequence
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


class Query[T](dp.AbstractQuery[T]):
    def parse(self, answer: Answer) -> T:
        """
        A more convenient method to override instead of `parse_answer`.

        Raises dp.ParseError
        """
        if isinstance(answer, dp.Structured) and not answer.tool_calls:
            return _parse_structured_output(self.answer_type(), answer)
        cls = type(self)
        __parser__ = "__parser__"
        if hasattr(cls, __parser__):
            parser: Any = getattr(cls, __parser__)
            assert callable(parser)
            parser = cast(Any, parser)
            import inspect

            sig = inspect.signature(parser)
            nargs = len(sig.parameters)
            assert nargs == 1 or nargs == 2
            text = _get_text_answer(answer)
            if nargs == 1:
                # Normal parser
                return parser(text)
            else:
                # Generic parser
                return parser(self.answer_type(), text)
        assert False, f"No {__parser__} attribute found."

    def query_config(self) -> dp.QueryConfig:
        return dp.QueryConfig(
            force_structured_output=(not hasattr(self, "__parser__"))
        )

    def parse_answer(self, answer: dp.Answer) -> T | dp.ParseError:
        try:
            return self.parse(answer)
        except dp.ParseError as e:
            return e

    def globals(self) -> dict[str, object] | None:
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
        return env.prompt(kind, self.name(), args)

    def serialize_args(self) -> dict[str, object]:
        return cast(dict[str, object], ty.pydantic_dump(type(self), self))

    def answer_type(self) -> TypeAnnot[T]:
        return dpi.first_parameter_of_base_class(type(self))

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


def _no_prompt_manager_error() -> str:
    return (
        "Please provide an explicit prompt manager "
        + " or override the `system_prompt` and `instance_prompt` functions."
    )


def _parse_structured_output[T](type: TypeAnnot[T], arg: dp.Structured):
    try:
        return ty.pydantic_load(type, arg.structured)
    except ValidationError as e:
        raise dp.ParseError(description=str(e))


#####
##### Parsers
#####


class GenericParser(Protocol):
    def __call__[T](self, type: TypeAnnot[T], res: str) -> T: ...


def _get_text_answer(ans: Answer) -> str:
    if ans.tool_calls:
        raise dp.ParseError(
            description="Trying to parse answer with tool calls."
        )
    if not isinstance(ans.content, str):
        raise dp.ParseError(
            description="Trying to parse answer with non-string text."
        )
    return ans.content


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


def raw_string[T](typ: TypeAnnot[T], res: str) -> T:
    try:
        if isinstance(typ, type):  # if `typ` is a class
            return typ(res)  # type: ignore
        # TODO: check that `typ` is a string alias
        return res  # type: ignore
    except Exception as e:
        raise dp.ParseError(description=str(e))


def trimmed_raw_string[T](typ: TypeAnnot[T], res: str) -> T:
    return raw_string(typ, res.strip())


def string_from_last_block[T](typ: TypeAnnot[T], res: str) -> T:
    final = extract_final_block(res)
    if final is None:
        raise dp.ParseError(description="No final code block found.")
    return raw_string(typ, final)


def trimmed_string_from_last_block[T](typ: TypeAnnot[T], res: str) -> T:
    final = extract_final_block(res)
    if final is None:
        raise dp.ParseError(description="No final code block found.")
    return trimmed_raw_string(typ, final)


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
    req = md.LLMRequest(prompt, 1, {})
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
