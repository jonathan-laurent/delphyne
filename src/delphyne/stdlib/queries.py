"""
Standard queries and building blocks for prompting policies.
"""

import re
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, cast

import yaml

import delphyne.core as dp
import delphyne.core.inspect as dpi
from delphyne.stdlib.models import LLM, Chat, ChatMessage
from delphyne.stdlib.policies import log, prompting_policy
from delphyne.utils import typing as ty
from delphyne.utils.typing import TypeAnnot, ValidationError

REPAIR_PROMPT = "repair"
REQUEST_OTHER_PROMPT = "more"


#####
##### Standard Queries
#####


class Query[T](dp.AbstractQuery[T]):
    def parse(self, mode: dp.AnswerModeName | None, answer: str) -> T:
        """
        A more convenient method to override instead of `parse_answer`.

        Raises dp.ParseError
        """
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
            if nargs == 1:
                # Normal parser
                return parser(answer)
            else:
                # Generic parser
                return parser(self.answer_type(), answer)
        assert False, f"No {__parser__} attribute found."

    def parse_answer(self, answer: dp.Answer) -> T | dp.ParseError:
        try:
            return self.parse(answer.mode, answer.text)
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


#####
##### Parsers
#####


class GenericParser(Protocol):
    def __call__[T](self, type: TypeAnnot[T], res: str) -> T: ...


def raw_yaml[T](type: TypeAnnot[T], res: str) -> T:
    try:
        parsed = yaml.safe_load(res)
        return ty.pydantic_load(type, parsed)
    except ValidationError as e:
        raise dp.ParseError(str(e))
    except Exception as e:
        raise dp.ParseError(str(e))


def yaml_from_last_block[T](type: TypeAnnot[T], res: str) -> T:
    final = extract_final_block(res)
    if final is None:
        raise dp.ParseError("No final code block found.")
    return raw_yaml(type, final)


def raw_string[T](typ: TypeAnnot[T], res: str) -> T:
    try:
        if isinstance(typ, type):  # if `typ` is a class
            return typ(res)  # type: ignore
        # TODO: check that `typ` is a string alias
        return res  # type: ignore
    except Exception as e:
        raise dp.ParseError(str(e))


def trimmed_raw_string[T](typ: TypeAnnot[T], res: str) -> T:
    return raw_string(typ, res.strip())


def string_from_last_block[T](typ: TypeAnnot[T], res: str) -> T:
    final = extract_final_block(res)
    if final is None:
        raise dp.ParseError("No final code block found.")
    return raw_string(typ, final)


def trimmed_string_from_last_block[T](typ: TypeAnnot[T], res: str) -> T:
    final = extract_final_block(res)
    if final is None:
        raise dp.ParseError("No final code block found.")
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


def system_message(content: str) -> ChatMessage:
    return ChatMessage("system", content)


def user_message(content: str) -> ChatMessage:
    return ChatMessage("user", content)


def assistant_message(answer: dp.Answer) -> ChatMessage:
    # TODO: how do we integrate tool calls?
    assert not answer.tool_calls
    return ChatMessage("assistant", answer.text)


def create_prompt(
    query: dp.AbstractQuery[Any],
    examples: Sequence[tuple[dp.AbstractQuery[Any], dp.Answer]],
    params: dict[str, object],
    env: dp.TemplatesManager | None,
) -> Chat:
    msgs: list[ChatMessage] = []
    msgs.append(
        system_message(query.generate_prompt("system", None, params, env))
    )
    for q, ans in examples:
        msgs.append(
            user_message(q.generate_prompt("instance", ans.mode, params, env))
        )
        msgs.append(assistant_message(ans))
    msgs.append(
        user_message(query.generate_prompt("instance", None, params, env))
    )
    return msgs


def log_oracle_answer(
    env: dp.PolicyEnv,
    query: dp.AttachedQuery[Any],
    prompt: Chat,
    answer: dp.Answer,
    parsed: object,
    meta: dict[str, Any],
):
    info: dict[str, Any] = {"prompt": prompt, "answer": answer}
    if isinstance(parsed, dp.ParseError):
        info["parse_error"] = parsed.error
    if meta:
        info["meta"] = meta
    log(env, "Answer received", info, loc=query)


@prompting_policy
def few_shot[T](
    query: dp.AttachedQuery[T],
    env: dp.PolicyEnv,
    model: LLM,
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
    options: dict[str, Any] = {}
    cost_estimate = model.estimate_budget(prompt, options)
    while True:
        yield dp.Barrier(cost_estimate)
        try:
            outputs, budget, meta = model.send_request(prompt, 1, options)
        except Exception as e:
            msg = "Exception while querying the LLM. Quitting."
            args = {"error": str(e), "prompt": prompt, "options": options}
            args["model"] = str(model)
            log(env, msg, args, loc=query)
            return
        output = outputs[0]
        answer = dp.Answer(None, output.message, tuple(output.tool_calls))
        element = query.parse_answer(answer)
        env.tracer.trace_answer(query.ref, answer)
        if enable_logging:
            log_oracle_answer(env, query, prompt, answer, element, meta)
        yield dp.Spent(budget)
        if not isinstance(element, dp.ParseError):
            yield dp.Yield(element)
        if iterative_mode:
            if isinstance(element, dp.ParseError):
                try:
                    repair = query.query.generate_prompt(
                        REPAIR_PROMPT,
                        query.query.name(),
                        {"error": element.error},
                        mngr,
                    )
                except dp.TemplateFileMissing:
                    repair = (
                        "Invalid answer. Please consider the following"
                        + f" feedback and try again:\n\n{element.error}"
                    )
                new_message = user_message(repair)
            else:
                try:
                    gen_new = query.query.generate_prompt(
                        REQUEST_OTHER_PROMPT, query.query.name(), {}, mngr
                    )
                except dp.TemplateFileMissing:
                    gen_new = "Good! Can you generate a different answer now?"
                new_message = user_message(gen_new)
            prompt = (*prompt, assistant_message(answer), new_message)
