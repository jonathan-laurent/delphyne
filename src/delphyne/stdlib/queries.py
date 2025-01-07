"""
Standard queries and building blocks for prompting policies.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self, cast

import yaml
import yaml.parser

import delphyne.core as dp
import delphyne.core.inspect as dpi
from delphyne.stdlib.dsl import prompting_policy
from delphyne.stdlib.models import LLM, Chat, ChatMessage
from delphyne.stdlib.policies import log
from delphyne.utils import typing as ty
from delphyne.utils.typing import TypeAnnot, ValidationError

#####
##### Standard Queries
#####


class Query[T](dp.AbstractQuery[T]):
    @classmethod
    def modes(cls) -> "AnswerModes[T]":
        __answer__ = "__answer__"
        __parser__ = "__parser__"
        if hasattr(cls, __answer__):
            assert not hasattr(cls, __parser__)
            return getattr(cls, __answer__)
        elif hasattr(cls, __parser__):
            assert not hasattr(cls, __answer__)
            return single_parser(getattr(cls, __parser__))
        assert False, (
            "Please define the `modes` method "
            + "or the `__answer__` class attribute."
        )

    def system_prompt(
        self,
        mode: dp.AnswerModeName,
        params: dict[str, object],
        env: dp.TemplatesManager | None = None,
    ) -> str:
        """
        Raises `TemplateNotFound`.
        """
        assert env is not None, _no_prompt_manager_error()
        args = {"query": self, "mode": mode, "params": params}
        return env.prompt("system", self.name(), args)

    def instance_prompt(
        self,
        mode: dp.AnswerModeName,
        params: dict[str, object],
        env: dp.TemplatesManager | None = None,
    ) -> str:
        """
        Raises `TemplateNotFound`.
        """
        assert env is not None, _no_prompt_manager_error()
        args = {"query": self, "mode": mode, "params": params}
        return env.prompt("instance", self.name(), args)

    def serialize_args(self) -> dict[str, object]:
        return cast(dict[str, object], ty.pydantic_dump(type(self), self))

    @classmethod
    def parse(cls, args: dict[str, object]) -> Self:
        return ty.pydantic_load(cls, args)

    def answer_type(self) -> TypeAnnot[T]:
        return dpi.first_parameter_of_base_class(type(self))

    def using[P](
        self,
        get_policy: Callable[[P], dp.PromptingPolicy],
        inner_policy_type: type[P] | None = None,
    ) -> dp.Builder[dp.OpaqueSpace[P, T]]:
        return dp.OpaqueSpace[P, T].from_query(self, get_policy)


def _no_prompt_manager_error() -> str:
    return (
        "Please provide an explicit prompt manager "
        + " or override the `system_prompt` and `instance_prompt` functions."
    )


type AnswerModes[T] = Mapping[dp.AnswerModeName, dp.AnswerMode[T]]
type Modes = AnswerModes[Any]


def single_parser[T](parser: dp.Parser[T]) -> AnswerModes[T]:
    return {None: dp.AnswerMode(parse=parser)}


#####
##### Parsers
#####


def raw_yaml[T](type: TypeAnnot[T], res: str) -> T | dp.ParseError:
    try:
        parsed = yaml.safe_load(res)
        return ty.pydantic_load(type, parsed)
    except ValidationError as e:
        return dp.ParseError(str(e))
    except yaml.parser.ParserError as e:
        return dp.ParseError(str(e))


def yaml_from_last_block[T](type: TypeAnnot[T], res: str) -> T | dp.ParseError:
    final = extract_final_block(res)
    if final is None:
        return dp.ParseError("No final code block found.")
    return raw_yaml(type, final)


def raw_string[T](typ: TypeAnnot[T], res: str) -> T | dp.ParseError:
    if isinstance(typ, type):  # if `typ` is a class
        return typ(res)  # type: ignore
    return res  # type: ignore  # TODO: assuming that `typ` is a string alias


def trimmed_raw_string[T](typ: TypeAnnot[T], res: str) -> T | dp.ParseError:
    return raw_string(typ, res.strip())


def string_from_last_block[T](
    typ: TypeAnnot[T], res: str
) -> T | dp.ParseError:  # fmt: skip
    final = extract_final_block(res)
    if final is None:
        return dp.ParseError("No final code block found.")
    return raw_string(typ, final)


def trimmed_string_from_last_block[T](
    typ: TypeAnnot[T], res: str
) -> T | dp.ParseError:  # fmt: skip
    final = extract_final_block(res)
    if final is None:
        return dp.ParseError("No final code block found.")
    return trimmed_raw_string(typ, final)


def extract_final_block(s: str) -> str | None:
    code_blocks = re.findall(r"```[^\n]*\n(.*?)```", s, re.DOTALL)
    return code_blocks[-1] if code_blocks else None


#####
##### Prompting Policies
#####


def find_all_examples(
    database: dp.ExampleDatabase, query: dp.AbstractQuery[Any]
) -> Sequence[tuple[dp.AbstractQuery[Any], dp.Answer]]:
    raw = database.examples(query.name())
    return [(query.parse(args), ans) for args, ans in raw]


def system_message(content: str) -> ChatMessage:
    return {"role": "system", "content": content}


def user_message(content: str) -> ChatMessage:
    return {"role": "user", "content": content}


def assistant_message(content: str) -> ChatMessage:
    return {"role": "assistant", "content": content}


def create_prompt(
    query: dp.AbstractQuery[Any],
    examples: Sequence[tuple[dp.AbstractQuery[Any], dp.Answer]],
    params: dict[str, object],
    env: dp.TemplatesManager | None = None,
) -> Chat:
    msgs: list[ChatMessage] = []
    msgs.append(system_message(query.system_prompt(None, params, env)))
    for q, ans in examples:
        msgs.append(user_message(q.instance_prompt(ans.mode, params, env)))
        msgs.append(assistant_message(ans.text))
    msgs.append(user_message(query.instance_prompt(None, params, env)))
    return msgs


def log_oracle_answer(
    env: dp.PolicyEnv,
    query: dp.AttachedQuery[Any],
    mode: dp.AnswerModeName,
    answer: str,
    parsed: object,
    meta: dict[str, Any],
):
    info: dict[str, Any] = {"mode": mode, "answer": answer}
    if isinstance(parsed, dp.ParseError):
        info["parse_error"] = parsed.error
    if meta:
        info["meta"] = meta
    log(env, "Answer received", info, loc=query)


@prompting_policy
async def few_shot[T](
    query: dp.AttachedQuery[T],
    env: dp.PolicyEnv,
    model: LLM,
    enable_logging: bool = True,
) -> dp.Stream[T]:
    """
    The standard few-shot prompting sequential prompting policy.

    TODO: have an example limit and randomly sample examples.
    TODO: we are not using any prompt param.
    """
    examples = find_all_examples(env.examples, query.query)
    prompt = create_prompt(query.query, examples, {})
    options: dict[str, Any] = {}
    cost_estimate = model.estimate_budget(prompt, options)
    while True:
        yield dp.Barrier(cost_estimate)
        answers, budget, meta = await model.send_request(prompt, 1, options)
        answer = answers[0]
        element = query.answer(None, answer)
        if enable_logging:
            log_oracle_answer(env, query, None, answer, element, meta)
        yield dp.Spent(budget)
        if not isinstance(element, dp.ParseError):
            yield dp.Yield(element)
