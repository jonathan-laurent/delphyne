"""
Standard queries and building blocks for prompting policies.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self, cast

import yaml
import yaml.parser

from delphyne.core import environment as en
from delphyne.core import queries as qu
from delphyne.core import trees as tr
from delphyne.core.inspect import first_parameter_of_base_class
from delphyne.core.queries import AbstractQuery, AnswerModeName, ParseError
from delphyne.core.refs import Answer
from delphyne.core.streams import Barrier, Spent, Yield
from delphyne.core.trees import AttachedQuery, OpaqueSpace, PromptingPolicy
from delphyne.stdlib.dsl import prompting_policy
from delphyne.stdlib.models import LLM, Chat, ChatMessage
from delphyne.utils import typing as ty
from delphyne.utils.typing import TypeAnnot, ValidationError

#####
##### Standard Queries
#####


class Query[T](AbstractQuery[T]):
    def system_prompt(
        self,
        mode: AnswerModeName,
        params: dict[str, object],
        env: en.TemplatesManager | None = None,
    ) -> str:
        """
        Raises `TemplateNotFound`.
        """
        assert env is not None, _no_prompt_manager_error()
        args = {"query": self, "mode": mode, "params": params}
        return env.prompt("system", self.name(), args)

    def instance_prompt(
        self,
        mode: AnswerModeName,
        params: dict[str, object],
        env: en.TemplatesManager | None = None,
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
        return first_parameter_of_base_class(type(self))

    def search[P](
        self,
        get_policy: Callable[[P], PromptingPolicy],
        inner_policy_type: type[P] | None = None,
    ) -> tr.Builder[OpaqueSpace[P, T]]:
        return OpaqueSpace[P, T].from_query(self, get_policy)

    def __getitem__[P](
        self,
        get_policy: Callable[[P], PromptingPolicy]
        | tuple[Callable[[P], PromptingPolicy], type[P]],
    ) -> tr.Builder[OpaqueSpace[P, T]]:
        if isinstance(get_policy, tuple):
            return self.search(get_policy[0], inner_policy_type=get_policy[1])
        else:
            return self.search(get_policy)


def _no_prompt_manager_error() -> str:
    return (
        "Please provide an explicit prompt manager "
        + " or override the `system_prompt` and `instance_prompt` functions."
    )


type Modes[T] = Mapping[AnswerModeName, qu.AnswerMode[T]]


#####
##### Parsers
#####


def raw_yaml[T](type: TypeAnnot[T], res: str) -> T | ParseError:
    try:
        parsed = yaml.safe_load(res)
        return ty.pydantic_load(type, parsed)
    except ValidationError as e:
        return ParseError(str(e))
    except yaml.parser.ParserError as e:
        return ParseError(str(e))


def yaml_from_last_block[T](type: TypeAnnot[T], res: str) -> T | ParseError:
    final = extract_final_block(res)
    if final is None:
        return ParseError("No final code block found.")
    return raw_yaml(type, final)


def raw_string[T](typ: TypeAnnot[T], res: str) -> T | ParseError:
    if isinstance(typ, type):  # if `typ` is a class
        return typ(res)  # type: ignore
    return res  # type: ignore  # TODO: assuming that `typ` is a string alias


def trimmed_raw_string[T](typ: TypeAnnot[T], res: str) -> T | ParseError:
    return raw_string(typ, res.strip())


def string_from_last_block[T](
    typ: TypeAnnot[T], res: str
) -> T | ParseError:  # fmt: skip
    final = extract_final_block(res)
    if final is None:
        return ParseError("No final code block found.")
    return raw_string(typ, final)


def trimmed_string_from_last_block[T](
    typ: TypeAnnot[T], res: str
) -> T | ParseError:  # fmt: skip
    final = extract_final_block(res)
    if final is None:
        return ParseError("No final code block found.")
    return trimmed_raw_string(typ, final)


def extract_final_block(s: str) -> str | None:
    code_blocks = re.findall(r"```[^\n]*\n(.*?)```", s, re.DOTALL)
    return code_blocks[-1] if code_blocks else None


#####
##### Prompting Policies
#####


def find_all_examples(
    database: en.ExampleDatabase, query: AbstractQuery[Any]
) -> Sequence[tuple[AbstractQuery[Any], Answer]]:
    raw = database.examples(query.name())
    return [(query.parse(args), ans) for args, ans in raw]


def system_message(content: str) -> ChatMessage:
    return {"role": "system", "content": content}


def user_message(content: str) -> ChatMessage:
    return {"role": "user", "content": content}


def assistant_message(content: str) -> ChatMessage:
    return {"role": "assistant", "content": content}


def create_prompt(
    query: AbstractQuery[Any],
    examples: Sequence[tuple[AbstractQuery[Any], Answer]],
    params: dict[str, object],
    env: en.TemplatesManager | None = None,
) -> Chat:
    msgs: list[ChatMessage] = []
    msgs.append(system_message(query.system_prompt(None, params, env)))
    for q, ans in examples:
        msgs.append(user_message(q.instance_prompt(ans.mode, params, env)))
        msgs.append(assistant_message(ans.text))
    msgs.append(user_message(query.instance_prompt(None, params, env)))
    return msgs


@prompting_policy
async def few_shot[T](
    query: AttachedQuery[T],
    env: en.PolicyEnv,
    model: LLM,
) -> tr.Stream[T]:
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
        yield Barrier(cost_estimate)
        answers, budget, _meta = await model.send_request(prompt, 1, options)
        answer = answers[0]
        element = query.answer(None, answer)
        yield Spent(budget)
        if not isinstance(element, ParseError):
            yield Yield(element)
