"""
Bridge with the OpenAI API.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import openai

from delphyne.core.queries import Message, Prompt, PromptOptions
from delphyne.stdlib.generators import Budget


DEFAULT_MODEL = "gpt-4o"
# DEFAULT_MODEL = "gpt-4o-mini"
# DEFAULT_MODEL = "gpt-3.5-turbo"
LLM_RETRY_DELAYS = [0.1, 1, 3]


def openai_message(m: Message) -> Any:
    return {"role": m.role, "content": m.content}


async def stream_openai_response(prompt: Prompt):
    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model=prompt.options.model or DEFAULT_MODEL,
        messages=[openai_message(m) for m in prompt.messages],
        stream=True,
    )
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def estimate_cost(prompt: Prompt) -> Budget:
    return Budget.spent(num_requests=1)  # TODO


def spent_budget(completion: Any) -> Budget:
    return Budget.spent(num_requests=1)  # TODO


@dataclass
class LLMCallException(Exception):
    exn: Exception


async def execute_prompt(
    prompt: Prompt, n: int
) -> tuple[Sequence[str | None], Budget]:
    for retry_delay in [*LLM_RETRY_DELAYS, None]:
        try:
            client = openai.AsyncOpenAI()
            response = await client.chat.completions.create(
                model=prompt.options.model or DEFAULT_MODEL,
                messages=[openai_message(m) for m in prompt.messages],
                n=n,
            )
            budget = spent_budget(response)
            answers = [c.message.content for c in response.choices]
            return answers, budget
        except Exception as e:
            if retry_delay is None:
                raise LLMCallException(e)
            else:
                await asyncio.sleep(retry_delay)
    assert False


_TEST_PROMPT = Prompt(
    [Message("user", "What is the capital of France?")],
    PromptOptions(model="gpt-4o-mini"),
)


async def test_completion():
    print(await execute_prompt(_TEST_PROMPT, 2))


async def test_stream():
    async for res in stream_openai_response(_TEST_PROMPT):
        print(res)


if __name__ == "__main__":
    # asyncio.run(test_stream())
    asyncio.run(test_completion())
