"""
Utilities to call OpenAI Models
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import openai

from delphyne.core.streams import Budget
from delphyne.stdlib.models import (
    LLM,
    NUM_REQUESTS,
    Chat,
    LLMCallException,
    LLMOutputMetadata,
    RequestOptions,
)

DEFAULT_MODEL = "gpt-4o"


@dataclass
class OpenAIModel(LLM):
    options: dict[str, Any]

    async def send_request(
        self, chat: Chat, num_completions: int, options: RequestOptions
    ) -> tuple[Sequence[str], Budget, LLMOutputMetadata]:
        client = openai.AsyncOpenAI()
        options = self.options | options
        response: Any = await client.chat.completions.create(
            messages=chat,  # type: ignore
            n=num_completions,
            **self.options,
        )
        budget = Budget({NUM_REQUESTS: num_completions})
        answers = [c.message.content for c in response.choices]
        if any(a is None for a in answers):
            raise LLMCallException(f"Some answer was empty: {answers}")
        return cast(Sequence[str], answers), budget, {}


type OpenAIModelName = Literal["gpt-40", "gpt-4o-mini", "gpt-3.5-turbo"]


def openai_model(model: OpenAIModelName | str):
    return OpenAIModel({"model": model})


_TEST_PROMPT: Chat = [
    {"role": "user", "content": "What is the capital of France?"}
]


async def _test_completion():
    model = openai_model("gpt-4o-mini")
    print(await model.send_request(_TEST_PROMPT, 2, {}))


if __name__ == "__main__":
    """
    To run the test: `python -m delphyne.stdlib.openai_models`
    """
    asyncio.run(_test_completion())
