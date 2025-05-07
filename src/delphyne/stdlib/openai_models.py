"""
Utilities to call OpenAI Models
"""

from collections.abc import AsyncIterable, Sequence
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

DEFAULT_MODEL = "gpt-4.1"


@dataclass
class OpenAIModel(LLM):
    options: dict[str, Any]

    def send_request(
        self, chat: Chat, num_completions: int, options: RequestOptions
    ) -> tuple[Sequence[str], Budget, LLMOutputMetadata]:
        client = openai.OpenAI()
        options = self.options | options
        response = cast(
            Any,
            client.chat.completions.create(
                messages=chat,  # type: ignore
                n=num_completions,
                **self.options,
            ),
        )
        budget = Budget({NUM_REQUESTS: num_completions})
        answers = [c.message.content for c in response.choices]
        if any(a is None for a in answers):
            raise LLMCallException(f"Some answer was empty: {answers}")
        return cast(Sequence[str], answers), budget, {}

    async def stream_request(
        self, chat: Chat, options: RequestOptions
    ) -> AsyncIterable[str]:
        client = openai.AsyncOpenAI()
        options = self.options | options
        response: Any = await cast(
            Any,
            client.chat.completions.create(
                messages=chat,  # type: ignore
                stream=True,
                **self.options,
            ),
        )
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content


type OpenAIModelName = Literal["gpt-4.1", "gpt-4.1-mini"]


def openai_model(model: OpenAIModelName | str):
    return OpenAIModel({"model": model})
