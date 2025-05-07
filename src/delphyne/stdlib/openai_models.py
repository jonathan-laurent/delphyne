"""
Utilities to call OpenAI Models
"""

from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import openai

from delphyne.core.refs import ToolCall
from delphyne.core.streams import Budget
from delphyne.stdlib.models import (
    LLM,
    NUM_REQUESTS,
    Chat,
    FinishReason,
    LLMCallException,
    LLMOutput,
    LLMOutputMetadata,
    RequestOptions,
)

DEFAULT_MODEL = "gpt-4.1"


@dataclass
class OpenAIModel(LLM):
    options: dict[str, Any]

    def send_request(
        self, chat: Chat, num_completions: int, options: RequestOptions
    ) -> tuple[Sequence[LLMOutput], Budget, LLMOutputMetadata]:
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
        outputs: list[LLMOutput] = []
        budget = Budget({NUM_REQUESTS: num_completions})
        for choice in response.choices:
            finish_reason: FinishReason = choice.finish_reason
            message = choice.message.content
            if message is None:
                if finish_reason != "tool_calls":
                    raise LLMCallException("Empty answer.")
                message = ""
            tool_calls: list[ToolCall] = []
            if choice.message.tool_calls is not None:
                for c in choice.message.tool_calls:
                    call = ToolCall(c.function.name, c.function.arguments)
                    tool_calls.append(call)
            output = LLMOutput(
                message=message,
                logprobs=None,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
            )
            outputs.append(output)
        return outputs, budget, {}

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
