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


class ToolCallIdGenerator:
    def __init__(self):
        self.next_id = 1
        self.calls: dict[ToolCall, int] = {}

    def get_id(self, tool_call: ToolCall) -> int:
        if tool_call in self.calls:
            return self.calls[tool_call]
        else:
            self.calls[tool_call] = self.next_id
            self.next_id += 1
            return self.calls[tool_call]


def translate_chat(chat: Chat) -> Sequence[dict[str, Any]]:
    """
    We translate the chat into the format expected by OpenAI API.

    Unique ids are generated for tool calls.
    """
    gen = ToolCallIdGenerator()
    res: Sequence[dict[str, Any]] = []
    for m in chat:
        msg: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": f"call_{gen.get_id(call)}",
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.args,
                    },
                }
                for call in m.tool_calls
            ]
        res.append(msg)
    return res


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
                messages=translate_chat(chat),  # type: ignore
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
                messages=translate_chat(chat),  # type: ignore
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
