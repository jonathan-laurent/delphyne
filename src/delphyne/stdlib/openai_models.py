"""
Utilities to call OpenAI Models
"""

import json
from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass
from typing import Literal

import openai
import openai.types.chat as ochat

from delphyne.core.refs import ToolCall
from delphyne.core.streams import Budget
from delphyne.stdlib import models as md

DEFAULT_MODEL = "gpt-4.1"


class ToolCallIdGenerator:
    """
    Our LLM API does not keep track of tool call IDs (which is a good
    thing since doing otherwise would pollute demonstrations). Thus, we
    need the ability to generate unique tool call IDs on the fly for
    each request.
    """

    def __init__(self):
        self.next_id = 1
        self.calls: dict[ToolCall, int] = {}

    def get_raw_id(self, tool_call: ToolCall) -> int:
        if tool_call in self.calls:
            return self.calls[tool_call]
        else:
            self.calls[tool_call] = self.next_id
            self.next_id += 1
            return self.calls[tool_call]

    def get_id(self, tool_call: ToolCall) -> str:
        return f"call_{self.get_raw_id(tool_call)}"


def translate_chat(
    chat: md.Chat,
) -> Sequence[ochat.ChatCompletionMessageParam]:
    """
    We translate the chat into the format expected by OpenAI API.

    Unique ids are generated for tool calls.
    """
    gen = ToolCallIdGenerator()

    def translate(msg: md.ChatMessage) -> ochat.ChatCompletionMessageParam:
        match msg:
            case md.SystemMessage(content):
                return {"role": "system", "content": content}
            case md.UserMessage(content=content):
                return {"role": "user", "content": content}
            case md.AssistantMessage(answer):
                if isinstance(answer.content, str):
                    content = answer.content
                else:
                    # We serialize the structured answer
                    content = json.dumps(answer.content, indent=2)
                res: ochat.ChatCompletionMessageParam = {
                    "role": "assistant",
                    "content": content,
                }
                if answer.tool_calls:
                    res["tool_calls"] = [
                        {
                            "id": gen.get_id(call),
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.args),
                            },
                        }
                        for call in answer.tool_calls
                    ]
                return res
            case md.ToolMessage(call=call, result=result):
                return {
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": gen.get_id(call),
                }

    return [translate(msg) for msg in chat]


@dataclass
class OpenAIModel(md.LLM):
    options: md.RequestOptions
    api_key: str | None = None
    base_url: str | None = None

    def send_request(self, req: md.LLMRequest) -> md.LLMResponse:
        # TODO: better handling of budget beyond `num_requests`

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        options = self.options | req.options
        assert "model" in options, "No model was specified"
        try:
            response: ochat.ChatCompletion = client.chat.completions.create(
                model=options["model"],
                messages=translate_chat(req.chat),
                n=req.num_completions,
            )
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            raise md.LLMBusyException(e)
        outputs: list[md.LLMOutput] = []
        log: list[md.LLMResponseLogItem] = []
        budget = Budget({md.NUM_REQUESTS: req.num_completions})
        for choice in response.choices:
            finish_reason: md.FinishReason = (
                choice.finish_reason
                if choice.finish_reason != "function_call"
                else "tool_calls"
            )
            message = choice.message.content
            if message is None:
                if finish_reason != "tool_calls":
                    log.append(md.LLMResponseLogItem("error", "empty_answer"))
                    continue
                message = ""
            tool_calls: list[ToolCall] = []
            if choice.message.tool_calls is not None:
                ok = True
                for c in choice.message.tool_calls:
                    try:
                        args = json.loads(c.function.arguments)
                        tool_calls.append(ToolCall(c.function.name, args))
                    except Exception:
                        ok = False
                        log.append(
                            md.LLMResponseLogItem(
                                "error",
                                "failed_to_parse_tool_call:",
                                metadata={"tool_call": c},
                            )
                        )
                if not ok:
                    continue
            output = md.LLMOutput(
                message=message,
                logprobs=None,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
            )
            outputs.append(output)
        return md.LLMResponse(outputs, budget, log)

    async def stream_request(
        self, chat: md.Chat, options: md.RequestOptions
    ) -> AsyncIterable[str]:
        client = openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url
        )
        options = self.options | options
        assert "model" in options, "No model was specified"
        response = await client.chat.completions.create(
            model=options["model"],
            messages=translate_chat(chat),
            stream=True,
        )
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content


type OpenAIModelName = Literal["gpt-4.1", "gpt-4.1-mini"]


def openai_model(model: OpenAIModelName | str):
    return OpenAIModel({"model": model})
