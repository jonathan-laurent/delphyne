"""
Utilities to call models through OpenAI-compatible APIs.
"""

import json
from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass
from typing import Any

import openai
import openai.types.chat as ochat
import openai.types.chat.chat_completion as ochatc
import openai.types.chat.completion_create_params as ochatp

from delphyne.core.refs import Structured, ToolCall
from delphyne.core.streams import Budget
from delphyne.stdlib import models as md
from delphyne.utils.yaml import pretty_yaml

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


def translate_logprob_info(
    info: ochatc.ChoiceLogprobs,
) -> Sequence[md.TokenInfo]:
    assert info.content is not None
    ret: list[md.TokenInfo] = []
    for tok in info.content:
        token = md.Token(bytes=tok.bytes, token=tok.token)
        top = [
            (md.Token(bytes=t.bytes, token=t.token), t.logprob)
            for t in tok.top_logprobs
        ]
        ret.append(md.TokenInfo(token, tok.logprob, top))
    return ret


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
            case md.SystemMessage(content=content):
                return {"role": "system", "content": content}
            case md.UserMessage(content=content):
                return {"role": "user", "content": content}
            case md.AssistantMessage(answer=answer):
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
                if isinstance(result, str):
                    content = result
                else:
                    content = pretty_yaml(result.structured)
                return {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": gen.get_id(call),
                }

    return [translate(msg) for msg in chat]


def _strict_schema(schema: Any):
    from copy import deepcopy

    from openai.lib._pydantic import _ensure_strict_json_schema  # type: ignore

    schema = deepcopy(schema)
    _ensure_strict_json_schema(schema, path=(), root=schema)
    return schema


def _chat_response_format(
    structured_output: md.Schema | None,
) -> ochatp.ResponseFormat:
    if structured_output is None:
        return {"type": "text"}
    # We do not use `description` keys.
    resp: ochatp.ResponseFormat = {
        "type": "json_schema",
        "json_schema": {
            "strict": True,
            "name": structured_output.name,
            "schema": _strict_schema(structured_output.schema),
        },
    }
    if structured_output.description is not None:
        resp["json_schema"]["description"] = structured_output.description
    return resp


def _make_chat_tool(
    tool: md.Schema,
) -> ochat.ChatCompletionToolParam:
    ret: ochat.ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": tool.name,
            "parameters": _strict_schema(tool.schema),
            "strict": True,
        },
    }
    if tool.description is not None:
        ret["function"]["description"] = tool.description
    return ret


@dataclass
class OpenAICompatibleModel(md.LLM):
    options: md.RequestOptions
    api_key: str | None = None
    base_url: str | None = None

    def send_request(self, req: md.LLMRequest) -> md.LLMResponse:
        # TODO: better handling of budget beyond `num_requests`

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        options = self.options | req.options
        assert "model" in options, "No model was specified"
        tools = [_make_chat_tool(tool) for tool in req.tools]
        try:
            response: ochat.ChatCompletion = client.chat.completions.create(
                model=options["model"],
                messages=translate_chat(req.chat),
                n=req.num_completions,
                temperature=options.get("temperature", openai.NOT_GIVEN),
                max_completion_tokens=options.get(
                    "max_completion_tokens", openai.NOT_GIVEN
                ),
                logprobs=options.get("logprobs", openai.NOT_GIVEN),
                top_logprobs=options.get("top_logprobs", openai.NOT_GIVEN),
                tools=tools if tools else openai.NOT_GIVEN,
                response_format=_chat_response_format(req.structured_output),
                tool_choice=options.get("tool_choice", openai.NOT_GIVEN),
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
            content = choice.message.content
            if content is None:
                if (msg := choice.message.refusal) is not None:
                    log.append(
                        md.LLMResponseLogItem("error", "llm_refusal", msg)
                    )
                    continue
                if finish_reason != "tool_calls":
                    log.append(md.LLMResponseLogItem("error", "empty_answer"))
                    continue
                content = ""
            if (
                req.structured_output is not None
                and choice.message.tool_calls is None
            ):
                content = Structured(json.loads(content))
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
            logprobs: Sequence[md.TokenInfo] | None = None
            if options.get("logprobs", False):
                assert choice.logprobs
                logprobs = translate_logprob_info(choice.logprobs)
            # returned by DeepSeek for example
            reasoning_content: str | None = getattr(
                choice.message, "reasoning_content", None
            )
            output = md.LLMOutput(
                content=content,
                logprobs=logprobs,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
            )
            outputs.append(output)
        usage: dict[str, Any] | None = None
        if response.usage:
            usage = response.usage.to_dict()
        return md.LLMResponse(outputs, budget, log, response.model, usage)

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
