"""
Utilities to call models through OpenAI-compatible APIs.
"""

import json
from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, override

import openai
import openai.types.chat as ochat
import openai.types.chat.chat_completion as ochatc
import openai.types.chat.completion_create_params as ochatp
from openai.types import CompletionUsage

from delphyne.core.refs import Structured, ToolCall
from delphyne.core.streams import Budget
from delphyne.stdlib import models as md
from delphyne.utils.yaml import pretty_yaml


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
                    content = json.dumps(answer.content.structured, indent=2)
                if answer.justification is not None:
                    content += f"\n\n{answer.justification}"
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
    structured_output: md.Schema | None, no_json_schema: bool
) -> ochatp.ResponseFormat:
    if structured_output is None:
        return {"type": "text"}
    elif no_json_schema:
        return {"type": "json_object"}
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


def _base_budget(
    n: int, model_info: md.ModelInfo | None = None
) -> dict[str, float]:
    budget: dict[str, float] = {md.NUM_REQUESTS: 1, md.NUM_COMPLETIONS: n}
    if model_info is not None:
        budget[md.budget_entry("num_requests", model_info)] = 1
        budget[md.budget_entry("num_completions", model_info)] = 1

    return budget


def _compute_spent_budget(
    n: int,
    model_info: md.ModelInfo | None = None,
    pricing: md.ModelPricing | None = None,
    usage: CompletionUsage | None = None,
) -> dict[str, float]:
    budget = _base_budget(n, model_info)

    def add(cat: md.BudgetCategory, value: float):
        budget[md.budget_entry(cat)] = value
        if model_info is not None:
            budget[md.budget_entry(cat, model_info)] = value

    if usage is not None:
        add("input_tokens", usage.prompt_tokens)
        add("output_tokens", usage.completion_tokens)
        if usage.prompt_tokens_details:
            cached = usage.prompt_tokens_details.cached_tokens or 0
            add("cached_input_tokens", cached)
        else:
            cached = 0
        if pricing is not None:
            non_cached = usage.prompt_tokens - cached
            assert non_cached >= 0
            price = (
                pricing.dollars_per_cached_input_token * cached
                + pricing.dollars_per_input_token * non_cached
                + pricing.dollars_per_output_token * usage.completion_tokens
            )
            add("price", price)

    return budget


@dataclass
class OpenAICompatibleModel(md.LLM):
    """
    A Model accessible via an OpenAI-compatible API.

    - `no_json_schema`: if `True`, JSON mode is used for structured
      output instead of JSON Schema. This is useful for providers like
      DeepSeek that do not support structured output with schemas.
    """

    options: md.RequestOptions
    api_key: str | None = None
    base_url: str | None = None
    no_json_schema: bool = False
    model_info: md.ModelInfo | None = None
    pricing: md.ModelPricing | None = None

    @override
    def add_model_defaults(self, req: md.LLMRequest) -> md.LLMRequest:
        return replace(req, options=self.options | req.options)

    @override
    def estimate_budget(self, req: md.LLMRequest) -> Budget:
        return Budget(_base_budget(req.num_completions, self.model_info))

    @override
    def _send_final_request(self, req: md.LLMRequest) -> md.LLMResponse:
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        options = req.options
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
                response_format=_chat_response_format(
                    req.structured_output, self.no_json_schema
                ),
                tool_choice=options.get("tool_choice", openai.NOT_GIVEN),
            )
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            raise md.LLMBusyException(e)
        outputs: list[md.LLMOutput] = []
        log: list[md.LLMResponseLogItem] = []
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
                try:
                    content = Structured(json.loads(content))
                except Exception as e:
                    log.append(
                        md.LLMResponseLogItem(
                            "error",
                            "failed_to_parse_structured_output",
                            metadata={"content": content, "error": str(e)},
                        )
                    )
                    continue  # we skip the answer
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
                    # If we failed to parse a tool call, we skip the answer
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
        budget = Budget(
            _compute_spent_budget(
                req.num_completions,
                self.model_info,
                self.pricing,
                response.usage,
            )
        )
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
