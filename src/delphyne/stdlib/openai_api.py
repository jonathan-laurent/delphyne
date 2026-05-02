"""
Utilities to call models through OpenAI-compatible APIs.
"""

import json
from abc import abstractmethod
from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, override

import openai
import openai.types.chat as ochat
import openai.types.chat.chat_completion as ochatc
import openai.types.chat.completion_create_params as ochatp
import openai.types.responses as oresp
from openai import Omit, omit
from openai.types import CompletionUsage
from openai.types.responses import ResponseUsage
from openai.types.responses import response_input_item_param as oresp_param
from openai.types.shared_params import Reasoning

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
    info: ochatc.ChoiceLogprobs | oresp.ResponseOutputText,
) -> Sequence[md.TokenInfo]:
    """
    Translates the logprob info from OpenAI format.
    Supports both Chat Completions and Responses APIs.
    """
    logprobs = (
        info.content
        if isinstance(info, ochatc.ChoiceLogprobs)
        else info.logprobs
    )
    assert logprobs is not None
    ret: list[md.TokenInfo] = []
    for tok in logprobs:
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
    We translate the chat into the format expected by OpenAI's
    Chat Completions API.

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


def _patch_prefix_items_arrays(schema: Any) -> Any:
    """
    Recursively patch a JSON schema *in place* so that every array with
    `prefixItems` also has `"items": false"`. Such arrays are used to
    represent tuples. This fixes OpenAI's "array schema missing items"
    error.
    """
    if isinstance(schema, dict):
        if (
            schema.get("type") == "array"  # type: ignore
            and "prefixItems" in schema
            and "items" not in schema
        ):
            schema["items"] = {"type": "null"}
        for v in schema.values():  # type: ignore
            _patch_prefix_items_arrays(v)
    elif isinstance(schema, list):
        for v in schema:  # type: ignore
            _patch_prefix_items_arrays(v)
    return schema  # type: ignore


def _strict_schema(schema: Any):
    from copy import deepcopy

    from openai.lib._pydantic import _ensure_strict_json_schema  # type: ignore

    schema = deepcopy(schema)
    _ensure_strict_json_schema(schema, path=(), root=schema)
    _patch_prefix_items_arrays(schema)
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


def _base_budget(n: int, model_class: str | None = None) -> dict[str, float]:
    budget: dict[str, float] = {md.NUM_REQUESTS: 1, md.NUM_COMPLETIONS: n}
    if model_class is not None:
        budget[md.budget_entry("num_requests", model_class)] = 1
        budget[md.budget_entry("num_completions", model_class)] = 1

    return budget


def _usage_details(
    usage: CompletionUsage | ResponseUsage, cat: md.BudgetCategory
) -> int | None:
    """
    Extract the number of tokens of a given category
    from the usage information.

    Supports both Chat Completions and Responses APIs.
    Returns `None` if input token details are not available
    (for cached tokens).
    """
    comp_api = isinstance(usage, CompletionUsage)
    assert comp_api or isinstance(usage, ResponseUsage)
    match cat:
        case "input_tokens":
            return usage.prompt_tokens if comp_api else usage.input_tokens
        case "output_tokens":
            return usage.completion_tokens if comp_api else usage.output_tokens
        case "cached_input_tokens":
            details = (
                usage.prompt_tokens_details
                if comp_api
                else usage.input_tokens_details
            )
            return None if details is None else (details.cached_tokens or 0)
        case _:
            raise ValueError(f"No details for given category: {cat}")


def _compute_spent_budget(
    n: int,
    model_class: str | None = None,
    pricing: md.ModelPricing | None = None,
    usage: CompletionUsage | ResponseUsage | None = None,
) -> dict[str, float]:
    """
    Compute the spent budget for a given usage information.

    Supports both Chat Completions and Responses APIs.
    """
    budget = _base_budget(n, model_class)

    def add(cat: md.BudgetCategory, value: float):
        budget[md.budget_entry(cat)] = value
        if model_class is not None:
            budget[md.budget_entry(cat, model_class)] = value

    if usage is not None:
        input_tokens = _usage_details(usage, "input_tokens")
        output_tokens = _usage_details(usage, "output_tokens")
        assert input_tokens is not None and output_tokens is not None
        add("input_tokens", input_tokens)
        add("output_tokens", output_tokens)
        if (
            cached := _usage_details(usage, "cached_input_tokens")
        ) is not None:
            add("cached_input_tokens", cached)
        else:
            cached = 0
        if pricing is not None:
            non_cached = input_tokens - cached
            assert non_cached >= 0
            price = (
                pricing.dollars_per_cached_input_token * cached
                + pricing.dollars_per_input_token * non_cached
                + pricing.dollars_per_output_token * output_tokens
            )
            add("price", price)

    return budget


@dataclass(kw_only=True)
class StandardModel(md.LLM):
    """
    Base class for standard LLM models (reachable either through
    OpenAI Chat Completions API or OpenAI Responses API).

    Attributes:

        options: the default options to use for requests.
        api_key: the API key to use for authentication.
        base_url: the base URL of the API.
        model_class: an optional identifier for the model class (e.g.,
            "reasoning_large"). When provided, class-specific budget
            metrics are reported, so that resource consumption can be
            tracked separately for different classes of models (e.g.,
            tracking "num_requests__reasoning_large" separately from
            "num_requests__chat_small").
        pricing: pricing information for the model.
        no_json_schema: if `True`, JSON mode is used for structured
            output instead of JSON Schema. This is useful for providers
            like DeepSeek that do not support structured output with
            schemas.
    """

    options: md.RequestOptions
    api_key: str | None = None
    base_url: str | None = None
    no_json_schema: bool = False
    model_class: str | None = None
    pricing: md.ModelPricing | None = None

    @override
    def add_model_defaults(self, req: md.LLMRequest) -> md.LLMRequest:
        return replace(req, options=self.options | req.options)

    @override
    def estimate_budget(self, req: md.LLMRequest) -> Budget:
        return Budget(_base_budget(req.num_completions, self.model_class))

    @override
    @abstractmethod
    def _send_final_request(self, req: md.LLMRequest) -> md.LLMResponse:
        pass


@dataclass(kw_only=True)
class OpenAICompatibleModel(StandardModel):
    """
    A Model accessible via an OpenAI-compatible Chat Completions API.
    """

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
                temperature=options.get("temperature", omit),
                reasoning_effort=options.get("reasoning_effort", omit),
                max_completion_tokens=options.get(
                    "max_completion_tokens", omit
                ),
                logprobs=options.get("logprobs", omit),
                top_logprobs=options.get("top_logprobs", omit),
                tools=tools if tools else omit,
                response_format=_chat_response_format(
                    req.structured_output, self.no_json_schema
                ),
                tool_choice=options.get("tool_choice", omit),
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
                    # We only support function tool calls and not custom
                    # tool calls.
                    assert c.type == "function"
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
                self.model_class,
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


#####
##### OpenAI Responses API Support
#####


@dataclass(frozen=True)
class ReasoningCacheKey:
    """
    Key for caching reasoning items. See `ReasoningMessage` to see what a
    reasoning item is.

    Key includes the `LLMRequest` (which encompasses the chat history prefix,
    model options, etc.) which was sent just before the reasoning items are
    received, as well as the content and tool calls of the `AssistantMessage`
    that accompany the reasoning. The assistant message content and tool calls
    are included here because there can be different assistant answers for
    requests with the exact same chat history.
    In such a case, while processing a chat history that has the same prefix
    as a previously encountered chat history (but a different upcoming
    assistant message), we should not send the cached reasoning items
    that were generated for the previous assistant message.

    For more information see `ReasoningCache`.
    """

    prefix: md.LLMRequest
    answer_content: str | Structured
    tool_calls: tuple[ToolCall, ...]


@dataclass
class ReasoningMessage:
    """
    Represents a single reasoning item that is returned by the Responses API.
    It has no human-readable content, apart from a summary.

    It mirrors `oresp.ResponseReasoningItemParam`. Currently, only the `id`
    field is sufficient for resending reasoning items to the API and
    recovering the LLM state, because we are setting `store=True`, when we
    send a request to the API in `OpenAIResponsesModel._send_final_request`.
    The `encrypted_content` field will be `None` unless we explicitly add a
    "reasoning.encrypted_content" option to the `include` parameter,
    when we send a request to the API.

    For more information see `ReasoningCache`.
    """

    id: str
    summary: Sequence[str]
    encrypted_content: str | None


@dataclass
class ReasoningCache:
    """
    A cache mapping a `ReasoningCacheKey` to a sequence of `ReasoningMessage`s,
    which are the reasoning items returned by an LLM model (accessed through
    Responses API), together with an assistant message in response to a
    request. The assistant message and the request are captured in the
    `ReasoningCacheKey`. Also see that class for details.

    Unlike the Chat Completions API, the Responses API returns reasoning items
    (not human-readable) alongside assistant messages. This allows resending
    these reasoning items to recover the LLM state, which saves costs because
    the LLM can reuse its previous state instead of redoing the reasoning
    for the whole chat history. Although in practice it is almost always one
    reasoning item per per response, there is nothing that prevents the
    case that multiple reasoning items are returned by the LLM in a single
    response, that is why we have `Sequence[ReasoningMessage]`.

    Currently, the internal reasoning state of the LLM can only be persisted
    across multiple tool calls that follow the same user message and not across
    multiple conversation turns between user and assistant. See
    [OpenAI documentation][1] for more information. Nevertheless, we cache
    and send reasoning items for all kinds of chat histories, as there is no
    harm in doing so even if the cache is not hit, and future OpenAI caching
    behavior might change.

    Building a cache instead of adding reasoning items directly to the chat
    history prevents reasoning items from polluting the conversation.

    [1]:https://developers.openai.com/cookbook/examples/responses_api/reasoning_items#caching
    """

    cache: md.Cache[ReasoningCacheKey, Sequence[ReasoningMessage]]


class _ReasoningCacheMiss(Exception):
    pass


def _raise_missing(_: ReasoningCacheKey) -> Sequence[ReasoningMessage]:
    raise _ReasoningCacheMiss()


def translate_chat_for_responses(
    req: md.LLMRequest,
    reasoning_cache: ReasoningCache | None = None,
) -> tuple[list[oresp.ResponseInputItemParam], list[md.LLMResponseLogItem]]:
    """
    Translate a chat into the format expected by the OpenAI Responses API.

    Tool calls are flattened into separate top-level items, unlike the
    Chat Completions API where they are nested inside assistant
    messages. If there is a cache hit for a chat history prefix
    together with a following assistant message, we add reasoning
    items from the cache as separate top-level items as well, just
    like the API expects. See `ReasoningCache` for more details.
    """
    gen = ToolCallIdGenerator()
    input_items: list[oresp.ResponseInputItemParam] = []
    log: list[md.LLMResponseLogItem] = []

    for i, msg in enumerate(req.chat):
        match msg:
            case md.SystemMessage(content=content):
                input_items.append({"role": "system", "content": content})
            case md.UserMessage(content=content):
                input_items.append({"role": "user", "content": content})
            case md.AssistantMessage(answer=answer):
                if isinstance(answer.content, str):
                    content = answer.content
                else:
                    content = json.dumps(answer.content.structured, indent=2)
                if answer.justification is not None:
                    content += f"\n\n{answer.justification}"

                if reasoning_cache is not None:
                    prefix_req = replace(req, chat=req.chat[:i])
                    key = ReasoningCacheKey(
                        prefix=prefix_req,
                        answer_content=answer.content,
                        tool_calls=tuple(answer.tool_calls),
                    )
                    try:
                        # Query the cache for reasoning items
                        cached_items = reasoning_cache.cache(_raise_missing)(
                            key
                        )
                        for cached in cached_items:
                            reasoning_item: oresp.ResponseReasoningItemParam = {
                                "type": "reasoning",
                                "id": cached.id,
                                "summary": [
                                    {"type": "summary_text", "text": s}
                                    for s in cached.summary
                                ],
                                "encrypted_content": cached.encrypted_content,
                            }
                            input_items.append(reasoning_item)
                        log.append(
                            md.LLMResponseLogItem(
                                "info",
                                "reasoning_cache_hit",
                                metadata={
                                    "msg_index_in_chat": i,
                                    "number_of_reasoning_items": len(
                                        cached_items  # almost always 1
                                    ),
                                },
                            )
                        )
                    except _ReasoningCacheMiss:
                        # At this point, we are sure that that there is an
                        # assistant message in the chat history and there is
                        # no reasoning items in the cache for this message,
                        # so we can log a cache miss without worrying about
                        # false positives.
                        log.append(
                            md.LLMResponseLogItem(
                                "info",
                                "reasoning_cache_miss",
                                metadata={"msg_index_in_chat": i},
                            )
                        )
                input_items.append({"role": "assistant", "content": content})
                # Append tool calls as separate items
                for call in answer.tool_calls:
                    call_item: oresp.ResponseFunctionToolCallParam = {
                        "type": "function_call",
                        "name": call.name,
                        "arguments": json.dumps(call.args),
                        "call_id": gen.get_id(call),
                    }
                    input_items.append(call_item)
            case md.ToolMessage(call=call, result=result):
                if isinstance(result, str):
                    content = result
                else:
                    content = pretty_yaml(result.structured)
                output_item: oresp_param.FunctionCallOutput = {
                    "type": "function_call_output",
                    "call_id": gen.get_id(call),
                    "output": content,
                }
                input_items.append(output_item)
    return input_items, log


def _make_responses_tool(
    tool: md.Schema,
) -> oresp.FunctionToolParam:
    """
    Build a Responses API tool specification from a schema.
    """
    ret: oresp.FunctionToolParam = {
        "type": "function",
        "name": tool.name,
        "parameters": _strict_schema(tool.schema),
        "strict": True,
    }
    if tool.description is not None:
        ret["description"] = tool.description
    return ret


def _responses_response_format(
    structured_output: md.Schema | None,
    no_json_schema: bool,  # verbosity: Literal
) -> oresp.ResponseTextConfigParam:
    """
    Build the `text` parameter for a Responses API request
    from a structured output schema.
    """
    if structured_output is None:
        return {"format": {"type": "text"}}
    elif no_json_schema:
        return {"format": {"type": "json_object"}}
    fmt: oresp.ResponseFormatTextJSONSchemaConfigParam = {
        "type": "json_schema",
        "strict": True,
        "name": structured_output.name,
        "schema": _strict_schema(structured_output.schema),
    }
    if structured_output.description is not None:
        fmt["description"] = structured_output.description
    return {"format": fmt}


def _determine_finish_reason(
    response: oresp.Response,
) -> tuple[md.FinishReason | None, list[md.LLMResponseLogItem]]:
    """
    Determines the finish reason for a response obtained from the
    Responses API. Logs errors if the response is not completed.
    """
    log: list[md.LLMResponseLogItem] = []
    match response.status:
        case "completed":
            if any(
                isinstance(item, oresp.ResponseFunctionToolCall)
                for item in response.output
            ):
                return "tool_calls", log
            else:
                return "stop", log
        case "incomplete":
            if response.incomplete_details is not None:
                reason = response.incomplete_details.reason
                if reason == "max_output_tokens":
                    return "length", log
                elif reason == "content_filter":
                    return "content_filter", log
            log.append(
                md.LLMResponseLogItem(
                    "error",
                    "incomplete_response",
                    metadata={
                        "status": response.status,
                        "details": (response.incomplete_details),
                    },
                )
            )
            return "stop", log
        case "failed" | "cancelled" | "in_progress" | "queued" | None:
            log.append(
                md.LLMResponseLogItem(
                    "error",
                    "no_response",
                    metadata={
                        "status": response.status,
                        "error": response.error,
                    },
                )
            )
            return None, log


@dataclass(kw_only=True)
class OpenAIResponsesModel(StandardModel):
    """
    A model accessible via the OpenAI Responses API.

    Attributes:
        use_reasoning_cache: Whether to use a reasoning cache to store and
            resend reasoning items. See `ReasoningCache` for more details.

    If `num_completions` > 1, multiple sequential request are made,
    as the Responses API does not support multiple completions per a
    single request unlike the Chat Completions API.
    """

    use_reasoning_cache: bool = True
    reasoning_cache: ReasoningCache | None = field(init=False, default=None)

    def __post_init__(self):
        if self.use_reasoning_cache:
            self.reasoning_cache = ReasoningCache(
                cache=md.Cache(dict={}, mode="read_write")
            )

    @override
    def _send_final_request(self, req: md.LLMRequest) -> md.LLMResponse:
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        options = req.options
        assert "model" in options, "No model was specified"
        tools = [_make_responses_tool(tool) for tool in req.tools]
        include: list[oresp.ResponseIncludable] = []
        if options.get("logprobs", False):
            include.append("message.output_text.logprobs")
        text_config = _responses_response_format(
            req.structured_output, self.no_json_schema
        )
        reasoning_effort = options.get("reasoning_effort")
        reasoning_config: Reasoning | Omit = (
            {"effort": reasoning_effort}
            if reasoning_effort is not None
            else omit
        )

        all_outputs: list[md.LLMOutput] = []
        all_log: list[md.LLMResponseLogItem] = []
        model_name: str | None = None
        usage_info: ResponseUsage | None = None
        input, log = translate_chat_for_responses(req, self.reasoning_cache)
        all_log.extend(log)

        def _send_single_request() -> oresp.Response:
            try:
                response: oresp.Response = client.responses.create(
                    model=options["model"],
                    input=input,
                    temperature=options.get("temperature", omit),
                    reasoning=reasoning_config,
                    max_output_tokens=options.get(
                        "max_completion_tokens", omit
                    ),
                    top_logprobs=options.get("top_logprobs", omit),
                    tools=tools if tools else omit,
                    text=text_config,
                    tool_choice=options.get("tool_choice", omit),
                    include=include if include else omit,
                    store=True,  # make the API remember sent reasoning id's
                )
            except (openai.RateLimitError, openai.APITimeoutError) as e:
                raise md.LLMBusyException(e)
            return response

        for _ in range(req.num_completions):
            response = _send_single_request()
            output, log = self._parse_response(response, req)
            if output is not None:
                all_outputs.append(output)
            all_log.extend(log)

            # Accumulate budget
            model_name = model_name or response.model
            if response.usage is not None:
                new_usage = response.usage
                if usage_info is None:
                    usage_info = new_usage
                else:
                    usage_info.input_tokens += new_usage.input_tokens
                    usage_info.output_tokens += new_usage.output_tokens
                    usage_info.input_tokens_details.cached_tokens += (
                        new_usage.input_tokens_details.cached_tokens
                    )
                    usage_info.output_tokens_details.reasoning_tokens += (
                        new_usage.output_tokens_details.reasoning_tokens
                    )
                    usage_info.total_tokens += new_usage.total_tokens
        budget = Budget(
            _compute_spent_budget(
                req.num_completions,
                self.model_class,
                self.pricing,
                usage_info,
            )
        )
        return md.LLMResponse(
            all_outputs,
            budget,
            all_log,
            model_name,
            usage_info.to_dict() if usage_info is not None else None,
        )

    def _parse_response(
        self, response: oresp.Response, req: md.LLMRequest
    ) -> tuple[md.LLMOutput | None, list[md.LLMResponseLogItem]]:
        output: md.LLMOutput | None = None
        finish_reason, log = _determine_finish_reason(response)
        if finish_reason is None:
            # response incomplete or not processed
            return (output, log)

        # Collect content and tool calls from output items
        content: str | Structured = ""
        tool_calls: list[ToolCall] = []
        logprobs: list[md.TokenInfo] | None = None
        raw_text: list[str] = []
        reasoning_content: str | None = None
        reasoning_items: list[oresp.ResponseReasoningItem] | None = None
        ok = True

        for item in response.output:
            if isinstance(item, oresp.ResponseOutputMessage):
                for part in item.content:
                    if isinstance(part, oresp.ResponseOutputText):
                        raw_text.append(part.text)
                        if (
                            req.options.get("logprobs", False)
                            and part.logprobs is not None
                        ):
                            if logprobs is None:
                                logprobs = []
                            logprobs.extend(translate_logprob_info(part))
                    else:
                        assert isinstance(part, oresp.ResponseOutputRefusal)
                        log.append(
                            md.LLMResponseLogItem(
                                "error",
                                "llm_refusal",
                                part.refusal,
                            )
                        )
            elif isinstance(item, oresp.ResponseFunctionToolCall):
                try:
                    args = json.loads(item.arguments)
                    tool_calls.append(ToolCall(item.name, args))
                except Exception:
                    ok = False  # response not valid
                    log.append(
                        md.LLMResponseLogItem(
                            "error",
                            "failed_to_parse_tool_call:",
                            metadata={"tool_call": item},
                        )
                    )
            elif isinstance(item, oresp.ResponseReasoningItem):
                if reasoning_items is None:
                    reasoning_items = []
                reasoning_items.append(item)
                if item.content:
                    if reasoning_content is None:
                        reasoning_content = ""
                    reasoning_content += "".join(
                        part.text for part in item.content
                    )  # not returned by OpenAI models
            else:
                # not handling other content like audio or image
                continue
        if ok and raw_text:
            full_text = "".join(raw_text)
            if (
                req.structured_output is not None
                and finish_reason != "tool_calls"
            ):
                try:
                    content = Structured(json.loads(full_text))
                except Exception as e:
                    ok = False  # response not valid
                    log.append(
                        md.LLMResponseLogItem(
                            "error",
                            "failed_to_parse_structured_output",
                            metadata={
                                "content": full_text,
                                "error": str(e),
                            },
                        )
                    )
            else:
                content = full_text
        if not content and not tool_calls:
            # response is empty
            ok = False

        if ok:
            if (
                self.reasoning_cache is not None
                and reasoning_items is not None
            ):
                key = ReasoningCacheKey(
                    prefix=req,
                    answer_content=content,
                    tool_calls=(*tool_calls,),
                )
                reasoning_msgs: list[ReasoningMessage] = []
                for item in reasoning_items:
                    summary_texts: list[str] = []
                    for s in item.summary:
                        summary_texts.append(s.text)

                    reasoning_msgs.append(
                        ReasoningMessage(
                            id=item.id,
                            summary=summary_texts,
                            encrypted_content=item.encrypted_content,
                        )
                    )
                # Cache the received reasoning items
                self.reasoning_cache.cache(lambda _: reasoning_msgs)(key)

            output = md.LLMOutput(
                content=content,
                logprobs=logprobs,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
            )

        return (output, log)
