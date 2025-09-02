"""
Standard interface for LLMs
"""

import inspect
import time
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, final, override

import pydantic

import delphyne.core.inspect as dpi
from delphyne.core.refs import Answer, Structured, ToolCall
from delphyne.core.streams import Budget
from delphyne.utils.caching import Cache, CacheMode, load_cache
from delphyne.utils.typing import TypeAnnot, pydantic_dump

#####
##### Tools
#####


# def _lower_snake_case_of_class_name(s: str) -> str:
#     bits = ["_" + c.lower() if c.isupper() else c for c in s]
#     return "".join(bits).lstrip("_")


def tool_name_of_class_name(s: str) -> str:
    # return _lower_snake_case_of_class_name(s)
    return s


class AbstractTool[T]:
    """
    Base class for an LLM tool interface.

    A new tool interface can be added by defining a dataclass `S` that
    inherits `AbstractTool[T]`, with `T` the tool output type. Instances
    of `S` correspond to tool calls, and an actual tool implementation
    maps values of type `S` to values of type `T`.

    A JSON tool specification can be extracted through the
    `tool_name`, `tool_description` and `tool_answer_type` class
    methods. The `render_result` method describes how to render the
    output of a tool implementation, in a way that can be added back as
    a message in a chat history.
    """

    @classmethod
    def tool_name(cls) -> str:
        return tool_name_of_class_name(cls.__name__)

    @classmethod
    def tool_description(cls) -> str | None:
        return inspect.getdoc(cls)

    @classmethod
    def tool_answer_type(cls) -> TypeAnnot[T]:
        return dpi.first_parameter_of_base_class(cls)

    def render_result(self, res: T) -> str | Structured:
        if isinstance(res, str):
            return res
        ans_type = self.tool_answer_type()
        return Structured(pydantic_dump(ans_type, res))


#####
##### Types for LLM Requests
#####


@dataclass(frozen=True)
class SystemMessage:
    role: Literal["system"]  # for deserialization
    content: str

    def __init__(self, content: str):
        # to bypass the frozen dataclass check
        object.__setattr__(self, "role", "system")
        object.__setattr__(self, "content", content)


@dataclass(frozen=True)
class UserMessage:
    role: Literal["user"]
    content: str

    def __init__(self, content: str):
        object.__setattr__(self, "role", "user")
        object.__setattr__(self, "content", content)


@dataclass(frozen=True)
class AssistantMessage:
    role: Literal["assistant"]
    answer: Answer  # Note: the mode does not really matter for the LLM.

    def __init__(self, answer: Answer):
        # to bypass the frozen dataclass check
        object.__setattr__(self, "role", "assistant")
        object.__setattr__(self, "answer", answer)


@dataclass(frozen=True)
class ToolMessage:
    role: Literal["tool"]
    call: ToolCall
    result: str | Structured

    def __init__(self, call: ToolCall, result: str | Structured):
        object.__setattr__(self, "role", "tool")
        object.__setattr__(self, "call", call)
        object.__setattr__(self, "result", result)


type ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage


type Chat = tuple[ChatMessage, ...]
# We specifically require tuples so that Chat is hashable.


class RequestOptions(typing.TypedDict, total=False):
    """
    LLM request options, inspired from the OpenAI chat API.

    All values are optional.

    Attributes:
        model: The name of the model to use for the request.
        reasoning_effort: The reasoning effort to use for the request,
            when applicable (e.g., for GPT-5 or o3).
        tool_choice: How the model should select which tool (or tools)
            to use when generating a response. `none` means the model
            will not call any tool and instead generates a message.
            `auto` means the model can pick between generating a message
            or calling one or more tools. `required` means the model must
            call one or more tools.
        temperature: The temperature to use for sampling, as a value
            between 0 and 2.
        max_completion_tokens: The maximum number of tokens to generate.
        logprobs: Whether to return log probabilities for the generated
            tokens.
        top_logprobs: The number of top log probabilities to return for
            each generated token, as an integer between 0 and 20.

    !!! warning
        Dictionaries of this type should be treated as immutable, since
        they are used as part of the hash of `LLMRequest` objects.
    """

    model: str
    reasoning_effort: Literal["minimal", "low", "medium", "high"]
    tool_choice: Literal["auto", "none", "required"]
    temperature: float
    max_completion_tokens: int
    logprobs: bool
    top_logprobs: int  # from 0 to 20


@dataclass(frozen=True)
class Schema:
    """
    The description of a schema for structured output or tool use.

    Attributes:
        name: Name of the tool or structured output type.
        description: Optional description.
        schema: The JSON schema of the tool or structured output type,
            typically generated using pydantic's `json_schema` method.
    """

    name: str
    description: str | None
    schema: Any

    @staticmethod
    def make(annot: TypeAnnot[Any], /) -> "Schema":
        """
        Build a schema from a Python type annotation
        """
        if isinstance(annot, type):
            if issubclass(annot, AbstractTool):
                name = annot.tool_name()
                description = annot.tool_description()
            else:
                name = tool_name_of_class_name(annot.__name__)
                # For a dataclass, if no docstring is provided,
                # `inspect.getdoc` shows its signature (name, attribute
                # names and types).
                description = inspect.getdoc(cast(Any, annot))
        elif isinstance(annot, typing.TypeAliasType):
            # TODO: we can do better here.
            name = str(annot)
            description = None
        else:
            # Any other type annotation, such as a union.
            name = str(annot)
            description = None
        adapter = pydantic.TypeAdapter(cast(Any, annot))
        return Schema(
            name=name,
            description=description,
            schema=adapter.json_schema(),
        )

    def _hashable_repr(self) -> str:
        # See comment in ToolCall._hashable_repr
        import json

        return json.dumps(self.__dict__, sort_keys=True)

    def __hash__(self) -> int:
        return hash(self._hashable_repr())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Schema):
            return NotImplemented
        return self._hashable_repr() == other._hashable_repr()


#####
##### Types for LLM Answers
#####


type FinishReason = Literal["stop", "length", "content_filter", "tool_calls"]
"""Reason why the LLM stopped generating content."""


@dataclass
class Token:
    """
    A token produced by an LLM.

    Attributes:
        token: String representation of the token.
        bytes: Optional sequence of integers representing the token's
            byte encoding.
    """

    token: str
    bytes: Sequence[int] | None


@dataclass
class TokenInfo:
    """
    Logprob information for a single token.
    """

    token: Token
    logprob: float
    top_logprobs: Sequence[tuple[Token, float]] | None


@dataclass
class LLMOutput:
    """
    A single LLM chat completion.

    Attributes:
        content: The completion content, as a string or as a structured
            object (if structured output was requested).
        tool_calls: A sequence of tool calls made by the model, if any.
        finish_reason: The reason why the model stopped generating
            content.
        logprobs: Optional sequence of token log probabilities, if
            requested.
        reasoning_content: Reasoning chain of thoughts, if provided (the
            DeepSeek API returns reasoning tokens, while the OpenAI API
            generally does not).
    """

    content: str | Structured
    tool_calls: Sequence[ToolCall]
    finish_reason: FinishReason
    logprobs: Sequence[TokenInfo] | None = None
    reasoning_content: str | None = None


#####
##### Standard Budget Keys
#####


type BudgetCategory = Literal[
    "num_requests",
    "num_completions",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "price",
]
"""
Standard metrics to measure LLM inference usage.
"""

NUM_REQUESTS = "num_requests"
NUM_COMPLETIONS = "num_completions"
NUM_INPUT_TOKENS = "input_tokens"
NUM_CACHED_INPUT_TOKENS = "cached_input_tokens"
NUM_OUTPUT_TOKENS = "output_tokens"
DOLLAR_PRICE = "price"


BUDGET_ENTRY_SEPARATOR = "__"


def budget_entry(
    category: BudgetCategory, model_class: str | None = None
) -> str:
    """
    Return a string that can be used as a key in a budget dictionary.
    """
    res = category
    if model_class is not None:
        res = f"{res}{BUDGET_ENTRY_SEPARATOR}{model_class}"
    return res


@dataclass
class ModelPricing:
    dollars_per_input_token: float
    dollars_per_cached_input_token: float
    dollars_per_output_token: float


PER_MILLION = 1e-6


#####
##### Standard LLM Interface
#####


@dataclass
class LLMBusyException(Exception):
    """
    This exception should be raised when an LLM call failed due to a
    timeout or a rate limit error that warrants a retry. In particular,
    it should not be raised for ill-formed requests (those assumptions
    should not be caught) or when the LLM gave a bad answer (in which
    case budget was consumed and should be counted, while errors are
    added into `LLMResponse`).

    See `WithRetry` for adding retrial logic to LLMs.
    """

    exn: Exception

    def __str__(self) -> str:
        return str(self.exn)


@dataclass
class StreamingNotImplemented(Exception):
    pass


@dataclass
class LLMResponseLogItem:
    severity: Literal["info", "warning", "error"]
    message: str
    metadata: Any = None


@dataclass(frozen=True)
class LLMRequest:
    """
    An LLM chat completion request.

    Attributes:
        chat: The chat history.
        num_completions: The number of completions to generate. Note
            that most LLM providers only bill input tokens once,
            regardless of the number of requested completions.
        options: Request options.
        tools: Available tools.
        structured_output: Provide a schema to enable structured output,
            or `None` for disabling it.

    !!! note
        This class is hashable, as needed by `LLMCache`. For soundness,
        it is assumed that `RequestOptions` dictionaries are immutable.
    """

    chat: Chat
    num_completions: int
    options: RequestOptions
    tools: tuple[Schema, ...] = ()
    structured_output: Schema | None = None

    def _hashable_repr(self) -> Any:
        import json

        return (
            self.chat,
            self.num_completions,
            json.dumps(self.options, sort_keys=True),
            self.tools,
            self.structured_output,
        )

    def __hash__(self) -> int:
        return hash(self._hashable_repr())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LLMRequest):
            return NotImplemented
        return self._hashable_repr() == other._hashable_repr()


@dataclass
class LLMResponse:
    """
    Response to an LLM request.

    Attributes:
        outputs: Generated completions.
        budget: Budget consumed by the request.
        log_items: Log items generated while evaluating the request.
        model_name: The name of the model used for the request, which is
            sometimes more detailed than the model name passed in
            `RequestOptions` (e.g., `gpt-4.1-mini-2025-04-14` vs
            `gpt-4.1-mini`).
        usage_info: Additional usage info metadata, in a
            provider-specific format.
    """

    outputs: Sequence[LLMOutput]
    budget: Budget
    log_items: list[LLMResponseLogItem]
    model_name: str | None = None
    usage_info: dict[str, Any] | None = None


class LLM(ABC):
    """
    Base class for an LLM.
    """

    def estimate_budget(self, req: LLMRequest) -> Budget:
        """
        Estimate the budget that is required to process a request.
        """
        return Budget({NUM_REQUESTS: 1, NUM_COMPLETIONS: req.num_completions})

    @abstractmethod
    def add_model_defaults(self, req: LLMRequest) -> LLMRequest:
        """
        Rewrite a request to take model-specific defaults into account.

        A model can carry default values for some of the request fields
        (e.g. the model name). Thus, requests must be processed through
        this function right before they are executed or cached.
        """
        pass

    @abstractmethod
    def _send_final_request(self, req: LLMRequest) -> LLMResponse:
        """
        Core method for processing a request.

        To be overriden by subclasses to implement the core
        functionality of `send_request`. The latter additionally handles
        model=specific defaults and caching.

        This function is allowed to raise exceptions (some
        provider-specific), including `LLMBusyException` for cases where
        retrials may be warranted.
        """
        pass

    def stream_request(
        self, chat: Chat, options: RequestOptions
    ) -> AsyncIterable[str]:
        """
        Stream the text answer to a request.

        This is currently not used but could be leveraged by the VSCode
        extension in the future.
        """
        raise StreamingNotImplemented()

    @final
    def send_request(
        self, req: LLMRequest, cache: "LLMCache | None"
    ) -> LLMResponse:
        """
        Send a request to a model and return the response.

        This function is allowed to raise exceptions (some
        provider-specific), including `LLMBusyException` for cases where
        retrials may be warranted.

        Attributes:
            req: The request to send.
            cache: An optional cache to use for the request.
        """
        if cache is not None:
            self = CachedModel(self, cache)
        full_req = self.add_model_defaults(req)
        return self._send_final_request(full_req)


@dataclass
class DummyModel(LLM):
    """
    A model that always fails to generate completions.

    Used by the `answer_query` command in particular.
    """

    @override
    def add_model_defaults(self, req: LLMRequest) -> LLMRequest:
        return req

    @override
    def _send_final_request(self, req: LLMRequest) -> LLMResponse:
        budget = Budget({NUM_REQUESTS: req.num_completions})
        return LLMResponse(
            outputs=[], budget=budget, log_items=[], model_name="<dummy>"
        )


#####
##### Retry Wrapper
#####


@dataclass
class WithRetry(LLM):
    """
    Retrying with exponential backoff.
    """

    model: LLM
    num_attempts: int = 5
    base_delay_seconds: float = 1.0
    exponential_factor: float = 2.0
    delay_noise: float | None = 0.1

    @override
    def add_model_defaults(self, req: LLMRequest) -> LLMRequest:
        return self.model.add_model_defaults(req)

    @override
    def estimate_budget(self, req: LLMRequest) -> Budget:
        return self.model.estimate_budget(req)

    def retry_delays(self) -> Iterable[float]:
        import random

        acc = self.base_delay_seconds
        for _ in range(self.num_attempts):
            delay = acc
            if self.delay_noise is not None:
                delay += random.uniform(0, self.delay_noise)
            yield delay
            acc *= self.exponential_factor

    @override
    def _send_final_request(self, req: LLMRequest) -> LLMResponse:
        for i, retry_delay in enumerate([*self.retry_delays(), None]):
            try:
                ret = self.model.send_request(req, None)
                if i > 0:
                    ret.log_items.append(
                        LLMResponseLogItem(
                            "info", "successful_retry", {"delay": retry_delay}
                        )
                    )
                return ret
            except LLMBusyException as e:
                if retry_delay is None:
                    raise e
                else:
                    time.sleep(retry_delay)
        assert False


#####
##### Caching Wrapper
#####


@dataclass(frozen=True)
class _CachedRequest:
    request: LLMRequest
    iter: int


@dataclass
class LLMCache:
    """
    A cache for LLM requests.

    More precisely, what are cached are `(r, i)` pairs where `r` is a
    request and `i` is the number of times the request has been answered
    since the model was instantiated. This way, caching works even when
    a policy samples multiple answers for the same request.

    Multiple models can share the same cache.

    `LLMCache` objects can be created using the `load_request_cache`
    context manager.
    """

    cache: Cache[_CachedRequest, LLMResponse]
    num_seen: dict[LLMRequest, int]

    def __init__(self, cache: Cache[_CachedRequest, LLMResponse]):
        self.cache = cache
        self.num_seen: dict[LLMRequest, int] = defaultdict(lambda: 0)


@contextmanager
def load_request_cache(file: Path, *, mode: CacheMode):
    """
    Context manager that loads an LLM request cache from a YAML file.
    """
    with load_cache(
        file, mode=mode, input_type=_CachedRequest, output_type=LLMResponse
    ) as cache:
        yield LLMCache(cache)


@dataclass
class CachedModel(LLM):
    """
    Wrap a model to use a given cache.

    !!! note
        The `LLM.send_request` method has a `cache` argument that can be
        used as a replacement for the `CachedModel` wrapper. In
        addition, all standard prompting policies use a global request
        cache (see `PolicyEnv`) when available. Thus, external users
        should rarely need to manually wrap models with `CachedModel`.
    """

    model: LLM
    cache: LLMCache

    def __post_init__(self):
        @self.cache.cache
        def run_request(req: _CachedRequest) -> LLMResponse:
            base = req.request
            return self.model.send_request(base, None)

        self.run_request = run_request

    @override
    def _send_final_request(self, req: LLMRequest) -> LLMResponse:
        self.cache.num_seen[req] += 1
        num_seen = self.cache.num_seen[req]
        return self.run_request(_CachedRequest(req, num_seen))

    @override
    def estimate_budget(self, req: LLMRequest) -> Budget:
        return self.model.estimate_budget(req)

    @override
    def add_model_defaults(self, req: LLMRequest) -> LLMRequest:
        return self.model.add_model_defaults(req)
