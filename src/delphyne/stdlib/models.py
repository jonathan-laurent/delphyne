"""
Standard interfaces for LLMs
"""

import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import pydantic
from why3py import defaultdict

import delphyne.core.inspect as dpi
from delphyne.core.refs import Answer, Structured, ToolCall
from delphyne.core.streams import Budget
from delphyne.utils.caching import cache
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


type Chat = Sequence[ChatMessage]


class RequestOptions(TypedDict, total=False):
    model: str
    tool_choice: Literal["auto", "none", "required"]


@dataclass(frozen=True)
class Schema:
    """
    The description of a schema for structured output or tool use.

    The `schema` argument is typically generated using
    `json_schema()` from `pydantic`.
    """

    name: str
    description: str | None
    schema: Any

    @staticmethod
    def make(tool: type[object]) -> "Schema":
        if issubclass(tool, AbstractTool):
            name = tool.tool_name()
            description = tool.tool_description()
        else:
            name = tool_name_of_class_name(tool.__name__)
            description = inspect.getdoc(tool)
        adapter = pydantic.TypeAdapter(cast(type[AbstractTool[Any]], tool))
        return Schema(
            name=name,
            description=description,
            schema=adapter.json_schema(),
        )


#####
##### Types for LLM Answers
#####


type FinishReason = Literal["stop", "length", "content_filter", "tool_calls"]


@dataclass
class Token:
    bytes: Sequence[int]
    token: str


@dataclass
class TokenInfo:
    token: Token
    logprob: float
    top_logprobs: Sequence[tuple[Token, float]] | None


@dataclass
class LLMOutput:
    # TODO: structured answers for which we need to parse the content...
    content: str | Structured
    logprobs: Sequence[TokenInfo] | None
    tool_calls: Sequence[ToolCall]
    finish_reason: FinishReason


#####
##### Standard Budget Keys
#####


NUM_REQUESTS = "num_requests"
NUM_INPUT_TOKENS = "input_tokens"
NUM_OUTPUT_TOKENS = "output_tokens"
DOLLAR_PRICE = "price"


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
    If `structured_output` is not None, it must be a type expression
    that can be plugged into a pydantic adapter.
    """

    # TODO: add support for getting tokens.

    chat: Chat
    num_completions: int
    options: RequestOptions
    tools: Sequence[Schema] = ()
    structured_output: Schema | None = None

    def __hash__(self) -> int:
        # LLMRequest needs to be hashable for `CachedModel` to work.
        return hash(repr((self.chat, self.num_completions, self.options)))


@dataclass
class LLMResponse:
    outputs: Sequence[LLMOutput]
    budget: Budget
    log_items: list[LLMResponseLogItem]


class LLM(ABC):
    def estimate_budget(self, req: LLMRequest) -> Budget:
        return Budget({NUM_REQUESTS: req.num_completions})

    @abstractmethod
    def send_request(self, req: LLMRequest) -> LLMResponse:
        """
        This function is allowed to raise exceptions, including
        `LLMBusyException`.
        """
        pass

    def stream_request(
        self, chat: Chat, options: RequestOptions
    ) -> AsyncIterable[str]:
        """
        Streaming is mostly useful for the UI.
        """
        raise StreamingNotImplemented()


@dataclass
class DummyModel(LLM):
    def send_request(self, req: LLMRequest) -> LLMResponse:
        budget = Budget({NUM_REQUESTS: req.num_completions})
        return LLMResponse([], budget, [])


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

    def send_request(self, req: LLMRequest) -> LLMResponse:
        for i, retry_delay in enumerate([*self.retry_delays(), None]):
            try:
                ret = self.model.send_request(req)
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

    def stable_repr(self) -> bytes:
        # We define a custom stable hash so that different iterations of the
        # same request are stored within the same bucket.
        adapter = pydantic.TypeAdapter(LLMRequest)
        return adapter.dump_json(self.request)


@dataclass
class CachedModel(LLM):
    """
    Wrap a model while caching its requests.

    More precisely, what are cached are `(r, i)` pairs where `r` is a
    request and `i` is the number of times the request has been answered
    since the model was instantiated. This way, caching works even when
    a policy samples multiple answers for the same request.

    Important: two models must _not_ share the same cache directory,
    without which their answers would become correlated.
    """

    model: LLM
    cache_dir: Path

    def __post_init__(self):
        self.num_seen: dict[LLMRequest, int] = defaultdict(lambda: 0)

        @cache(dir=self.cache_dir, hash_arg=_CachedRequest.stable_repr)
        def run_request(req: _CachedRequest) -> LLMResponse:
            base = req.request
            return self.model.send_request(base)

        self.run_request = run_request

    def estimate_budget(self, req: LLMRequest) -> Budget:
        return self.model.estimate_budget(req)

    def send_request(self, req: LLMRequest) -> LLMResponse:
        self.num_seen[req] += 1
        num_seen = self.num_seen[req]
        return self.run_request(_CachedRequest(req, num_seen))
