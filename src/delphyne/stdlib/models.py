"""
Standard interfaces for LLMs
"""

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import pydantic
from why3py import defaultdict

from delphyne.core.streams import Budget
from delphyne.utils.caching import cache

#####
##### Standard LLM Interface
#####


type ChatMessageRole = Literal["user", "system", "assistant"]


class ChatMessage(TypedDict):
    role: ChatMessageRole
    content: str


type Chat = Sequence[ChatMessage]


@dataclass
class LLMCallException(Exception):
    exn: Exception | str

    def __str__(self) -> str:
        if isinstance(self.exn, Exception):
            return str(self.exn)
        else:
            return self.exn


type LLMOutputMetadata = dict[str, Any]


type RequestOptions = dict[str, Any]
"""
Extra options to be passed to a request, overriding the model's default
(e.g. temperature).
"""


NUM_REQUESTS = "num_requests"


@dataclass
class StreamingNotImplemented(Exception):
    pass


class LLM(ABC):
    def estimate_budget(self, chat: Chat, options: RequestOptions) -> Budget:
        return Budget({NUM_REQUESTS: 1})

    @abstractmethod
    def send_request(
        self,
        chat: Chat,
        num_completions: int,
        options: RequestOptions,
    ) -> tuple[Sequence[str], Budget, LLMOutputMetadata]:
        """
        This function is allowed to raise exceptions.
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
    def send_request(
        self, chat: Chat, num_completions: int, options: RequestOptions
    ) -> tuple[Sequence[str], Budget, LLMOutputMetadata]:
        raise LLMCallException("No model was provided.")


#####
##### Retry Wrapper
#####


@dataclass
class WithRetry(LLM):
    model: LLM
    retry_delays: Sequence[float] = (0.1, 1, 3)

    def estimate_budget(self, chat: Chat, options: RequestOptions) -> Budget:
        return self.model.estimate_budget(chat, options)

    def send_request(
        self, chat: Chat, num_completions: int, options: RequestOptions
    ) -> tuple[Sequence[str], Budget, LLMOutputMetadata]:
        for retry_delay in [*self.retry_delays, None]:
            try:
                return self.model.send_request(chat, num_completions, options)
            except Exception as e:
                if retry_delay is None:
                    raise LLMCallException(e)
                else:
                    time.sleep(retry_delay)
        assert False


#####
##### Caching Wrapper
#####


@dataclass(frozen=True)
class _CachedBaseRequest:
    chat: Chat
    num_completions: int
    options: RequestOptions

    def __hash__(self) -> int:
        return hash(str(self.chat))


@dataclass(frozen=True)
class _CachedRequest:
    request: _CachedBaseRequest
    iter: int

    def stable_repr(self) -> bytes:
        # We define a custom stable hash so that different iterations of the
        # same request are stored within the same bucket.
        adapter = pydantic.TypeAdapter(_CachedBaseRequest)
        return adapter.dump_json(self.request)


type _CachedResponse = tuple[Sequence[str], Budget, LLMOutputMetadata]


@dataclass
class CachedModel(LLM):
    model: LLM
    cache_dir: Path

    def __post_init__(self):
        self.num_seen: dict[_CachedBaseRequest, int] = defaultdict(lambda: 0)

        @cache(dir=self.cache_dir, hash_arg=_CachedRequest.stable_repr)
        def run_request(req: _CachedRequest) -> _CachedResponse:
            base = req.request
            return self.model.send_request(
                base.chat, base.num_completions, base.options
            )

        self.run_request = run_request

    def estimate_budget(self, chat: Chat, options: RequestOptions) -> Budget:
        return self.model.estimate_budget(chat, options)

    def send_request(
        self, chat: Chat, num_completions: int, options: RequestOptions
    ) -> tuple[Sequence[str], Budget, LLMOutputMetadata]:
        base_req = _CachedBaseRequest(chat, num_completions, options)
        self.num_seen[base_req] += 1
        num_seen = self.num_seen[base_req]
        req = _CachedRequest(base_req, num_seen)
        return self.run_request(req)
