"""
Standard interfaces for LLMs
"""

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypedDict

from delphyne.core.streams import Budget

#####
##### Standard LLM Interface
#####


type ChatMessageRole = Literal["user", "system", "assistant"]


class ChatMessage(TypedDict):
    role: ChatMessageRole
    content: str


type Chat = Iterable[ChatMessage]


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
