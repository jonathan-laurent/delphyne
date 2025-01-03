"""
Standard interfaces for LLMs
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable
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


type LLMOutputMetadata = dict[str, Any]


type RequestOptions = dict[str, Any]
"""
Extra options to be passed to a request, overriding the model's default
(e.g. temperature).
"""


NUM_REQUESTS_BUDGET = "num_requests"


class LLM(ABC):
    def estimate_budget(self, chat: Chat, options: RequestOptions) -> Budget:
        return Budget({NUM_REQUESTS_BUDGET: 1})

    @abstractmethod
    async def send_request(
        self,
        chat: Chat,
        num_completions: int,
        options: RequestOptions,
    ) -> tuple[Sequence[str], Budget, LLMOutputMetadata]:
        pass


#####
##### Retry Wrapper
#####


@dataclass
class WithRetry(LLM):
    model: LLM
    retry_delays: Sequence[float] = (0.1, 1, 3)

    def estimate_budget(self, chat: Chat, options: RequestOptions) -> Budget:
        return self.model.estimate_budget(chat, options)

    async def send_request(
        self, chat: Chat, num_completions: int, options: RequestOptions
    ) -> tuple[Sequence[str], Budget, LLMOutputMetadata]:
        for retry_delay in [*self.retry_delays, None]:
            try:
                return await self.model.send_request(
                    chat, num_completions, options
                )
            except Exception as e:
                if retry_delay is None:
                    raise LLMCallException(e)
                else:
                    await asyncio.sleep(retry_delay)
        assert False
