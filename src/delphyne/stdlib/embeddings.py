"""
Tools to manipulate embeddings
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, final, override

import delphyne.core as dp
import delphyne.stdlib.models as md
from delphyne.utils.yaml import pretty_yaml

#####
##### Base Models and Caching
#####


@dataclass
class EmbeddingResponse:
    model: str
    embeddings: Sequence[Sequence[float]]
    total_tokens: int


@dataclass
class EmbeddingModel(ABC):
    """
    Base class for embedding models.
    """

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def embed(self, batch: Sequence[str]) -> EmbeddingResponse:
        pass

    @abstractmethod
    def spent(self, resp: EmbeddingResponse) -> dp.Budget:
        pass

    @final
    def embed_with_cache(
        self, batch: Sequence[str], cache: md.LLMCache | None
    ) -> EmbeddingResponse:
        req = _encode_embedding_request(batch, self.get_model_name())
        resp = md.fetch_or_answer(
            cache, req, lambda _: _encode_embedding_response(self.embed(batch))
        )
        return _decode_embedding_response(resp)


def _encode_embedding_request(
    batch: Sequence[str], model: str
) -> md.LLMRequest:
    content = pretty_yaml(batch)
    return md.LLMRequest(
        chat=(md.UserMessage(content=content),),
        num_completions=1,
        options={"model": model},
    )


def _encode_embedding_response(resp: EmbeddingResponse) -> md.LLMResponse:
    output = md.LLMOutput(dp.Structured(resp.embeddings))
    return md.LLMResponse(
        [output],
        model_name=resp.model,
        usage_info={"total_tokens": resp.total_tokens},
    )


def _decode_embedding_response(resp: md.LLMResponse) -> EmbeddingResponse:
    content = resp.outputs[0].content
    assert isinstance(content, dp.Structured)
    embeddings = content.structured
    assert resp.model_name is not None
    assert resp.usage_info is not None
    return EmbeddingResponse(
        model=resp.model_name,
        embeddings=embeddings,
        total_tokens=resp.usage_info["total_tokens"],
    )


#####
##### OpenAI-Compatible API
#####


# API Documentation:
# https://platform.openai.com/docs/api-reference/embeddings/create


@dataclass
class OpenAICompatibleEmbeddingModel(EmbeddingModel):
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    dollars_per_token: float | None = None

    @override
    def get_model_name(self) -> str:
        return self.model_name

    @override
    def spent(self, resp: EmbeddingResponse) -> dp.Budget:
        # TODO: handle model classes to register other metrics?
        budget: dict[str, float] = {}
        if self.dollars_per_token is not None:
            budget[md.DOLLAR_PRICE] = (
                self.dollars_per_token * resp.total_tokens
            )
        return dp.Budget(budget)

    @override
    def embed(self, batch: Sequence[str]) -> EmbeddingResponse:
        import openai

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(
            model=self.model_name, input=list(batch)
        )
        embeddings = [data.embedding for data in response.data]
        return EmbeddingResponse(
            model=response.model,
            embeddings=embeddings,
            total_tokens=response.usage.total_tokens,
        )


#####
##### Standard Models
#####


type StandardOpenAIEmbeddingModel = Literal[
    "text-embedding-3-small",
    "text-embedding-3-large",
]


def standard_openai_embedding_model(
    name: StandardOpenAIEmbeddingModel,
) -> EmbeddingModel:
    match name:
        case "text-embedding-3-small":
            price = 0.02 * md.PER_MILLION
        case "text-embedding-3-large":
            price = 0.13 * md.PER_MILLION
    return OpenAICompatibleEmbeddingModel(
        model_name=name, dollars_per_token=price
    )
